"""
Task Scheduler Module
====================

Manages scheduled tasks for the football prediction system.
Handles data collection, model training, predictions, and maintenance tasks.
"""

import asyncio
import logging
from datetime import datetime, timedelta, time
from typing import Dict, Any, List, Optional, Callable
import schedule
import pytz
from dataclasses import dataclass, field
from enum import Enum
import signal
import sys

from data_collector import DataCollector
from predictor_model import FootballPredictor
from database_manager import DatabaseManager
from cache_manager import CacheManager
from football_api import FootballAPI
from notification_service import NotificationService
from config import Config
from logger_module import setup_logger

logger = setup_logger(__name__)


class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ScheduledTask:
    """Represents a scheduled task."""
    name: str
    function: Callable
    schedule_time: str  # cron-like or specific time
    priority: TaskPriority = TaskPriority.NORMAL
    enabled: bool = True
    max_retries: int = 3
    retry_delay: int = 300  # seconds
    timeout: int = 3600  # seconds
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    status: TaskStatus = TaskStatus.PENDING
    error_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


class TaskScheduler:
    """Manages and executes scheduled tasks."""
    
    def __init__(self, config: Config):
        """
        Initialize the task scheduler.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.tasks: Dict[str, ScheduledTask] = {}
        self.running = False
        self.timezone = pytz.timezone(config.TIMEZONE)
        
        # Initialize services
        self.db_manager = DatabaseManager(config)
        self.cache_manager = CacheManager(config)
        self.football_api = FootballAPI(config)
        self.notification_service = NotificationService(config)
        
        # Initialize components
        self.data_collector = DataCollector(
            self.db_manager,
            self.football_api,
            self.cache_manager,
            config
        )
        self.predictor = FootballPredictor(config)
        
        # Register default tasks
        self._register_default_tasks()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _register_default_tasks(self) -> None:
        """Register default scheduled tasks."""
        # Data collection tasks
        self.register_task(
            ScheduledTask(
                name="collect_daily_data",
                function=self._collect_daily_data,
                schedule_time="02:00",  # 2 AM
                priority=TaskPriority.HIGH,
                timeout=7200  # 2 hours
            )
        )
        
        self.register_task(
            ScheduledTask(
                name="collect_live_data",
                function=self._collect_live_data,
                schedule_time="*/5 * * * *",  # Every 5 minutes
                priority=TaskPriority.CRITICAL,
                timeout=300,  # 5 minutes
                enabled=False  # Enable only during match times
            )
        )
        
        self.register_task(
            ScheduledTask(
                name="update_odds",
                function=self._update_odds,
                schedule_time="*/15 * * * *",  # Every 15 minutes
                priority=TaskPriority.HIGH,
                timeout=600  # 10 minutes
            )
        )
        
        # Model training tasks
        self.register_task(
            ScheduledTask(
                name="train_models",
                function=self._train_models,
                schedule_time="03:00",  # 3 AM daily
                priority=TaskPriority.NORMAL,
                timeout=10800  # 3 hours
            )
        )
        
        self.register_task(
            ScheduledTask(
                name="evaluate_models",
                function=self._evaluate_models,
                schedule_time="weekly:sun:04:00",  # Sunday 4 AM
                priority=TaskPriority.NORMAL,
                timeout=3600  # 1 hour
            )
        )
        
        # Prediction tasks
        self.register_task(
            ScheduledTask(
                name="generate_daily_predictions",
                function=self._generate_daily_predictions,
                schedule_time="06:00",  # 6 AM
                priority=TaskPriority.HIGH,
                timeout=3600  # 1 hour
            )
        )
        
        self.register_task(
            ScheduledTask(
                name="update_live_predictions",
                function=self._update_live_predictions,
                schedule_time="*/10 * * * *",  # Every 10 minutes
                priority=TaskPriority.HIGH,
                timeout=300,  # 5 minutes
                enabled=False  # Enable during match times
            )
        )
        
        # Maintenance tasks
        self.register_task(
            ScheduledTask(
                name="cleanup_old_data",
                function=self._cleanup_old_data,
                schedule_time="weekly:mon:01:00",  # Monday 1 AM
                priority=TaskPriority.LOW,
                timeout=3600  # 1 hour
            )
        )
        
        self.register_task(
            ScheduledTask(
                name="backup_database",
                function=self._backup_database,
                schedule_time="daily:00:00",  # Midnight
                priority=TaskPriority.HIGH,
                timeout=1800  # 30 minutes
            )
        )
        
        self.register_task(
            ScheduledTask(
                name="send_daily_report",
                function=self._send_daily_report,
                schedule_time="09:00",  # 9 AM
                priority=TaskPriority.NORMAL,
                timeout=600  # 10 minutes
            )
        )
        
    def register_task(self, task: ScheduledTask) -> None:
        """Register a new scheduled task."""
        self.tasks[task.name] = task
        logger.info(f"Registered task: {task.name}")
        
    def unregister_task(self, task_name: str) -> None:
        """Unregister a scheduled task."""
        if task_name in self.tasks:
            del self.tasks[task_name]
            logger.info(f"Unregistered task: {task_name}")
            
    def enable_task(self, task_name: str) -> None:
        """Enable a scheduled task."""
        if task_name in self.tasks:
            self.tasks[task_name].enabled = True
            logger.info(f"Enabled task: {task_name}")
            
    def disable_task(self, task_name: str) -> None:
        """Disable a scheduled task."""
        if task_name in self.tasks:
            self.tasks[task_name].enabled = False
            logger.info(f"Disabled task: {task_name}")
            
    async def run(self) -> None:
        """Start the scheduler."""
        self.running = True
        logger.info("Task scheduler started")
        
        # Schedule all tasks
        self._schedule_all_tasks()
        
        # Main scheduler loop
        while self.running:
            try:
                # Run pending tasks
                schedule.run_pending()
                
                # Check for tasks that need to be enabled/disabled
                self._check_dynamic_tasks()
                
                # Sleep for a short interval
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                await asyncio.sleep(5)
                
        logger.info("Task scheduler stopped")
        
    def stop(self) -> None:
        """Stop the scheduler."""
        self.running = False
        logger.info("Stopping task scheduler...")
        
    def _signal_handler(self, signum, frame) -> None:
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}")
        self.stop()
        sys.exit(0)
        
    def _schedule_all_tasks(self) -> None:
        """Schedule all registered tasks."""
        for task in self.tasks.values():
            if task.enabled:
                self._schedule_task(task)
                
    def _schedule_task(self, task: ScheduledTask) -> None:
        """Schedule a single task."""
        try:
            if task.schedule_time.startswith("*/"):
                # Interval scheduling (e.g., */5 * * * *)
                interval = int(task.schedule_time.split()[0][2:])
                schedule.every(interval).minutes.do(
                    lambda: asyncio.create_task(self._execute_task(task))
                )
                
            elif task.schedule_time.startswith("daily:"):
                # Daily at specific time
                time_str = task.schedule_time.split(":")[1:]
                time_str = ":".join(time_str)
                schedule.every().day.at(time_str).do(
                    lambda: asyncio.create_task(self._execute_task(task))
                )
                
            elif task.schedule_time.startswith("weekly:"):
                # Weekly on specific day and time
                parts = task.schedule_time.split(":")
                day = parts[1]
                time_str = ":".join(parts[2:])
                getattr(schedule.every(), day).at(time_str).do(
                    lambda: asyncio.create_task(self._execute_task(task))
                )
                
            else:
                # Assume it's a time string (HH:MM)
                schedule.every().day.at(task.schedule_time).do(
                    lambda: asyncio.create_task(self._execute_task(task))
                )
                
            logger.info(f"Scheduled task: {task.name} at {task.schedule_time}")
            
        except Exception as e:
            logger.error(f"Error scheduling task {task.name}: {e}")
            
    async def _execute_task(self, task: ScheduledTask) -> None:
        """Execute a scheduled task with error handling and retries."""
        if not task.enabled:
            return
            
        logger.info(f"Executing task: {task.name}")
        task.status = TaskStatus.RUNNING
        task.last_run = datetime.now(self.timezone)
        
        retry_count = 0
        while retry_count <= task.max_retries:
            try:
                # Execute task with timeout
                await asyncio.wait_for(
                    task.function(),
                    timeout=task.timeout
                )
                
                # Task completed successfully
                task.status = TaskStatus.COMPLETED
                task.error_count = 0
                logger.info(f"Task completed: {task.name}")
                
                # Send success notification for critical tasks
                if task.priority == TaskPriority.CRITICAL:
                    await self.notification_service.send_task_completion(
                        task_name=task.name,
                        status="success"
                    )
                
                break
                
            except asyncio.TimeoutError:
                logger.error(f"Task {task.name} timed out after {task.timeout} seconds")
                retry_count += 1
                
            except Exception as e:
                logger.error(f"Task {task.name} failed: {e}")
                retry_count += 1
                
                if retry_count <= task.max_retries:
                    logger.info(f"Retrying task {task.name} ({retry_count}/{task.max_retries})")
                    await asyncio.sleep(task.retry_delay)
                    
        if retry_count > task.max_retries:
            task.status = TaskStatus.FAILED
            task.error_count += 1
            
            # Send failure notification
            await self.notification_service.send_task_failure(
                task_name=task.name,
                error_count=task.error_count
            )
            
    def _check_dynamic_tasks(self) -> None:
        """Check and enable/disable tasks based on current conditions."""
        now = datetime.now(self.timezone)
        
        # Check if there are live matches
        live_matches = self.cache_manager.get("live_matches_count", 0)
        
        # Enable/disable live data collection
        if live_matches > 0:
            self.enable_task("collect_live_data")
            self.enable_task("update_live_predictions")
        else:
            # Check if matches are scheduled in the next hour
            upcoming_matches = self.db_manager.count_upcoming_matches(hours=1)
            if upcoming_matches > 0:
                self.enable_task("collect_live_data")
            else:
                self.disable_task("collect_live_data")
                self.disable_task("update_live_predictions")
                
    # Task implementation methods
    async def _collect_daily_data(self) -> None:
        """Collect daily data from all sources."""
        logger.info("Starting daily data collection")
        summary = await self.data_collector.collect_all_data(days_back=2)
        logger.info(f"Daily data collection completed: {summary}")
        
        # Cache summary
        self.cache_manager.set("last_data_collection", summary, ttl=86400)
        
    async def _collect_live_data(self) -> None:
        """Collect live match data."""
        summary = await self.data_collector.collect_live_data()
        self.cache_manager.set("live_matches_count", summary['live_matches'], ttl=300)
        
    async def _update_odds(self) -> None:
        """Update odds for upcoming matches."""
        upcoming_matches = self.db_manager.get_upcoming_matches(days=3)
        updated = 0
        
        for match in upcoming_matches:
            try:
                odds = await self.football_api.get_match_odds(match.id)
                if odds:
                    self.db_manager.insert_odds(match.id, odds)
                    updated += 1
            except Exception as e:
                logger.warning(f"Failed to update odds for match {match.id}: {e}")
                
        logger.info(f"Updated odds for {updated} matches")
        
    async def _train_models(self) -> None:
        """Train prediction models."""
        logger.info("Starting model training")
        
        # Load latest data
        training_data = self.db_manager.get_training_data()
        
        # Train models
        results = self.predictor.train(training_data)
        
        # Save model artifacts
        self.predictor.save_model(f"model_{datetime.now().strftime('%Y%m%d')}")
        
        logger.info(f"Model training completed: {results}")
        
    async def _evaluate_models(self) -> None:
        """Evaluate model performance."""
        from model_evaluator import ModelEvaluator
        
        evaluator = ModelEvaluator(self.config)
        results = await evaluator.evaluate_all_models()
        
        # Store evaluation results
        self.db_manager.store_model_evaluation(results)
        
        # Send evaluation report
        await self.notification_service.send_model_evaluation_report(results)
        
    async def _generate_daily_predictions(self) -> None:
        """Generate predictions for today's matches."""
        today_matches = self.db_manager.get_matches_by_date(datetime.now().date())
        
        predictions = []
        for match in today_matches:
            try:
                prediction = self.predictor.predict_match(match)
                self.db_manager.store_prediction(prediction)
                predictions.append(prediction)
            except Exception as e:
                logger.error(f"Failed to predict match {match.id}: {e}")
                
        logger.info(f"Generated {len(predictions)} predictions")
        
        # Send prediction summary
        if predictions:
            await self.notification_service.send_daily_predictions(predictions)
            
    async def _update_live_predictions(self) -> None:
        """Update predictions for live matches."""
        live_matches = self.db_manager.get_live_matches()
        
        for match in live_matches:
            try:
                # Get live data from cache
                live_data = self.cache_manager.get(f"live:{match.id}")
                if live_data:
                    # Update prediction with live data
                    prediction = self.predictor.predict_live(match, live_data)
                    self.db_manager.update_prediction(match.id, prediction)
                    
                    # Check for significant changes
                    if prediction.get('significant_change'):
                        await self.notification_service.send_live_alert(
                            match, prediction
                        )
            except Exception as e:
                logger.error(f"Failed to update live prediction for match {match.id}: {e}")
                
    async def _cleanup_old_data(self) -> None:
        """Clean up old data from database."""
        cutoff_date = datetime.now() - timedelta(days=self.config.DATA_RETENTION_DAYS)
        
        # Delete old matches
        deleted_matches = self.db_manager.delete_old_matches(cutoff_date)
        
        # Delete old predictions
        deleted_predictions = self.db_manager.delete_old_predictions(cutoff_date)
        
        # Clean up cache
        self.cache_manager.cleanup()
        
        logger.info(f"Cleanup completed: {deleted_matches} matches, {deleted_predictions} predictions")
        
    async def _backup_database(self) -> None:
        """Backup database."""
        backup_path = self.db_manager.backup_database()
        logger.info(f"Database backed up to: {backup_path}")
        
        # Upload to cloud storage if configured
        if self.config.CLOUD_BACKUP_ENABLED:
            # Implement cloud backup logic
            pass
            
    async def _send_daily_report(self) -> None:
        """Send daily system report."""
        report = {
            'date': datetime.now().date(),
            'system_status': self.get_system_status(),
            'task_status': self.get_task_status(),
            'data_status': self.data_collector.get_collection_status(),
            'prediction_summary': self.db_manager.get_daily_prediction_summary()
        }
        
        await self.notification_service.send_daily_report(report)
        
    def get_task_status(self) -> List[Dict[str, Any]]:
        """Get status of all scheduled tasks."""
        return [
            {
                'name': task.name,
                'enabled': task.enabled,
                'status': task.status.value,
                'last_run': task.last_run,
                'next_run': task.next_run,
                'error_count': task.error_count,
                'priority': task.priority.name
            }
            for task in self.tasks.values()
        ]
        
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        return {
            'scheduler_running': self.running,
            'total_tasks': len(self.tasks),
            'enabled_tasks': sum(1 for t in self.tasks.values() if t.enabled),
            'failed_tasks': sum(1 for t in self.tasks.values() if t.status == TaskStatus.FAILED),
            'uptime': self._calculate_uptime()
        }
        
    def _calculate_uptime(self) -> str:
        """Calculate scheduler uptime."""
        # Implement uptime calculation
        return "N/A"


def main():
    """Main entry point for the scheduler."""
    config = Config()
    scheduler = TaskScheduler(config)
    
    try:
        asyncio.run(scheduler.run())
    except KeyboardInterrupt:
        logger.info("Scheduler interrupted by user")
    except Exception as e:
        logger.error(f"Scheduler crashed: {e}")
        raise


if __name__ == "__main__":
    main()