"""
Notification Service Module
==========================

Handles all notifications including email, SMS, push notifications, and webhooks.
Manages alerts for predictions, system events, and user notifications.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from enum import Enum
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import aiohttp
import json
from dataclasses import dataclass
from jinja2 import Template

from config import Config
from database_manager import DatabaseManager
from formatters_module import format_currency, format_percentage
from logger_module import setup_logger

logger = setup_logger(__name__)


class NotificationType(Enum):
    """Types of notifications."""
    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"
    PUSH = "push"
    TELEGRAM = "telegram"
    DISCORD = "discord"


class NotificationPriority(Enum):
    """Notification priority levels."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4


@dataclass
class Notification:
    """Represents a notification to be sent."""
    type: NotificationType
    recipient: str
    subject: str
    body: str
    priority: NotificationPriority = NotificationPriority.NORMAL
    attachments: List[str] = None
    metadata: Dict[str, Any] = None
    retry_count: int = 0
    max_retries: int = 3


class NotificationService:
    """Manages all notification operations."""
    
    def __init__(self, config: Config):
        """
        Initialize notification service.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.db = DatabaseManager(config)
        self.notification_queue: List[Notification] = []
        self.templates = self._load_templates()
        
        # Email configuration
        self.smtp_server = config.SMTP_SERVER
        self.smtp_port = config.SMTP_PORT
        self.smtp_username = config.SMTP_USERNAME
        self.smtp_password = config.SMTP_PASSWORD
        self.from_email = config.FROM_EMAIL
        
        # Webhook configurations
        self.webhooks = config.WEBHOOKS
        
        # Telegram configuration
        self.telegram_bot_token = config.TELEGRAM_BOT_TOKEN
        self.telegram_chat_ids = config.TELEGRAM_CHAT_IDS
        
        # Discord configuration
        self.discord_webhook_url = config.DISCORD_WEBHOOK_URL
        
    def _load_templates(self) -> Dict[str, Template]:
        """Load notification templates."""
        templates = {
            'daily_predictions': Template("""
                <h2>Daily Football Predictions - {{ date }}</h2>
                <p>Here are today's match predictions:</p>
                
                {% for pred in predictions %}
                <div style="border: 1px solid #ddd; padding: 10px; margin: 10px 0;">
                    <h3>{{ pred.home_team }} vs {{ pred.away_team }}</h3>
                    <p><strong>Kick-off:</strong> {{ pred.kickoff_time }}</p>
                    <p><strong>League:</strong> {{ pred.league }}</p>
                    
                    <h4>Predictions:</h4>
                    <ul>
                        <li>Result: {{ pred.predicted_result }} (Confidence: {{ pred.confidence }}%)</li>
                        <li>Goals: {{ pred.predicted_goals }}</li>
                        <li>Recommended Bet: {{ pred.recommended_bet }}</li>
                        <li>Expected Value: {{ pred.expected_value }}</li>
                    </ul>
                </div>
                {% endfor %}
            """),
            
            'live_alert': Template("""
                🚨 LIVE MATCH ALERT 🚨
                
                {{ match.home_team }} {{ match.home_score }} - {{ match.away_score }} {{ match.away_team }}
                Minute: {{ match.minute }}'
                
                {{ alert_message }}
                
                Updated Predictions:
                - Result: {{ prediction.result }} ({{ prediction.confidence }}%)
                - Next Goal: {{ prediction.next_goal }}
                - Live Betting Opportunity: {{ prediction.opportunity }}
            """),
            
            'daily_report': Template("""
                <h2>Daily System Report - {{ date }}</h2>
                
                <h3>System Status</h3>
                <ul>
                    <li>Uptime: {{ system_status.uptime }}</li>
                    <li>Active Tasks: {{ system_status.enabled_tasks }}/{{ system_status.total_tasks }}</li>
                    <li>Failed Tasks: {{ system_status.failed_tasks }}</li>
                </ul>
                
                <h3>Data Collection</h3>
                <ul>
                    <li>Last Update: {{ data_status.last_update }}</li>
                    <li>Matches: {{ data_status.data_counts.matches }}</li>
                    <li>Teams: {{ data_status.data_counts.teams }}</li>
                    <li>Odds Records: {{ data_status.data_counts.odds_records }}</li>
                </ul>
                
                <h3>Prediction Performance</h3>
                <ul>
                    <li>Predictions Made: {{ prediction_summary.total_predictions }}</li>
                    <li>Success Rate: {{ prediction_summary.success_rate }}%</li>
                    <li>ROI: {{ prediction_summary.roi }}%</li>
                </ul>
            """),
            
            'model_evaluation': Template("""
                <h2>Model Evaluation Report</h2>
                
                {% for model_name, metrics in results.items() %}
                <h3>{{ model_name }}</h3>
                <table border="1" style="border-collapse: collapse;">
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    {% for metric, value in metrics.items() %}
                    <tr>
                        <td>{{ metric }}</td>
                        <td>{{ value }}</td>
                    </tr>
                    {% endfor %}
                </table>
                {% endfor %}
            """),
            
            'high_value_bet': Template("""
                💰 HIGH VALUE BET ALERT 💰
                
                Match: {{ match.home_team }} vs {{ match.away_team }}
                League: {{ match.league }}
                Kick-off: {{ match.kickoff_time }}
                
                Bet Details:
                - Type: {{ bet.type }}
                - Odds: {{ bet.odds }}
                - Stake: {{ bet.recommended_stake }}
                - Expected Value: {{ bet.expected_value }}
                - Confidence: {{ bet.confidence }}%
                
                Analysis: {{ bet.analysis }}
            """)
        }
        
        return templates
        
    async def send_notification(self, notification: Notification) -> bool:
        """
        Send a notification based on its type.
        
        Args:
            notification: Notification object
            
        Returns:
            Success status
        """
        try:
            if notification.type == NotificationType.EMAIL:
                return await self._send_email(notification)
            elif notification.type == NotificationType.WEBHOOK:
                return await self._send_webhook(notification)
            elif notification.type == NotificationType.TELEGRAM:
                return await self._send_telegram(notification)
            elif notification.type == NotificationType.DISCORD:
                return await self._send_discord(notification)
            elif notification.type == NotificationType.SMS:
                return await self._send_sms(notification)
            elif notification.type == NotificationType.PUSH:
                return await self._send_push(notification)
            else:
                logger.warning(f"Unsupported notification type: {notification.type}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send {notification.type.value} notification: {e}")
            
            # Retry logic
            if notification.retry_count < notification.max_retries:
                notification.retry_count += 1
                self.notification_queue.append(notification)
                logger.info(f"Queued notification for retry ({notification.retry_count}/{notification.max_retries})")
                
            return False
            
    async def _send_email(self, notification: Notification) -> bool:
        """Send email notification."""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = notification.recipient
            msg['Subject'] = notification.subject
            
            # Add body
            msg.attach(MIMEText(notification.body, 'html'))
            
            # Add attachments if any
            if notification.attachments:
                for file_path in notification.attachments:
                    self._attach_file(msg, file_path)
                    
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)
                
            logger.info(f"Email sent to {notification.recipient}")
            return True
            
        except Exception as e:
            logger.error(f"Email sending failed: {e}")
            return False
            
    def _attach_file(self, msg: MIMEMultipart, file_path: str) -> None:
        """Attach file to email."""
        try:
            with open(file_path, "rb") as file:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(file.read())
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename= {file_path.split("/")[-1]}'
                )
                msg.attach(part)
        except Exception as e:
            logger.warning(f"Failed to attach file {file_path}: {e}")
            
    async def _send_webhook(self, notification: Notification) -> bool:
        """Send webhook notification."""
        webhook_url = notification.recipient
        
        payload = {
            'subject': notification.subject,
            'body': notification.body,
            'priority': notification.priority.name,
            'timestamp': datetime.now().isoformat(),
            'metadata': notification.metadata or {}
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    webhook_url,
                    json=payload,
                    headers={'Content-Type': 'application/json'}
                ) as response:
                    if response.status == 200:
                        logger.info(f"Webhook sent to {webhook_url}")
                        return True
                    else:
                        logger.error(f"Webhook failed with status {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Webhook sending failed: {e}")
            return False
            
    async def _send_telegram(self, notification: Notification) -> bool:
        """Send Telegram notification."""
        if not self.telegram_bot_token:
            logger.warning("Telegram bot token not configured")
            return False
            
        url = f"https://api.telegram.org/bot{self.telegram_bot_token}/sendMessage"
        
        # Send to all configured chat IDs
        success_count = 0
        for chat_id in self.telegram_chat_ids:
            payload = {
                'chat_id': chat_id,
                'text': f"*{notification.subject}*\n\n{notification.body}",
                'parse_mode': 'Markdown'
            }
            
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=payload) as response:
                        if response.status == 200:
                            success_count += 1
                        else:
                            logger.error(f"Telegram message failed for chat {chat_id}")
                            
            except Exception as e:
                logger.error(f"Telegram sending failed: {e}")
                
        return success_count > 0
        
    async def _send_discord(self, notification: Notification) -> bool:
        """Send Discord notification."""
        if not self.discord_webhook_url:
            logger.warning("Discord webhook URL not configured")
            return False
            
        embed = {
            'title': notification.subject,
            'description': notification.body,
            'color': self._get_discord_color(notification.priority),
            'timestamp': datetime.now().isoformat(),
            'footer': {'text': 'Football Prediction System'}
        }
        
        payload = {'embeds': [embed]}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.discord_webhook_url,
                    json=payload
                ) as response:
                    if response.status == 204:
                        logger.info("Discord notification sent")
                        return True
                    else:
                        logger.error(f"Discord webhook failed with status {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Discord sending failed: {e}")
            return False
            
    def _get_discord_color(self, priority: NotificationPriority) -> int:
        """Get Discord embed color based on priority."""
        colors = {
            NotificationPriority.CRITICAL: 0xFF0000,  # Red
            NotificationPriority.HIGH: 0xFF9900,      # Orange
            NotificationPriority.NORMAL: 0x0099FF,    # Blue
            NotificationPriority.LOW: 0x00FF00        # Green
        }
        return colors.get(priority, 0x0099FF)
        
    async def _send_sms(self, notification: Notification) -> bool:
        """Send SMS notification (placeholder - implement with SMS provider)."""
        logger.info(f"SMS notification to {notification.recipient}: {notification.subject}")
        # Implement SMS sending logic with provider like Twilio
        return True
        
    async def _send_push(self, notification: Notification) -> bool:
        """Send push notification (placeholder - implement with push service)."""
        logger.info(f"Push notification: {notification.subject}")
        # Implement push notification logic with service like Firebase
        return True
        
    # Specific notification methods
    async def send_daily_predictions(self, predictions: List[Dict[str, Any]]) -> None:
        """Send daily predictions to all subscribers."""
        subject = f"Daily Football Predictions - {datetime.now().strftime('%Y-%m-%d')}"
        
        # Format predictions
        formatted_predictions = []
        for pred in predictions:
            formatted_predictions.append({
                'home_team': pred['home_team'],
                'away_team': pred['away_team'],
                'kickoff_time': pred['kickoff_time'].strftime('%H:%M'),
                'league': pred['league'],
                'predicted_result': pred['predicted_result'],
                'confidence': format_percentage(pred['confidence']),
                'predicted_goals': f"{pred['predicted_home_goals']}-{pred['predicted_away_goals']}",
                'recommended_bet': pred['recommended_bet'],
                'expected_value': format_currency(pred['expected_value'])
            })
            
        body = self.templates['daily_predictions'].render(
            date=datetime.now().strftime('%Y-%m-%d'),
            predictions=formatted_predictions
        )
        
        # Get subscribers
        subscribers = self.db.get_notification_subscribers('daily_predictions')
        
        # Send to all subscribers
        for subscriber in subscribers:
            notification = Notification(
                type=NotificationType(subscriber.notification_type),
                recipient=subscriber.recipient,
                subject=subject,
                body=body,
                priority=NotificationPriority.NORMAL
            )
            await self.send_notification(notification)
            
    async def send_live_alert(self, match: Dict[str, Any], prediction: Dict[str, Any]) -> None:
        """Send live match alert."""
        alert_message = self._generate_alert_message(match, prediction)
        
        body = self.templates['live_alert'].render(
            match=match,
            prediction=prediction,
            alert_message=alert_message
        )
        
        # Get subscribers for live alerts
        subscribers = self.db.get_notification_subscribers('live_alerts')
        
        # Send high-priority notifications
        for subscriber in subscribers:
            notification = Notification(
                type=NotificationType(subscriber.notification_type),
                recipient=subscriber.recipient,
                subject=f"⚡ Live Alert: {match['home_team']} vs {match['away_team']}",
                body=body,
                priority=NotificationPriority.HIGH
            )
            await self.send_notification(notification)
            
    def _generate_alert_message(self, match: Dict[str, Any], prediction: Dict[str, Any]) -> str:
        """Generate contextual alert message based on match events."""
        if prediction.get('goal_probability', 0) > 0.7:
            return "🎯 High probability of goal in next 10 minutes!"
        elif prediction.get('momentum_shift'):
            return "🔄 Momentum has shifted! Betting opportunity detected."
        elif match.get('red_card'):
            return "🟥 Red card! Match dynamics have changed significantly."
        else:
            return "📊 Significant change in match predictions detected."
            
    async def send_high_value_bet(self, bet_opportunity: Dict[str, Any]) -> None:
        """Send high value betting opportunity alert."""
        body = self.templates['high_value_bet'].render(
            match=bet_opportunity['match'],
            bet=bet_opportunity['bet']
        )
        
        # Get VIP subscribers
        subscribers = self.db.get_notification_subscribers('high_value_bets')
        
        for subscriber in subscribers:
            notification = Notification(
                type=NotificationType(subscriber.notification_type),
                recipient=subscriber.recipient,
                subject="💎 High Value Bet Opportunity",
                body=body,
                priority=NotificationPriority.CRITICAL
            )
            await self.send_notification(notification)
            
    async def send_daily_report(self, report: Dict[str, Any]) -> None:
        """Send daily system report to administrators."""
        body = self.templates['daily_report'].render(
            date=report['date'],
            system_status=report['system_status'],
            data_status=report['data_status'],
            prediction_summary=report['prediction_summary']
        )
        
        # Get admin recipients
        admins = self.config.ADMIN_EMAILS
        
        for admin_email in admins:
            notification = Notification(
                type=NotificationType.EMAIL,
                recipient=admin_email,
                subject=f"Daily System Report - {report['date']}",
                body=body,
                priority=NotificationPriority.LOW
            )
            await self.send_notification(notification)
            
    async def send_model_evaluation_report(self, results: Dict[str, Any]) -> None:
        """Send model evaluation report."""
        body = self.templates['model_evaluation'].render(results=results)
        
        # Send to data science team
        for recipient in self.config.DATA_SCIENCE_EMAILS:
            notification = Notification(
                type=NotificationType.EMAIL,
                recipient=recipient,
                subject="Weekly Model Evaluation Report",
                body=body,
                priority=NotificationPriority.NORMAL,
                attachments=[results.get('detailed_report_path')]
            )
            await self.send_notification(notification)
            
    async def send_task_failure(self, task_name: str, error_count: int) -> None:
        """Send task failure notification."""
        body = f"""
        Task Failure Alert
        
        Task: {task_name}
        Error Count: {error_count}
        Time: {datetime.now()}
        
        Please check the logs for more details.
        """
        
        # Send critical alert to admins
        for admin in self.config.ADMIN_EMAILS:
            notification = Notification(
                type=NotificationType.EMAIL,
                recipient=admin,
                subject=f"🚨 Task Failure: {task_name}",
                body=body,
                priority=NotificationPriority.CRITICAL
            )
            await self.send_notification(notification)
            
    async def send_task_completion(self, task_name: str, status: str) -> None:
        """Send task completion notification for critical tasks."""
        body = f"""
        Critical Task Completed
        
        Task: {task_name}
        Status: {status}
        Completed: {datetime.now()}
        """
        
        # Log completion (could be sent to monitoring dashboard)
        logger.info(f"Critical task completed: {task_name}")
        
    async def process_notification_queue(self) -> None:
        """Process pending notifications in the queue."""
        while self.notification_queue:
            notification = self.notification_queue.pop(0)
            await self.send_notification(notification)
            await asyncio.sleep(1)  # Rate limiting
            
    def subscribe_user(self, 
                      user_email: str,
                      notification_types: List[str],
                      preferences: Dict[str, Any]) -> bool:
        """Subscribe user to notifications."""
        try:
            self.db.add_notification_subscriber(
                email=user_email,
                notification_types=notification_types,
                preferences=preferences
            )
            return True
        except Exception as e:
            logger.error(f"Failed to subscribe user: {e}")
            return False
            
    def unsubscribe_user(self, user_email: str, notification_type: Optional[str] = None) -> bool:
        """Unsubscribe user from notifications."""
        try:
            if notification_type:
                self.db.remove_notification_subscription(user_email, notification_type)
            else:
                self.db.remove_all_subscriptions(user_email)
            return True
        except Exception as e:
            logger.error(f"Failed to unsubscribe user: {e}")
            return False


if __name__ == "__main__":
    # Test notifications
    async def test():
        config = Config()
        service = NotificationService(config)
        
        # Test email
        notification = Notification(
            type=NotificationType.EMAIL,
            recipient="test@example.com",
            subject="Test Notification",
            body="<h1>Test</h1><p>This is a test notification.</p>"
        )
        
        result = await service.send_notification(notification)
        print(f"Notification sent: {result}")
        
    asyncio.run(test())