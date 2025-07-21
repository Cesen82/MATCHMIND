"""
Advanced logging system for ProFootballAI
"""

import logging
import logging.config
import sys
from pathlib import Path
from datetime import datetime
import json
from typing import Dict, Any, Optional
import colorlog
from pythonjsonlogger import jsonlogger

from config import LOGGING_CONFIG, DATA_DIR


class ContextFilter(logging.Filter):
    """Add context information to log records"""
    
    def __init__(self, context: Dict[str, Any] = None):
        super().__init__()
        self.context = context or {}
        
    def filter(self, record):
        # Add context to record
        for key, value in self.context.items():
            setattr(record, key, value)
            
        # Add additional metadata
        record.hostname = 'profootball-ai'
        record.timestamp = datetime.utcnow().isoformat()
        
        return True


class PerformanceFilter(logging.Filter):
    """Filter to track performance metrics in logs"""
    
    def __init__(self):
        super().__init__()
        self.request_times = []
        
    def filter(self, record):
        # Track API request times
        if hasattr(record, 'request_time'):
            self.request_times.append(record.request_time)
            
            # Calculate average
            if len(self.request_times) > 100:
                self.request_times = self.request_times[-100:]
                
            record.avg_request_time = sum(self.request_times) / len(self.request_times)
            
        return True


class ErrorAggregator(logging.Handler):
    """Aggregate errors for reporting"""
    
    def __init__(self):
        super().__init__()
        self.errors = []
        self.error_counts = {}
        
    def emit(self, record):
        if record.levelno >= logging.ERROR:
            error_key = f"{record.module}:{record.funcName}:{record.lineno}"
            
            self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
            
            self.errors.append({
                'timestamp': datetime.utcnow().isoformat(),
                'level': record.levelname,
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno,
                'message': record.getMessage(),
                'exception': record.exc_info
            })
            
            # Keep only last 1000 errors
            if len(self.errors) > 1000:
                self.errors = self.errors[-1000:]
                
    def get_error_summary(self) -> Dict[str, Any]:
        """Get error summary statistics"""
        return {
            'total_errors': len(self.errors),
            'unique_errors': len(self.error_counts),
            'top_errors': sorted(
                self.error_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10],
            'recent_errors': self.errors[-10:]
        }


def setup_logging(config: Dict[str, Any] = None):
    """Setup advanced logging configuration"""
    
    if config is None:
        config = LOGGING_CONFIG
        
    # Create logs directory
    log_dir = DATA_DIR / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Update file handler paths
    for handler in config.get('handlers', {}).values():
        if 'filename' in handler:
            handler['filename'] = str(log_dir / Path(handler['filename']).name)
            
    # Apply configuration
    logging.config.dictConfig(config)
    
    # Get root logger
    root_logger = logging.getLogger()
    
    # Add context filter
    context_filter = ContextFilter({
        'app': 'profootball-ai',
        'version': '2.0'
    })
    
    for handler in root_logger.handlers:
        handler.addFilter(context_filter)
        
    # Add performance filter to specific loggers
    perf_filter = PerformanceFilter()
    api_logger = logging.getLogger('profootball.api')
    for handler in api_logger.handlers:
        handler.addFilter(perf_filter)
        
    # Add error aggregator
    error_aggregator = ErrorAggregator()
    error_aggregator.setLevel(logging.ERROR)
    root_logger.addHandler(error_aggregator)
    
    # Store aggregator for access
    logging.error_aggregator = error_aggregator
    
    # Setup colored console output for development
    if sys.stderr.isatty():
        console_handler = colorlog.StreamHandler()
        console_handler.setFormatter(
            colorlog.ColoredFormatter(
                '%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(name)s%(reset)s %(message)s',
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'red,bg_white',
                }
            )
        )
        
        # Replace default console handler
        for handler in root_logger.handlers[:]:
            if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stderr:
                root_logger.removeHandler(handler)
                
        root_logger.addHandler(console_handler)
        
    logger = logging.getLogger(__name__)
    logger.info("Logging system initialized")
    
    return root_logger


def get_logger(name: str, context: Dict[str, Any] = None) -> logging.Logger:
    """Get logger with optional context"""
    logger = logging.getLogger(name)
    
    if context:
        context_filter = ContextFilter(context)
        for handler in logger.handlers:
            handler.addFilter(context_filter)
            
    return logger


class LoggerAdapter(logging.LoggerAdapter):
    """Logger adapter with additional context"""
    
    def process(self, msg, kwargs):
        # Add context to message
        if self.extra:
            context_str = ' '.join(f"{k}={v}" for k, v in self.extra.items())
            return f"[{context_str}] {msg}", kwargs
        return msg, kwargs


def log_performance(func):
    """Decorator to log function performance"""
    import functools
    import time
    
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start_time = time.time()
        
        try:
            result = await func(*args, **kwargs)
            elapsed = time.time() - start_time
            
            logger.debug(
                f"{func.__name__} completed in {elapsed:.3f}s",
                extra={'request_time': elapsed}
            )
            
            return result
            
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                f"{func.__name__} failed after {elapsed:.3f}s: {str(e)}",
                extra={'request_time': elapsed},
                exc_info=True
            )
            raise
            
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            
            logger.debug(
                f"{func.__name__} completed in {elapsed:.3f}s",
                extra={'request_time': elapsed}
            )
            
            return result
            
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                f"{func.__name__} failed after {elapsed:.3f}s: {str(e)}",
                extra={'request_time': elapsed},
                exc_info=True
            )
            raise
            
    # Return appropriate wrapper
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


def log_errors(logger_name: str = None):
    """Decorator to log exceptions"""
    import functools
    
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            logger = logging.getLogger(logger_name or func.__module__)
            
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.error(
                    f"Error in {func.__name__}: {str(e)}",
                    exc_info=True,
                    extra={
                        'function': func.__name__,
                        'module': func.__module__,
                        'args': str(args)[:200],
                        'kwargs': str(kwargs)[:200]
                    }
                )
                raise
                
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            logger = logging.getLogger(logger_name or func.__module__)
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(
                    f"Error in {func.__name__}: {str(e)}",
                    exc_info=True,
                    extra={
                        'function': func.__name__,
                        'module': func.__module__,
                        'args': str(args)[:200],
                        'kwargs': str(kwargs)[:200]
                    }
                )
                raise
                
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
            
    return decorator


class StructuredLogger:
    """Structured logging with JSON output"""
    
    def __init__(self, name: str, context: Dict[str, Any] = None):
        self.logger = logging.getLogger(name)
        self.context = context or {}
        
        # Add JSON formatter to file handler
        for handler in self.logger.handlers:
            if isinstance(handler, logging.FileHandler):
                formatter = jsonlogger.JsonFormatter(
                    '%(timestamp)s %(level)s %(name)s %(message)s'
                )
                handler.setFormatter(formatter)
                
    def _log(self, level: int, message: str, **kwargs):
        """Log with structured data"""
        extra = self.context.copy()
        extra.update(kwargs)
        
        self.logger.log(level, message, extra=extra)
        
    def debug(self, message: str, **kwargs):
        self._log(logging.DEBUG, message, **kwargs)
        
    def info(self, message: str, **kwargs):
        self._log(logging.INFO, message, **kwargs)
        
    def warning(self, message: str, **kwargs):
        self._log(logging.WARNING, message, **kwargs)
        
    def error(self, message: str, **kwargs):
        self._log(logging.ERROR, message, **kwargs)
        
    def critical(self, message: str, **kwargs):
        self._log(logging.CRITICAL, message, **kwargs)


# Import asyncio at module level
import asyncio