"""
Logging configuration for PhishNetra
Production-ready logging with structured output
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

from .config import settings


def setup_logging(
    level: Optional[str] = None,
    format_type: Optional[str] = None,
    enable_file_logging: bool = True
) -> logging.Logger:
    """
    Setup comprehensive logging configuration

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: Log format type ('json' or 'text')
        enable_file_logging: Whether to enable file logging

    Returns:
        Root logger instance
    """

    # Get configuration from settings
    log_level = level or settings.log_level
    format_type = format_type or settings.log_format

    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(numeric_level)

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatters
    if format_type == "json":
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
            '"logger": "%(name)s", "message": "%(message)s", '
            '"module": "%(module)s", "function": "%(funcName)s", '
            '"line": %(lineno)d, "process": %(process)d, "thread": %(thread)d}'
        )
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (rotating)
    if enable_file_logging:
        log_file = Path("./logs/riskanalyzer.log")
        log_file.parent.mkdir(exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Create specific loggers for different components
    loggers = {
        'fasttext': logging.getLogger('fasttext'),
        'sentence_transformer': logging.getLogger('sentence_transformer'),
        'distilbert': logging.getLogger('distilbert'),
        'similarity': logging.getLogger('similarity'),
        'fusion': logging.getLogger('fusion'),
        'api': logging.getLogger('api'),
        'explainability': logging.getLogger('explainability')
    }

    # Set levels for component loggers
    for component_logger in loggers.values():
        component_logger.setLevel(numeric_level)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific component

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class LoggerMixin:
    """Mixin class to add logging to any class"""

    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class"""
        class_name = self.__class__.__name__
        return logging.getLogger(f"{self.__module__}.{class_name}")


# Performance logging decorator
def log_performance(logger: logging.Logger):
    """Decorator to log function performance"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.info(
                    f"Function {func.__name__} completed in {duration:.4f}s"
                )
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    f"Function {func.__name__} failed after {duration:.4f}s: {str(e)}"
                )
                raise
        return wrapper
    return decorator


# Request logging for API endpoints
def log_request(logger: logging.Logger):
    """Decorator to log API requests"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            import time
            start_time = time.time()
            try:
                # Log request start
                logger.info(f"Processing request: {func.__name__}")

                result = func(*args, **kwargs)

                # Log request completion
                duration = time.time() - start_time
                logger.info(
                    f"Request {func.__name__} completed in {duration:.4f}s"
                )
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.error(
                    f"Request {func.__name__} failed after {duration:.4f}s: {str(e)}"
                )
                raise
        return wrapper
    return decorator


# Initialize logging on import
setup_logging()