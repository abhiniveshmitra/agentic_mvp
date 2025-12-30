"""
Structured logging with run context.

Provides consistent logging across all modules with:
- Run ID tracking
- Timestamped entries
- Structured output
"""

import logging
import sys
from pathlib import Path
from typing import Optional


class RunContextFilter(logging.Filter):
    """Add run context to log records."""
    
    def __init__(self, run_id: Optional[str] = None):
        super().__init__()
        self.run_id = run_id or "NO_RUN"
    
    def filter(self, record):
        record.run_id = self.run_id
        return True


def setup_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
    run_id: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with consistent formatting.
    
    Args:
        name: Logger name (usually __name__)
        log_file: Optional file path for logging
        level: Logging level
        run_id: Optional run ID for context
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Format string
    fmt = "%(asctime)s | %(run_id)s | %(name)s | %(levelname)s | %(message)s"
    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(RunContextFilter(run_id))
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.addFilter(RunContextFilter(run_id))
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get an existing logger or create a new one."""
    return logging.getLogger(name)
