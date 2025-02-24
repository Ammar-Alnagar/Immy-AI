import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Optional

class Logger:
    """Centralized logging system for the WebRTC AI Assistant."""
    
    def __init__(self, service_name: str, log_dir: str = "logs"):
        self.service_name = service_name
        self.log_dir = log_dir
        
        # Create logs directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Create logger
        self.logger = logging.getLogger(service_name)
        self.logger.setLevel(logging.DEBUG)
        
        # Remove any existing handlers
        self.logger.handlers = []
        
        # Create formatters
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # File handler with rotation (10MB max size, keep 5 backup files)
        file_handler = RotatingFileHandler(
            os.path.join(log_dir, f"{service_name}.log"),
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.INFO)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def _format_context(self, context: Optional[dict] = None) -> str:
        """Format context dictionary into a string."""
        if not context:
            return ""
        context_str = " | ".join(f"{k}={v}" for k, v in context.items())
        return f" | {context_str}"
    
    def info(self, message: str, context: Optional[dict] = None):
        """Log info message with optional context."""
        self.logger.info(f"{message}{self._format_context(context)}")
    
    def warning(self, message: str, context: Optional[dict] = None):
        """Log warning message with optional context."""
        self.logger.warning(f"{message}{self._format_context(context)}")
    
    def error(self, message: str, context: Optional[dict] = None, exc_info: bool = True):
        """Log error message with optional context and exception info."""
        self.logger.error(f"{message}{self._format_context(context)}", exc_info=exc_info)
    
    def debug(self, message: str, context: Optional[dict] = None):
        """Log debug message with optional context."""
        self.logger.debug(f"{message}{self._format_context(context)}")
    
    def performance(self, operation: str, start_time: float, context: Optional[dict] = None):
        """Log performance metrics."""
        end_time = datetime.now().timestamp()
        duration = round((end_time - start_time) * 1000, 2)  # Convert to milliseconds
        
        perf_context = {"duration_ms": duration}
        if context:
            perf_context.update(context)
        
        self.info(
            f"Performance: {operation}",
            context=perf_context
        )
    
    def api_call(self, api_name: str, success: bool, duration_ms: float, context: Optional[dict] = None):
        """Log API call results."""
        api_context = {
            "success": success,
            "duration_ms": duration_ms
        }
        if context:
            api_context.update(context)
        
        if success:
            self.info(f"API Call: {api_name}", context=api_context)
        else:
            self.error(f"API Call Failed: {api_name}", context=api_context)
