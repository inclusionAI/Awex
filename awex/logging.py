# Licensed to the Awex developers under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import os
import sys
import traceback
from datetime import datetime
from enum import IntEnum


class LogLevel(IntEnum):
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


class Logger:
    """Simple logger that prints directly to stdout, bypassing Python's logging system."""

    def __init__(self, name: str = None, level: LogLevel = LogLevel.INFO):
        self.name = name or "root"
        self.level = level

    def _format_message(self, level_name: str, message: str, frame_info=None) -> str:
        """Format message according to: asctime\tlevelname filename:lineno -- process -- message"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]  # milliseconds
        process_id = os.getpid()

        if frame_info:
            filename = os.path.basename(frame_info.filename)
            lineno = frame_info.lineno
        else:
            filename = "unknown"
            lineno = 0

        return f"{timestamp}\t{level_name} {filename}:{lineno} -- {process_id} -- {message}"

    def _log(self, level: LogLevel, level_name: str, message: str, *args, **kwargs):
        """Internal log method that prints directly to stdout."""
        if level < self.level:
            return

        # Get caller's frame info
        import inspect

        frame = inspect.currentframe()
        try:
            # Go up the stack: _log -> debug/info/warning/error -> actual caller
            caller_frame = frame.f_back.f_back
            frame_info = inspect.getframeinfo(caller_frame)
        finally:
            del frame

        # Format message with args if provided
        if args:
            try:
                message = message % args
            except (TypeError, ValueError):
                pass

        formatted = self._format_message(level_name, message, frame_info)
        print(formatted, flush=True, file=sys.stdout)

    def debug(self, message: str, *args, **kwargs):
        """Log debug message."""
        self._log(LogLevel.DEBUG, "DEBUG", message, *args, **kwargs)

    def info(self, message: str, *args, **kwargs):
        """Log info message."""
        self._log(LogLevel.INFO, "INFO", message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs):
        """Log warning message."""
        self._log(LogLevel.WARNING, "WARNING", message, *args, **kwargs)

    def error(self, message: str, *args, **kwargs):
        """Log error message."""
        self._log(LogLevel.ERROR, "ERROR", message, *args, **kwargs)

    def critical(self, message: str, *args, **kwargs):
        """Log critical message."""
        self._log(LogLevel.CRITICAL, "CRITICAL", message, *args, **kwargs)

    def exception(self, message: str, *args, **kwargs):
        """Log exception with traceback."""
        self.error(message, *args, **kwargs)
        # Print traceback directly
        exc_info = sys.exc_info()
        if exc_info[0] is not None:
            tb_lines = traceback.format_exception(*exc_info)
            for line in tb_lines:
                print(line.rstrip(), flush=True, file=sys.stdout)

    def setLevel(self, level):
        """Set logging level."""
        if isinstance(level, int):
            self.level = level
        elif isinstance(level, str):
            level_map = {
                "DEBUG": LogLevel.DEBUG,
                "INFO": LogLevel.INFO,
                "WARNING": LogLevel.WARNING,
                "ERROR": LogLevel.ERROR,
                "CRITICAL": LogLevel.CRITICAL,
            }
            self.level = level_map.get(level.upper(), LogLevel.INFO)


# Global logger registry
_loggers = {}
_default_level = LogLevel.INFO


def getLogger(name: str = None) -> Logger:
    """Get or create a logger with the given name."""
    if name is None:
        name = "root"

    if name not in _loggers:
        _loggers[name] = Logger(name, _default_level)

    return _loggers[name]


def setLevel(level):
    """Set default level for all future loggers."""
    global _default_level
    if isinstance(level, int):
        _default_level = level
    elif isinstance(level, str):
        level_map = {
            "DEBUG": LogLevel.DEBUG,
            "INFO": LogLevel.INFO,
            "WARNING": LogLevel.WARNING,
            "ERROR": LogLevel.ERROR,
            "CRITICAL": LogLevel.CRITICAL,
        }
        _default_level = level_map.get(level.upper(), LogLevel.INFO)

    # Update all existing loggers
    for logger in _loggers.values():
        logger.setLevel(_default_level)


def basicConfig(level=None, format=None, force=False, **kwargs):
    """
    Compatibility function for logging.basicConfig().

    Args:
        level: Logging level (int or string)
        format: Format string (ignored - we use our own format)
        force: Whether to force reconfiguration (ignored for compatibility)
        **kwargs: Other arguments (ignored for compatibility)
    """
    if level is not None:
        setLevel(level)


# Expose standard logging levels for compatibility
DEBUG = LogLevel.DEBUG
INFO = LogLevel.INFO
WARNING = LogLevel.WARNING
ERROR = LogLevel.ERROR
CRITICAL = LogLevel.CRITICAL

# Create default root logger
logger = getLogger()
