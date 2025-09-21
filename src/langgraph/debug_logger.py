"""Debug logger for tracking evidence flow in LangGraph agents."""

import os
from datetime import datetime
from typing import Any, Optional

class DebugLogger:
    """Simple debug logger that writes to file instead of console."""
    
    def __init__(self, log_file: Optional[str] = None):
        if log_file is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_file = f"logs/debug/langgraph_debug_{timestamp}.log"
        
        self.log_file = log_file
        # Ensure debug directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Clear existing log file
        with open(self.log_file, 'w') as f:
            f.write(f"=== LangGraph Debug Session Started: {datetime.now()} ===\n")
    
    def log(self, agent: str, message: str, data: Any = None):
        """Log a debug message with optional data (limited)."""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(f"[{timestamp}] {agent}: {message}\n")
            
            if data is not None:
                # Limit data output to avoid huge logs
                if isinstance(data, str):
                    limited_data = data[:200] + "..." if len(data) > 200 else data
                elif isinstance(data, dict):
                    limited_data = {k: (str(v)[:100] + "..." if len(str(v)) > 100 else v) 
                                  for k, v in list(data.items())[:5]}
                elif isinstance(data, list):
                    limited_data = [str(item)[:100] + "..." if len(str(item)) > 100 else str(item) 
                                  for item in data[:3]]
                else:
                    limited_data = str(data)[:200] + "..." if len(str(data)) > 200 else str(data)
                
                f.write(f"    Data: {limited_data}\n")

# Global debug logger instance
_debug_logger = None

def get_debug_logger() -> DebugLogger:
    """Get or create the global debug logger."""
    global _debug_logger
    if _debug_logger is None:
        _debug_logger = DebugLogger()
    return _debug_logger

def debug_log(agent: str, message: str, data: Any = None):
    """Convenience function for logging."""
    get_debug_logger().log(agent, message, data)
