"""
Web logger that captures events for display in web interface.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class Event:
    """Represents a single event in the RLM execution."""

    type: str  # 'query_start', 'model_response', 'code_execution', 'repl_output', 'final_answer', 'error'
    timestamp: str
    data: Dict[str, Any]
    step: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for JSON serialization."""
        return asdict(self)


class WebLogger:
    """Logger that captures events for web display instead of printing."""

    def __init__(self):
        self.events: List[Event] = []
        self.conversation_step = 0
        self.current_query = ""

    def log_query_start(self, query: str):
        """Log the start of a new query."""
        self.current_query = query
        self.conversation_step = 0
        self.events.clear()

        self.events.append(
            Event(
                type="query_start",
                timestamp=datetime.now().isoformat(),
                data={"query": query},
            )
        )

    def log_model_response(self, response: str, has_tool_calls: bool):
        """Log the model's response."""
        self.conversation_step += 1

        self.events.append(
            Event(
                type="model_response",
                timestamp=datetime.now().isoformat(),
                step=self.conversation_step,
                data={
                    "response": response,
                    "has_tool_calls": has_tool_calls,
                },
            )
        )

    def log_code_execution(
        self,
        code: str,
        stdout: str,
        stderr: str = "",
        execution_time: Optional[float] = None,
    ):
        """Log code execution in REPL."""
        self.events.append(
            Event(
                type="code_execution",
                timestamp=datetime.now().isoformat(),
                step=self.conversation_step,
                data={
                    "code": code,
                    "stdout": stdout,
                    "stderr": stderr,
                    "execution_time": execution_time,
                },
            )
        )

    def log_repl_output(self, output: str):
        """Log REPL environment output."""
        self.events.append(
            Event(
                type="repl_output",
                timestamp=datetime.now().isoformat(),
                step=self.conversation_step,
                data={"output": output},
            )
        )

    def log_final_response(self, response: str):
        """Log the final response."""
        self.events.append(
            Event(
                type="final_answer",
                timestamp=datetime.now().isoformat(),
                data={"answer": response},
            )
        )

    def log_error(self, error: str):
        """Log an error."""
        self.events.append(
            Event(
                type="error",
                timestamp=datetime.now().isoformat(),
                data={"error": error},
            )
        )

    def log_tool_execution(self, tool_name: str, result: str):
        """Log tool execution (for compatibility with utils.check_for_final_answer)."""
        self.events.append(
            Event(
                type="tool_execution",
                timestamp=datetime.now().isoformat(),
                step=self.conversation_step,
                data={"tool": tool_name, "result": result},
            )
        )

    def get_events(self) -> List[Dict[str, Any]]:
        """Get all events as dictionaries."""
        return [asdict(event) for event in self.events]

    def clear(self):
        """Clear all events."""
        self.events.clear()
        self.conversation_step = 0
