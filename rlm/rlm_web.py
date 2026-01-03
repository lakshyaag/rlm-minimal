"""
RLM wrapper for web interface that uses WebLogger.
"""

from typing import Dict, List, Optional, Any
from rlm import RLM
from rlm.repl import REPLEnv
from rlm.utils.llm import OpenAIClient
from rlm.utils.prompts import DEFAULT_QUERY, next_action_prompt, build_system_prompt
import rlm.utils.utils as utils
from rlm.logger.web_logger import WebLogger
from rlm.logger.repl_logger import REPLEnvLogger


class RLM_WEB(RLM):
    """
    RLM implementation for web interface with event logging.
    """

    def __init__(
        self,
        base_url: str = "https://api.openai.com/v1",
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        recursive_model: str = "gpt-4o-mini",
        max_iterations: int = 20,
        depth: int = 0,
    ):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.recursive_model = recursive_model
        self.llm = OpenAIClient(base_url, api_key, model)

        self.repl_env = None
        self.depth = depth
        self._max_iterations = max_iterations

        # Use WebLogger instead of ColorfulLogger
        self.logger = WebLogger()
        self.repl_env_logger = REPLEnvLogger(
            enabled=False
        )  # Disable console output, but still log executions

        self.messages = []
        self.query = None

    def setup_context(
        self,
        context: List[str] | str | List[Dict[str, str]],
        query: Optional[str] = None,
    ):
        """Setup the context for the RLMClient."""
        if query is None:
            query = DEFAULT_QUERY

        self.query = query
        self.logger.log_query_start(query)
        if hasattr(self, "event_callback") and self.event_callback:
            # Send query start event
            event = self.logger.events[-1]
            self.event_callback(event)

        # Initialize the conversation with the REPL prompt
        self.messages = build_system_prompt()

        # Initialize REPL environment with context data
        context_data, context_str = utils.convert_context_for_repl(context)

        self.repl_env = REPLEnv(
            base_url=self.base_url,
            context_json=context_data,
            context_str=context_str,
            recursive_model=self.recursive_model,
        )

        return self.messages

    def completion(
        self,
        context: List[str] | str | List[Dict[str, str]],
        query: Optional[str] = None,
        event_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Given a query and a (potentially long) context, recursively call the LM
        to explore the context and provide an answer using a REPL environment.

        Returns a dictionary with 'answer' and 'events'.
        """
        self.event_callback = event_callback
        self.messages = self.setup_context(context, query)

        # Main loop runs for fixed # of root LM iterations
        for iteration in range(self._max_iterations):
            # Query root LM to interact with REPL environment
            response = self.llm.completion(
                self.messages + [next_action_prompt(query, iteration)]
            )

            # Check for code blocks
            code_blocks = utils.find_code_blocks(response)
            self.logger.log_model_response(
                response, has_tool_calls=code_blocks is not None
            )
            # Stream model response event
            if self.event_callback:
                event = self.logger.events[-1]
                self.event_callback(event)

            # Process code execution or add assistant message
            if code_blocks is not None:
                # Track how many executions we had before
                previous_execution_count = len(self.repl_env_logger.executions)

                # Process code execution (this executes the code)
                # We use a disabled console logger to avoid console output
                from rlm.logger.root_logger import ColorfulLogger

                console_logger = ColorfulLogger(enabled=False)

                self.messages = utils.process_code_execution(
                    response,
                    self.messages,
                    self.repl_env,
                    self.repl_env_logger,
                    console_logger,
                )

                # Log only the new code executions to web logger
                new_executions = self.repl_env_logger.executions[
                    previous_execution_count:
                ]
                for execution in new_executions:
                    self.logger.log_code_execution(
                        code=execution.code,
                        stdout=execution.stdout,
                        stderr=execution.stderr,
                        execution_time=execution.execution_time,
                    )
                    # Stream code execution event
                    if self.event_callback:
                        event = self.logger.events[-1]
                        self.event_callback(event)

            else:
                # Add assistant message when there are no code blocks
                assistant_message = {
                    "role": "assistant",
                    "content": "You responded with:\n" + response,
                }
                self.messages.append(assistant_message)

            # Check that model produced a final answer
            final_answer = utils.check_for_final_answer(
                response,
                self.repl_env,
                self.logger,
            )

            if final_answer:
                self.logger.log_final_response(final_answer)
                # Stream final answer event
                if self.event_callback:
                    event = self.logger.events[-1]
                    self.event_callback(event)
                return {"answer": final_answer, "events": self.logger.get_events()}

        # If we reach here, no final answer was found in any iteration
        self.messages.append(next_action_prompt(query, iteration, final_answer=True))
        final_answer = self.llm.completion(self.messages)
        self.logger.log_final_response(final_answer)
        # Stream final answer event
        if self.event_callback:
            event = self.logger.events[-1]
            self.event_callback(event)

        return {"answer": final_answer, "events": self.logger.get_events()}

    def cost_summary(self) -> Dict[str, Any]:
        """Get the cost summary of the Root LM + Sub-RLM Calls."""
        raise NotImplementedError("Cost tracking not implemented for RLM WEB.")

    def reset(self):
        """Reset the (REPL) environment and message history."""
        self.repl_env = REPLEnv()
        self.messages = []
        self.query = None
        self.logger.clear()
