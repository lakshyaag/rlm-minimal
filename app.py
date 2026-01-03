"""
Flask web application for RLM interactive interface.
"""

import os
import json
import queue
import threading
from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    Response,
    stream_with_context,
)
from dotenv import load_dotenv
from rlm.rlm_web import RLM_WEB

# Load environment variables
load_dotenv()

app = Flask(__name__)


@app.route("/")
def index():
    """Render the main page."""
    return render_template("index.html")


@app.route("/api/query", methods=["POST"])
def query():
    """Handle RLM query requests with Server-Sent Events streaming."""
    try:
        data = request.json
        context = data.get("context", "")
        query_text = data.get("query", "")
        model = data.get("model", "gpt-4o-mini")
        recursive_model = data.get("recursive_model", "gpt-4o-mini")
        max_iterations = int(data.get("max_iterations", 10))

        if not context or not query_text:
            return jsonify({"error": "Context and query are required"}), 400

        def generate():
            """Generator function for Server-Sent Events."""
            from dataclasses import asdict

            events_queue = queue.Queue()
            final_answer = None
            error_occurred = None

            def event_callback(event):
                """Callback to capture events as they happen."""
                events_queue.put(event)

            def run_rlm():
                """Run RLM in a separate thread."""
                nonlocal final_answer, error_occurred
                try:
                    # Initialize RLM
                    rlm = RLM_WEB(
                        base_url="https://api.pinference.ai/api/v1",
                        api_key=os.getenv("OPENAI_API_KEY"),
                        model=model,
                        recursive_model=recursive_model,
                        max_iterations=max_iterations,
                    )

                    # Run the query with streaming
                    result = rlm.completion(
                        context=context, query=query_text, event_callback=event_callback
                    )

                    final_answer = result["answer"]
                    events_queue.put(None)  # Signal completion
                except Exception as e:
                    error_occurred = str(e)
                    events_queue.put(None)  # Signal completion

            # Start RLM in a thread
            rlm_thread = threading.Thread(target=run_rlm)
            rlm_thread.start()

            # Stream events as they come
            while True:
                try:
                    event = events_queue.get(timeout=1)
                    if event is None:  # Completion signal
                        break

                    # Send event via SSE
                    event_dict = (
                        event.to_dict() if hasattr(event, "to_dict") else asdict(event)
                    )
                    yield f"data: {json.dumps(event_dict)}\n\n"
                except queue.Empty:
                    continue

            # Wait for thread to finish
            rlm_thread.join()

            if error_occurred:
                yield f"data: {json.dumps({'type': 'error', 'error': error_occurred})}\n\n"
            else:
                # Send completion signal
                yield f"data: {json.dumps({'type': 'complete', 'answer': final_answer})}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
