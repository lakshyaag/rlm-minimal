# Recursive Language Models (minimal version) 

[Link to the original blogpost 📝](https://alexzhang13.github.io/blog/2025/rlm/)

I received a lot of requests to put out a notebook or gist version of the codebase I've been using. Sadly it's a bit entangled with a bunch of random state, cost, and code execution tracking logic that I want to clean up while I run other experiments. In the meantime, I've re-written a simpler version of what I'm using so people can get started building on top and writing their own RLM implementations. Happy hacking!

![teaser](media/rlm.png)

I've provided a basic, minimal implementation of a recursive language model (RLM) with a REPL environment for OpenAI clients. Like the blogpost, we only implement recursive sub-calls with `depth=1` inside the RLM environment. Enabling further depths is as simple as replacing the `Sub_RLM` class with the `RLM_REPL` class, but you may need to finagle the `exec`-based REPL environments to work better here (because now your sub-RLMs have their own REPL environments!).

In this stripped implementation, we exclude a lot of the logging, cost tracking, prompting, and REPL execution details of the experiments run in the blogpost. It's relatively easy to modify and build on top of this code to reproduce those results, but it's currently harder to go from my full codebase to supporting any new functionality.

## Installation

Install the required dependencies using `uv`:

```bash
uv venv
uv pip install -r requirements.txt
```

Make sure you have your OpenAI-compatible API key and base URL set in a `.env` file:

```bash
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=your_base_url_here
```

## Basic Example
We have all the basic dependencies in `requirements.txt`, although none are really necessary if you change your implementation (`openai` for LM API calls, `dotenv` for .env loading, `rich` for logging, and `flask` for the web interface).

In `main.py`, we have a basic needle-in-the-haystack (NIAH) example that embeds a random number inside ~1M lines of random words, and asks the model to go find it. It's a silly Hello World type example to emphasize that `RLM.completion()` calls are meant to replace `LM.completion()` calls.

## Web Interface

The project includes a web-based interactive interface for exploring RLM behavior in real-time.

### Running the Web App

1. **Start the Flask server:**
   ```bash
   uv run app.py
   ```

2. **Open your browser:**
   Navigate to `http://localhost:5000`

3. **Use the interface:**
   - Enter your context in the "Context" field
   - Enter your query in the "Query" field
   - Select your Root LM and Recursive LM models (change the model list in `templates/index.html`)
   - Set the maximum number of iterations
   - Click "Run Query" to see the recursive RLM process in action

## Code Structure
In the `rlm/` folder, the two main files are `rlm_repl.py` and `repl.py`. 
* `rlm_repl.py` offers a basic implementation of an RLM using a REPL environment in the `RLM_REPL` class. The `completion()` function gets called when we query an RLM.
* `repl.py` is a simple `exec`-based implementation of a REPL environment that adds an LM sub-call function. To make the system truly recursive beyond `depth=1`, you can replace the `Sub_RLM` class with `RLM_REPL` (they all inherit from the `RLM` base class).
* `rlm_web.py` provides a web-compatible version of the RLM that streams events for the web interface.

The functionality for parsing and handling base LLM clients are all in `rlm/utils/`. We also add example prompts here.

### Web App Files
* `app.py` - Flask web server that provides the API and serves the web interface
* `templates/index.html` - The web interface frontend with real-time event streaming
* `rlm/logger/web_logger.py` - Event logger that captures RLM execution events for web display

> The `rlm/logger/` folder mainly contains optional logging utilities used by the RLM REPL implementation. If you want to enable colorful or enhanced logging outputs, you may need to install the [`rich`](https://github.com/Textualize/rich) library as a dependency.
```bash
uv pip install rich
```

When you run your code, you'll see something like this:

![Example logging output using `rich`](media/rich.png)