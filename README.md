# AI Router

AI Router is a local Streamlit app that searches documents indexed on your computer and routes questions to local Ollama models. It can also send approved prompts and document context to OpenAI when cloud routing is enabled.

## Requirements

- macOS, Linux, or Windows
- Python 3.11
- [Ollama](https://ollama.com/download) installed and running
- The Ollama models `qwen2.5:3b` and `nomic-embed-text`
- An OpenAI API key only if you plan to use cloud routes

## Install

Clone the repository into a folder named `ai-router` and run the installer for your operating system.

### macOS and Linux

```bash
git clone https://github.com/jbferguson75/AI-Router.git ai-router
cd ai-router
./install.sh
```

### Windows

In PowerShell, run:

```powershell
git clone https://github.com/jbferguson75/AI-Router.git ai-router
cd ai-router
powershell -NoProfile -ExecutionPolicy Bypass -File .\install.ps1
```

The installer creates a local `.venv`, installs the pinned runtime dependencies from `requirements-lock.txt`, creates `.env` from `.env.example` if needed, and creates the `documents/` folder. It does not install Ollama or overwrite an existing `.env`.

If the Ollama service is running, the installer offers to download the two required models. You can decline and pull them later:

```bash
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

If Ollama is not installed or running, install/open Ollama and run those commands before starting AI Router.

## Choose Ollama models

You can use other models supported by Ollama. Set the model names for each role in `config.py`: `CHAT_MODEL`, `EMBED_MODEL`, `ROUTER_MODEL`, `EVALUATOR_MODEL`, `MEMORY_EXTRACTOR_MODEL`, and `MEMORY_SCORER_MODEL`. Pull each selected model with `ollama pull <model-name>` before running the app. Choose an embedding model for `EMBED_MODEL` and chat-capable models for the other roles. The installer only pulls the default models listed above.

## Configure

Edit `.env` and set `OPENAI_API_KEY` if you want cloud routing:

```dotenv
OPENAI_API_KEY=your-api-key
```

Local document search and answers use Ollama. Requests sent to OpenAI require the app's privacy review and explicit approval. Keep `.env` private; it is excluded from Git.

Add your own `.txt`, `.md`, `.pdf`, `.docx`, `.pptx`, or `.xlsx` files under `documents/`. The watcher indexes new and changed files automatically. The documents and Chroma database stay on the local machine and are excluded from Git.

## Run

From the project directory, start the app using the command for your operating system:

```bash
./.venv/bin/python run_app.py
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe .\run_app.py
```

Open [http://localhost:8501](http://localhost:8501). Stop the app with Ctrl+C in the terminal.

## Development

Install development tools into the environment created by the installer and run the offline test suite:

```bash
./.venv/bin/python -m pip install -r requirements-dev.txt
./.venv/bin/python -m pytest -q
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

Tests use a fake in-memory Chroma database and mock model/API calls; they do not need Ollama or an OpenAI key.

## Configuration

Application models, routing thresholds, and behavior switches are in `config.py`. Runtime dependencies are listed in `requirements.txt`; `requirements-lock.txt` pins the resolved versions used by both installers. The lock excludes `uvloop` on Windows, where it is not supported. Regenerate the lock when changing runtime dependencies, then validate installation on Python 3.11 for macOS, Linux, and Windows.
