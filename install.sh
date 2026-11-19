#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv"

case "$(uname -s)" in
    Darwin|Linux) ;;
    *)
        echo "AI Router currently supports macOS and Linux. Windows is not supported." >&2
        exit 1
        ;;
esac

python_candidates=()
if [[ -n "${AI_ROUTER_PYTHON:-}" ]]; then
    python_candidates+=("${AI_ROUTER_PYTHON}")
else
    python_candidates+=(python3.11 python3)
fi

python_bin=""
for candidate in "${python_candidates[@]}"; do
    if ! command -v "${candidate}" >/dev/null 2>&1; then
        continue
    fi

    version="$("${candidate}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    if [[ "${version}" == "3.11" ]]; then
        python_bin="${candidate}"
        break
    fi
done

if [[ -z "${python_bin}" ]]; then
    echo "Python 3.11 is required. Install it and rerun this script." >&2
    echo "You can select a specific interpreter with AI_ROUTER_PYTHON=/path/to/python3.11." >&2
    exit 1
fi

echo "Creating the virtual environment with ${python_bin}..."
"${python_bin}" -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install -r "${PROJECT_DIR}/requirements-lock.txt"

if [[ ! -f "${PROJECT_DIR}/.env" ]]; then
    cp "${PROJECT_DIR}/.env.example" "${PROJECT_DIR}/.env"
    echo "Created .env from .env.example. Add an OpenAI key only if you plan to use cloud routes."
fi

mkdir -p "${PROJECT_DIR}/documents"

if command -v ollama >/dev/null 2>&1; then
    if ollama list >/dev/null 2>&1; then
        if [[ -t 0 ]]; then
            read -r -p "Download qwen2.5:3b and nomic-embed-text now? [Y/n] " answer
            if [[ -z "${answer}" || "${answer}" =~ ^[Yy]([Ee][Ss])?$ ]]; then
                ollama pull qwen2.5:3b
                ollama pull nomic-embed-text
            else
                echo "Download the models later with: ollama pull qwen2.5:3b && ollama pull nomic-embed-text"
            fi
        else
            echo "Ollama is running. Download the models with: ollama pull qwen2.5:3b && ollama pull nomic-embed-text"
        fi
    else
        echo "Ollama is installed but its service is not responding. Start Ollama, then pull qwen2.5:3b and nomic-embed-text."
    fi
else
    echo "Ollama was not found. Install it from https://ollama.com/download, then pull qwen2.5:3b and nomic-embed-text."
fi

echo
echo "Setup is complete. Add documents under ${PROJECT_DIR}/documents and start AI Router with:"
echo "  ${VENV_DIR}/bin/python ${PROJECT_DIR}/run_app.py"
