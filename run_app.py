"""Run the Streamlit UI and document watcher as one local application."""

from __future__ import annotations

import atexit
import fcntl
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
shutdown_signal = None
shutdown_requested = threading.Event()


def request_shutdown(signum, _frame):
    global shutdown_signal
    shutdown_signal = signum
    shutdown_requested.set()


def stop_process(process, name):
    if process is None or process.poll() is not None:
        return

    print(f"Stopping {name}...")
    try:
        process.send_signal(signal.SIGTERM)
    except ProcessLookupError:
        return

    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        print(f"{name} did not stop in time; forcing shutdown.")
        process.kill()
        process.wait()


def acquire_instance_lock():
    lock_file = (PROJECT_DIR / ".ai_router.lock").open("a+")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_file.close()
        print("AI Router is already running from this project directory.")
        return None

    lock_file.seek(0)
    lock_file.truncate()
    lock_file.write(f"{os.getpid()}\n")
    lock_file.flush()
    atexit.register(lock_file.close)
    return lock_file


def main():
    global shutdown_signal

    instance_lock = acquire_instance_lock()
    if instance_lock is None:
        return 1

    watcher_script = PROJECT_DIR / "watcher.py"
    app_script = PROJECT_DIR / "app.py"

    if not watcher_script.is_file() or not app_script.is_file():
        print("Could not find watcher.py and app.py beside run_app.py.")
        return 1

    signal.signal(signal.SIGINT, request_shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, request_shutdown)

    watcher = None
    streamlit = None
    exit_code = 0

    try:
        watcher = subprocess.Popen(
            [sys.executable, str(watcher_script)],
            cwd=PROJECT_DIR
        )
        print("Document watcher started.")

        streamlit = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(app_script)
            ],
            cwd=PROJECT_DIR
        )
        print("Streamlit app starting at http://localhost:8501")

        while not shutdown_requested.wait(0.5):
            watcher_code = watcher.poll()
            streamlit_code = streamlit.poll()

            if watcher_code is not None:
                print(f"Document watcher exited with status {watcher_code}.")
                exit_code = watcher_code or 1
                break

            if streamlit_code is not None:
                exit_code = streamlit_code
                break

    except OSError as exc:
        print(f"Could not start the application: {exc}")
        exit_code = 1
    finally:
        stop_process(streamlit, "Streamlit")
        stop_process(watcher, "document watcher")

    if shutdown_signal == signal.SIGINT:
        return 130
    if shutdown_signal == getattr(signal, "SIGTERM", None):
        return 143
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
