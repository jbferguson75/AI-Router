"""Small shared status file for the watcher and Streamlit UI."""

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path


STATUS_FILE = Path(".index_status.json")
_STATUS_LOCK = threading.Lock()


def get_index_status():
    try:
        with STATUS_FILE.open("r", encoding="utf-8") as status_file:
            data = json.load(status_file)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def record_index_status(path, status, error=None, chunk_count=None):
    """Atomically record the latest indexing state for one source file."""
    path_key = str(path)
    with _STATUS_LOCK:
        statuses = get_index_status()
        entry = {
            "status": status,
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        if error:
            entry["error"] = str(error)[:500]
        if chunk_count is not None:
            entry["chunk_count"] = chunk_count
        statuses[path_key] = entry

        # Keep this operational file bounded over long-running sessions.
        statuses = dict(
            sorted(
                statuses.items(),
                key=lambda item: item[1].get("updated_at", ""),
                reverse=True
            )[:500]
        )

        temporary_file = STATUS_FILE.with_suffix(".tmp")
        try:
            with temporary_file.open("w", encoding="utf-8") as status_file:
                json.dump(statuses, status_file, indent=2)
            os.replace(temporary_file, STATUS_FILE)
        finally:
            if temporary_file.exists():
                temporary_file.unlink()
