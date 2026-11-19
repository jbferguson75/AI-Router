"""Watch the documents folder and serialize reliable index updates."""

import signal
import threading
import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from config import DOCUMENT_DIR
from file_lock import acquire_file_lock
from index_status import record_index_status
from ingest import (
    ingest_documents,
    ingest_file,
    is_supported_file,
    remove_document
)


DEBOUNCE_SECONDS = 1.5
FILE_STABILITY_POLL_SECONDS = 0.5
FILE_STABILITY_TIMEOUT_SECONDS = 30
MAX_ATTEMPTS = 5
RECONCILE_INTERVAL_SECONDS = 300
RECONCILE_KEY = "__reconcile_documents__"


def acquire_watcher_lock():
    lock_path = Path(__file__).resolve().parent / ".document_watcher.lock"
    lock_file = lock_path.open("a+b")
    try:
        acquire_file_lock(lock_file, blocking=False)
    except BlockingIOError:
        lock_file.close()
        print("A document watcher is already running for this project.")
        return None

    return lock_file


class IngestionQueue:
    """Coalesce repeated file events and process one index write at a time."""

    def __init__(self):
        self._condition = threading.Condition()
        self._pending = {}
        self._stopping = False

    def enqueue(self, path, action):
        key = str(path)
        with self._condition:
            self._pending[key] = {
                "path": Path(path),
                "action": action,
                "queued_at": time.monotonic()
            }
            self._condition.notify()

    def enqueue_reconciliation(self):
        self.enqueue(RECONCILE_KEY, "reconcile")

    def stop(self):
        with self._condition:
            self._stopping = True
            self._condition.notify_all()

    def run(self, shutdown_requested):
        while not shutdown_requested.is_set():
            item = self._next_ready_item(shutdown_requested)
            if item is None:
                continue
            self._process(item, shutdown_requested)

    def _next_ready_item(self, shutdown_requested):
        with self._condition:
            while not self._stopping and not shutdown_requested.is_set():
                if not self._pending:
                    self._condition.wait()
                    continue

                now = time.monotonic()
                key, item = min(
                    self._pending.items(),
                    key=lambda pair: pair[1]["queued_at"]
                )
                delay = DEBOUNCE_SECONDS - (now - item["queued_at"])
                if delay > 0:
                    self._condition.wait(delay)
                    continue

                return self._pending.pop(key)
            return None

    def _process(self, item, shutdown_requested):
        path = item["path"]
        action = item["action"]

        if action == "reconcile":
            self._retry_reconciliation(shutdown_requested)
            return

        if action == "delete":
            self._retry_delete(path, shutdown_requested)
            return

        self._retry_ingestion(path, shutdown_requested)

    def _retry_ingestion(self, path, shutdown_requested):
        last_error = None
        for attempt in range(MAX_ATTEMPTS):
            if shutdown_requested.is_set():
                return

            if not path.exists():
                try:
                    remove_document(path)
                    return
                except Exception as exc:
                    last_error = exc
            else:
                try:
                    if self._wait_until_stable(path, shutdown_requested):
                        if ingest_file(path):
                            return
                        last_error = RuntimeError("Ingestion did not complete")
                    else:
                        last_error = RuntimeError(
                            "File did not become stable before timeout"
                        )
                except Exception as exc:
                    last_error = exc

            if attempt + 1 < MAX_ATTEMPTS:
                delay = min(2 ** attempt, 16)
                print(
                    f"Retrying {path.name} in {delay}s "
                    f"(attempt {attempt + 2}/{MAX_ATTEMPTS}): {last_error}"
                )
                if shutdown_requested.wait(delay):
                    return

        error = str(last_error or "Unknown ingestion error")
        record_index_status(path, "failed", error=error)
        print(f"Giving up on {path}: {error}")

    def _retry_delete(self, path, shutdown_requested):
        last_error = None
        for attempt in range(MAX_ATTEMPTS):
            if shutdown_requested.is_set():
                return
            try:
                remove_document(path)
                return
            except Exception as exc:
                last_error = exc
                if attempt + 1 < MAX_ATTEMPTS:
                    delay = min(2 ** attempt, 16)
                    print(
                        f"Retrying removal of {path} in {delay}s "
                        f"(attempt {attempt + 2}/{MAX_ATTEMPTS}): {exc}"
                    )
                    if shutdown_requested.wait(delay):
                        return

        record_index_status(path, "failed", error=last_error)
        print(f"Giving up removing {path}: {last_error}")

    def _retry_reconciliation(self, shutdown_requested):
        for attempt in range(MAX_ATTEMPTS):
            if shutdown_requested.is_set():
                return
            try:
                ingest_documents()
                return
            except Exception as exc:
                if attempt + 1 == MAX_ATTEMPTS:
                    record_index_status(
                        DOCUMENT_DIR,
                        "failed",
                        error=f"Reconciliation failed: {exc}"
                    )
                    print(f"Reconciliation failed: {exc}")
                    return
                delay = min(2 ** attempt, 16)
                print(
                    f"Retrying document reconciliation in {delay}s "
                    f"(attempt {attempt + 2}/{MAX_ATTEMPTS}): {exc}"
                )
                if shutdown_requested.wait(delay):
                    return

    @staticmethod
    def _wait_until_stable(path, shutdown_requested):
        deadline = time.monotonic() + FILE_STABILITY_TIMEOUT_SECONDS
        previous_stat = None
        stable_observations = 0

        while time.monotonic() < deadline:
            if shutdown_requested.is_set():
                return False
            try:
                stat = path.stat()
            except FileNotFoundError:
                return False

            current_stat = (stat.st_size, stat.st_mtime_ns)
            if current_stat == previous_stat:
                stable_observations += 1
                if stable_observations >= 2:
                    return True
            else:
                previous_stat = current_stat
                stable_observations = 0

            if shutdown_requested.wait(FILE_STABILITY_POLL_SECONDS):
                return False

        return False


class DocumentHandler(FileSystemEventHandler):
    def __init__(self, ingestion_queue):
        super().__init__()
        self.ingestion_queue = ingestion_queue

    def on_created(self, event):
        if not event.is_directory and is_supported_file(Path(event.src_path)):
            self.ingestion_queue.enqueue(event.src_path, "index")

    def on_modified(self, event):
        if not event.is_directory and is_supported_file(Path(event.src_path)):
            self.ingestion_queue.enqueue(event.src_path, "index")

    def on_deleted(self, event):
        if event.is_directory:
            self.ingestion_queue.enqueue_reconciliation()
        elif is_supported_file(Path(event.src_path)):
            self.ingestion_queue.enqueue(event.src_path, "delete")

    def on_moved(self, event):
        if event.is_directory:
            self.ingestion_queue.enqueue_reconciliation()
            return

        if is_supported_file(Path(event.src_path)):
            self.ingestion_queue.enqueue(event.src_path, "delete")
        if is_supported_file(Path(event.dest_path)):
            self.ingestion_queue.enqueue(event.dest_path, "index")


def main():
    watcher_lock = acquire_watcher_lock()
    if watcher_lock is None:
        return 1

    document_root = Path(DOCUMENT_DIR)
    document_root.mkdir(parents=True, exist_ok=True)

    print("DOCUMENT WATCHER STARTED")
    print(f"Watching: {document_root.resolve()}")

    shutdown_requested = threading.Event()

    def request_shutdown(_signum, _frame):
        shutdown_requested.set()

    signal.signal(signal.SIGINT, request_shutdown)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, request_shutdown)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, request_shutdown)

    ingestion_queue = IngestionQueue()
    observer = Observer()
    observer.schedule(
        DocumentHandler(ingestion_queue),
        str(document_root),
        recursive=True
    )
    observer.start()

    worker = None
    try:
        print("Reconciling documents with the local index...")
        try:
            ingest_documents()
        except Exception as exc:
            print(f"Initial reconciliation failed: {exc}")
            ingestion_queue.enqueue_reconciliation()

        worker = threading.Thread(
            target=ingestion_queue.run,
            args=(shutdown_requested,),
            name="document-ingestion-worker",
            daemon=True
        )
        worker.start()

        next_reconciliation = time.monotonic() + RECONCILE_INTERVAL_SECONDS
        while not shutdown_requested.wait(1):
            if not observer.is_alive():
                raise RuntimeError("Filesystem observer stopped unexpectedly")
            if time.monotonic() >= next_reconciliation:
                ingestion_queue.enqueue_reconciliation()
                next_reconciliation = (
                    time.monotonic() + RECONCILE_INTERVAL_SECONDS
                )
    except KeyboardInterrupt:
        shutdown_requested.set()
    finally:
        shutdown_requested.set()
        ingestion_queue.stop()
        observer.stop()
        observer.join()
        if worker is not None:
            worker.join(timeout=15)


if __name__ == "__main__":
    main()
