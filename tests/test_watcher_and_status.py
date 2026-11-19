from pathlib import Path
from types import SimpleNamespace

import index_status
import watcher


class ImmediateEvent:
    def is_set(self):
        return False

    def wait(self, _seconds):
        return False


def test_file_events_coalesce_and_ignore_unsupported_files(tmp_path):
    queue = watcher.IngestionQueue()
    handler = watcher.DocumentHandler(queue)
    source = tmp_path / "notes.txt"
    unsupported = tmp_path / "image.png"

    handler.on_created(SimpleNamespace(is_directory=False, src_path=str(source)))
    handler.on_modified(SimpleNamespace(is_directory=False, src_path=str(source)))
    handler.on_created(SimpleNamespace(is_directory=False, src_path=str(unsupported)))

    assert list(queue._pending) == [str(source)]
    assert queue._pending[str(source)]["action"] == "index"


def test_file_move_queues_source_removal_and_destination_index(tmp_path):
    queue = watcher.IngestionQueue()
    handler = watcher.DocumentHandler(queue)
    source = tmp_path / "old.txt"
    destination = tmp_path / "new.txt"

    handler.on_moved(SimpleNamespace(
        is_directory=False,
        src_path=str(source),
        dest_path=str(destination)
    ))

    assert queue._pending[str(source)]["action"] == "delete"
    assert queue._pending[str(destination)]["action"] == "index"


def test_directory_delete_queues_full_reconciliation(tmp_path):
    queue = watcher.IngestionQueue()
    handler = watcher.DocumentHandler(queue)

    handler.on_deleted(SimpleNamespace(is_directory=True, src_path=str(tmp_path)))

    assert queue._pending[watcher.RECONCILE_KEY]["action"] == "reconcile"


def test_watcher_retries_failed_ingestion(monkeypatch, tmp_path):
    queue = watcher.IngestionQueue()
    source = tmp_path / "new.txt"
    source.write_text("body", encoding="utf-8")
    attempts = []

    monkeypatch.setattr(watcher.IngestionQueue, "_wait_until_stable", staticmethod(lambda *_args: True))
    monkeypatch.setattr(
        watcher,
        "ingest_file",
        lambda _path: attempts.append("attempt") or len(attempts) > 1
    )

    queue._retry_ingestion(source, ImmediateEvent())

    assert len(attempts) == 2


def test_watcher_records_failed_ingestion_after_retry_limit(monkeypatch, tmp_path):
    queue = watcher.IngestionQueue()
    source = tmp_path / "broken.txt"
    source.write_text("body", encoding="utf-8")
    monkeypatch.setattr(watcher, "MAX_ATTEMPTS", 1)
    monkeypatch.setattr(watcher.IngestionQueue, "_wait_until_stable", staticmethod(lambda *_args: True))
    monkeypatch.setattr(watcher, "ingest_file", lambda _path: False)

    queue._retry_ingestion(source, ImmediateEvent())

    assert index_status.get_index_status()[str(source)]["status"] == "failed"


def test_file_stability_wait_returns_after_repeated_stat(monkeypatch, tmp_path):
    source = tmp_path / "stable.txt"
    source.write_text("done", encoding="utf-8")
    monkeypatch.setattr(watcher, "FILE_STABILITY_POLL_SECONDS", 0)

    assert watcher.IngestionQueue._wait_until_stable(source, ImmediateEvent()) is True


def test_index_status_is_written_and_updated_atomically(monkeypatch, tmp_path):
    monkeypatch.setattr(index_status, "STATUS_FILE", tmp_path / "status.json")
    source = tmp_path / "doc.txt"

    index_status.record_index_status(source, "indexing")
    index_status.record_index_status(source, "failed", error="Ollama offline")
    status = index_status.get_index_status()[str(source)]

    assert status["status"] == "failed"
    assert status["error"] == "Ollama offline"
    assert "updated_at" in status
    assert not (tmp_path / "status.tmp").exists()
