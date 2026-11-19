import hashlib
from pathlib import Path

import ingest
import index_status


def set_test_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(ingest, "HASH_CACHE_FILE", tmp_path / "hash-cache.json")
    monkeypatch.setattr(ingest, "INDEX_LOCK_FILE", tmp_path / "index-write.lock")
    monkeypatch.setattr(index_status, "STATUS_FILE", tmp_path / "index-status.json")
    ingest.hash_cache.clear()


def test_chunk_text_overlaps_chunks():
    original_size = ingest.CHUNK_SIZE
    original_overlap = ingest.CHUNK_OVERLAP
    ingest.CHUNK_SIZE = 6
    ingest.CHUNK_OVERLAP = 2
    try:
        assert ingest.chunk_text("abcdefghij") == ["abcdef", "efghij", "ij"]
    finally:
        ingest.CHUNK_SIZE = original_size
        ingest.CHUNK_OVERLAP = original_overlap


def test_supported_file_filter_ignores_hidden_and_temp_files():
    assert ingest.is_supported_file(Path("notes.txt"))
    assert not ingest.is_supported_file(Path(".notes.txt"))
    assert not ingest.is_supported_file(Path("~$notes.docx"))
    assert not ingest.is_supported_file(Path("notes.exe"))


def test_ingest_replaces_old_chunks_only_after_embedding_success(
    monkeypatch, tmp_path, document_collection
):
    set_test_paths(monkeypatch, tmp_path)
    source = tmp_path / "car.txt"
    source.write_text("first version", encoding="utf-8")
    monkeypatch.setattr(ingest, "extract_text", lambda _path: "first version")
    monkeypatch.setattr(ingest, "get_embedding", lambda text: [float(len(text)), 1.0])

    assert ingest.ingest_file(source) is True
    original_ids = document_collection.get()["ids"]
    assert len(original_ids) == 1

    source.write_text("replacement version", encoding="utf-8")
    monkeypatch.setattr(ingest, "extract_text", lambda _path: "replacement version")
    monkeypatch.setattr(
        ingest,
        "get_embedding",
        lambda _text: (_ for _ in ()).throw(RuntimeError("embedding offline"))
    )

    assert ingest.ingest_file(source) is False
    assert document_collection.get()["ids"] == original_ids
    assert document_collection.get()["documents"] == ["first version"]
    assert index_status.get_index_status()[str(source.resolve())]["status"] == "failed"


def test_ingest_replaces_chunks_after_success_and_skips_unchanged_file(
    monkeypatch, tmp_path, document_collection
):
    set_test_paths(monkeypatch, tmp_path)
    source = tmp_path / "project.txt"
    source.write_text("first", encoding="utf-8")
    text = {"value": "first"}
    monkeypatch.setattr(ingest, "extract_text", lambda _path: text["value"])
    monkeypatch.setattr(ingest, "get_embedding", lambda _text: [1.0, 0.0])

    assert ingest.ingest_file(source)
    text["value"] = "second version"
    source.write_text("second version", encoding="utf-8")

    assert ingest.ingest_file(source)
    assert document_collection.get()["documents"] == ["second version"]
    assert ingest.hash_cache[str(source.resolve())]["hash"] == hashlib.md5(
        source.read_bytes()
    ).hexdigest()
    assert ingest.ingest_file(source) is True
    assert document_collection.get()["documents"] == ["second version"]


def test_same_hash_reindexes_if_index_chunks_are_missing(
    monkeypatch, tmp_path, document_collection
):
    set_test_paths(monkeypatch, tmp_path)
    source = tmp_path / "lost-index.txt"
    source.write_text("recover me", encoding="utf-8")
    current_hash = ingest.calculate_file_hash(source)
    ingest.hash_cache[str(source.resolve())] = {
        "hash": current_hash,
        "chunk_count": 1
    }
    monkeypatch.setattr(ingest, "extract_text", lambda _path: "recover me")
    monkeypatch.setattr(ingest, "get_embedding", lambda _text: [1.0, 0.0])

    assert ingest.ingest_file(source)
    assert document_collection.count() == 1


def test_zero_byte_cloud_placeholder_preserves_old_chunks(
    monkeypatch, tmp_path, document_collection
):
    set_test_paths(monkeypatch, tmp_path)
    source = tmp_path / "empty.txt"
    source.write_text("old content", encoding="utf-8")
    monkeypatch.setattr(ingest, "extract_text", lambda _path: "old content")
    monkeypatch.setattr(ingest, "get_embedding", lambda _text: [1.0, 0.0])
    assert ingest.ingest_file(source)
    assert document_collection.count() == 1

    source.write_text("", encoding="utf-8")
    assert ingest.ingest_file(source)
    assert document_collection.count() == 1
    assert ingest.hash_cache[str(source.resolve())]["hash"] == hashlib.md5(
        b"old content"
    ).hexdigest()
    status = index_status.get_index_status()[str(source.resolve())]
    assert status["status"] == "ready"
    assert status["availability"] == "online_only"


def test_downloaded_changed_file_replaces_index_after_placeholder(
    monkeypatch, tmp_path, document_collection
):
    set_test_paths(monkeypatch, tmp_path)
    source = tmp_path / "cloud-note.txt"
    source.write_text("original cloud content", encoding="utf-8")
    embeddings = []
    monkeypatch.setattr(
        ingest,
        "extract_text",
        lambda path: Path(path).read_text(encoding="utf-8")
    )
    monkeypatch.setattr(
        ingest,
        "get_embedding",
        lambda text: embeddings.append(text) or [float(len(text)), 1.0]
    )

    assert ingest.ingest_file(source)
    original_hash = ingest.hash_cache[str(source.resolve())]["hash"]
    original_ids = document_collection.get()["ids"]
    assert document_collection.get()["documents"] == ["original cloud content"]

    source.write_text("", encoding="utf-8")
    assert ingest.ingest_file(source)
    assert document_collection.get()["ids"] == original_ids
    assert document_collection.get()["documents"] == ["original cloud content"]
    assert ingest.hash_cache[str(source.resolve())]["hash"] == original_hash
    assert index_status.get_index_status()[str(source.resolve())]["availability"] == "online_only"

    source.write_text("updated cloud content", encoding="utf-8")
    assert ingest.ingest_file(source)

    updated_hash = hashlib.md5(source.read_bytes()).hexdigest()
    status = index_status.get_index_status()[str(source.resolve())]
    assert document_collection.get()["documents"] == ["updated cloud content"]
    assert document_collection.get()["ids"] != original_ids
    assert ingest.hash_cache[str(source.resolve())]["hash"] == updated_hash
    assert status["status"] == "ready"
    assert "availability" not in status
    assert embeddings == ["original cloud content", "updated cloud content"]


def test_reconciliation_prunes_deleted_files(monkeypatch, tmp_path, document_collection):
    set_test_paths(monkeypatch, tmp_path)
    root = tmp_path / "documents"
    root.mkdir()
    source = root / "gone.txt"
    source.write_text("indexed text", encoding="utf-8")
    monkeypatch.setattr(ingest, "DOCUMENT_DIR", root)
    monkeypatch.setattr(ingest, "extract_text", lambda _path: "indexed text")
    monkeypatch.setattr(ingest, "get_embedding", lambda _text: [1.0, 0.0])
    assert ingest.ingest_file(source)
    source.unlink()

    ingest.ingest_documents()

    assert document_collection.count() == 0
    assert str(source.resolve()) not in ingest.hash_cache
