import hashlib
import fcntl
import json
import os
import time
import uuid

from contextlib import contextmanager
from pathlib import Path
from zipfile import BadZipFile

import chromadb
from ollama_client import embed as ollama_embed

from pypdf import PdfReader
from docx import Document
from pptx import Presentation
from openpyxl import load_workbook
from index_status import record_index_status

from config import (
    DOCUMENT_DIR,
    CHROMA_DB_DIR,
    DOCUMENT_COLLECTION,
    EMBED_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    ALLOWED_EXTENSIONS
)

# --------------------------------------------------
# Constants
# --------------------------------------------------

HASH_CACHE_FILE = ".file_hash_cache.json"
INDEX_LOCK_FILE = ".document_index.lock"

# --------------------------------------------------
# Chroma
# --------------------------------------------------

client = chromadb.PersistentClient(
    path=CHROMA_DB_DIR
)

collection = client.get_or_create_collection(
    DOCUMENT_COLLECTION
)

# --------------------------------------------------
# Hash Cache
# --------------------------------------------------

def load_hash_cache():

    cache_file = Path(
        HASH_CACHE_FILE
    )

    if not cache_file.exists():

        return {}

    try:

        with open(
            cache_file,
            "r",
            encoding="utf-8"
        ) as f:

            return json.load(f)

    except Exception:

        return {}


def save_hash_cache(cache):
    cache_file = Path(HASH_CACHE_FILE)
    temporary_file = cache_file.with_suffix(".tmp")
    with temporary_file.open("w", encoding="utf-8") as cache_file_handle:
        json.dump(cache, cache_file_handle, indent=2)
    os.replace(temporary_file, cache_file)


hash_cache = load_hash_cache()


@contextmanager
def index_write_lock():
    lock_path = Path(INDEX_LOCK_FILE)
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def refresh_hash_cache():
    hash_cache.clear()
    hash_cache.update(load_hash_cache())

# --------------------------------------------------
# File Filtering
# --------------------------------------------------

def is_supported_file(path):

    if path.name.startswith("."):
        return False

    if path.name.startswith("~$"):
        return False

    return (
        path.suffix.lower()
        in ALLOWED_EXTENSIONS
    )

# --------------------------------------------------
# Embeddings
# --------------------------------------------------

def get_embedding(text):

    response = ollama_embed(
        model=EMBED_MODEL,
        input=text
    )

    return response["embeddings"][0]

# --------------------------------------------------
# Hashing
# --------------------------------------------------

def calculate_file_hash(path):

    with open(path, "rb") as f:

        return hashlib.md5(
            f.read()
        ).hexdigest()

# --------------------------------------------------
# Existing Chunk Lookup
# --------------------------------------------------

def get_document_chunk_ids(path):

    source = str(Path(path).resolve())

    results = collection.get(
        where={"source": source},
        include=["metadatas"]
    )

    return results.get("ids", [])


def delete_document_chunks(path):
    delete_ids = get_document_chunk_ids(path)

    if delete_ids:
        collection.delete(ids=delete_ids)

        print(
            f"Removed {len(delete_ids)} chunks for {path.name}"
        )


def remove_document(path):
    """Remove a deleted source from Chroma and the local hash cache."""
    with index_write_lock():
        refresh_hash_cache()
        _remove_document_unlocked(path)


def _remove_document_unlocked(path):
    source = str(Path(path).resolve())
    delete_document_chunks(Path(path))
    hash_cache.pop(source, None)
    save_hash_cache(hash_cache)
    record_index_status(source, "removed")


def remove_missing_documents(present_sources):
    """Prune indexed files that no longer exist in the documents folder."""
    with index_write_lock():
        refresh_hash_cache()
        return _remove_missing_documents_unlocked(present_sources)


def _remove_missing_documents_unlocked(present_sources):
    present_sources = {
        str(Path(source).resolve())
        for source in present_sources
    }
    results = collection.get(include=["metadatas"])
    ids_to_delete = []
    metadata_sources = set()

    for chunk_id, metadata in zip(
        results.get("ids", []),
        results.get("metadatas", [])
    ):
        source = metadata.get("source") if metadata else None
        if not source:
            continue
        metadata_sources.add(source)
        normalized_source = str(Path(source).resolve())
        if normalized_source not in present_sources or source != normalized_source:
            ids_to_delete.append(chunk_id)

    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
        print(f"Removed {len(ids_to_delete)} chunks for deleted files")

    missing_sources = {
        source for source in metadata_sources
        if str(Path(source).resolve()) not in present_sources
    }
    for source in missing_sources:
        record_index_status(source, "removed")

    normalized_cache = {}
    for source, entry in hash_cache.items():
        normalized_source = str(Path(source).resolve())
        if normalized_source in present_sources:
            normalized_cache[normalized_source] = entry
    hash_cache.clear()
    hash_cache.update(normalized_cache)
    save_hash_cache(hash_cache)

    return missing_sources

# --------------------------------------------------
# Document Readers
# --------------------------------------------------

def read_pdf(path):

    text = ""

    reader = PdfReader(path)

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:

            text += page_text
            text += "\n"

    return text


def read_text(path):

    with open(
        path,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as f:

        return f.read()


def read_docx(path):

    max_attempts = 5

    for attempt in range(max_attempts):

        try:

            doc = Document(path)

            text = []

            for paragraph in doc.paragraphs:

                if paragraph.text.strip():

                    text.append(
                        paragraph.text
                    )

            return "\n".join(text)

        except BadZipFile:

            print(
                f"DOCX not ready "
                f"(attempt {attempt + 1})"
            )

            time.sleep(2)

    print(
        f"Failed to read DOCX: {path}"
    )

    return None


def read_pptx(path):

    max_attempts = 5

    for attempt in range(max_attempts):

        try:

            presentation = Presentation(path)

            text = []

            for slide in presentation.slides:

                for shape in slide.shapes:

                    if hasattr(shape, "text"):

                        value = shape.text.strip()

                        if value:

                            text.append(value)

            return "\n".join(text)

        except BadZipFile:

            print(
                f"PPTX not ready "
                f"(attempt {attempt + 1})"
            )

            time.sleep(2)

    print(
        f"Failed to read PPTX: {path}"
    )

    return None


def read_xlsx(path):

    max_attempts = 5

    for attempt in range(max_attempts):

        try:

            workbook = load_workbook(
                path,
                data_only=True
            )

            text = []

            for worksheet in workbook.worksheets:

                text.append(
                    f"Worksheet: "
                    f"{worksheet.title}"
                )

                for row in worksheet.iter_rows(
                    values_only=True
                ):

                    values = []

                    for cell in row:

                        if cell is not None:

                            values.append(
                                str(cell)
                            )

                    if values:

                        text.append(
                            " | ".join(values)
                        )

            return "\n".join(text)

        except BadZipFile:

            print(
                f"XLSX not ready "
                f"(attempt {attempt + 1})"
            )

            time.sleep(2)

    print(
        f"Failed to read XLSX: {path}"
    )

    return None

# --------------------------------------------------
# Text Extraction
# --------------------------------------------------

def extract_text(path):

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return read_pdf(path)

    if suffix in [".txt", ".md"]:
        return read_text(path)

    if suffix == ".docx":
        return read_docx(path)

    if suffix == ".pptx":
        return read_pptx(path)

    if suffix == ".xlsx":
        return read_xlsx(path)

    return None

# --------------------------------------------------
# Chunking
# --------------------------------------------------

def chunk_text(text):

    chunks = []

    start = 0

    while start < len(text):

        end = start + CHUNK_SIZE

        chunks.append(
            text[start:end]
        )

        start += (
            CHUNK_SIZE -
            CHUNK_OVERLAP
        )

    return chunks

# --------------------------------------------------
# Single File Ingestion
# --------------------------------------------------

def ingest_file(path):
    with index_write_lock():
        refresh_hash_cache()
        return _ingest_file_unlocked(path)


def _ingest_file_unlocked(path):

    path = Path(path).resolve()
    source = str(path)

    if not path.exists():
        return False

    if not is_supported_file(path):
        return False

    print(
        f"\nProcessing: "
        f"{path.name}"
    )

    try:
        current_hash = calculate_file_hash(path)
        old_chunk_ids = get_document_chunk_ids(path)
    except Exception as ex:
        record_index_status(source, "failed", error=ex)
        print(f"Failed preparing {path.name}: {ex}")
        return False

    cached_entry = hash_cache.get(source)
    cached_hash = (
        cached_entry.get("hash")
        if isinstance(cached_entry, dict)
        else cached_entry
    )
    cached_chunk_count = (
        cached_entry.get("chunk_count")
        if isinstance(cached_entry, dict)
        else None
    )

    if (
        cached_hash == current_hash
        and (old_chunk_ids or cached_chunk_count == 0)
    ):
        record_index_status(
            source,
            "ready",
            chunk_count=len(old_chunk_ids)
        )
        print(f"Skipping unchanged file: {path.name}")
        return True

    record_index_status(source, "indexing")

    try:
        text = extract_text(path)
    except Exception as ex:
        record_index_status(source, "failed", error=ex)
        print(f"Failed reading {path.name}: {ex}")
        return False

    if text is None:
        error = "No text could be extracted from this file."
        record_index_status(source, "failed", error=error)
        print(f"{error} {path.name}")
        return False

    chunks = chunk_text(text)
    file_hash = current_hash

    if not chunks:
        try:
            if old_chunk_ids:
                collection.delete(ids=old_chunk_ids)
            hash_cache[source] = {
                "hash": file_hash,
                "chunk_count": 0
            }
            save_hash_cache(hash_cache)
            record_index_status(source, "ready", chunk_count=0)
            print(f"Indexed {path.name} with no text chunks.")
            return True
        except Exception as ex:
            record_index_status(source, "failed", error=ex)
            print(f"Failed updating empty file {path.name}: {ex}")
            return False

    # Prepare the complete replacement before touching the current index.
    # If any embedding fails, the existing chunks remain searchable.
    try:
        embeddings = [get_embedding(chunk) for chunk in chunks]
        new_chunk_ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [
            {
                "source": source,
                "chunk": idx,
                "file_hash": file_hash
            }
            for idx in range(len(chunks))
        ]

        collection.add(
            ids=new_chunk_ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas
        )

        # Keep the old index until the replacement has been accepted.
        if old_chunk_ids:
            collection.delete(ids=old_chunk_ids)

        hash_cache[source] = {
            "hash": file_hash,
            "chunk_count": len(chunks)
        }
        save_hash_cache(hash_cache)
    except Exception as ex:
        record_index_status(source, "failed", error=ex)
        print(f"Failed indexing {path.name}: {ex}")
        return False

    record_index_status(source, "ready", chunk_count=len(chunks))
    print(f"Indexed {len(chunks)} chunks from {path.name}")
    return True

# --------------------------------------------------
# Full Scan
# --------------------------------------------------

def ingest_documents():
    with index_write_lock():
        refresh_hash_cache()
        document_root = Path(DOCUMENT_DIR).resolve()
        document_root.mkdir(parents=True, exist_ok=True)

        present_sources = set()

        for path in document_root.rglob("*"):
            if not path.is_file() or not is_supported_file(path):
                continue
            present_sources.add(str(path))
            _ingest_file_unlocked(path)

        _remove_missing_documents_unlocked(present_sources)

# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    ingest_documents()

    print()

    print(
        "Total document chunks:",
        collection.count()
    )
