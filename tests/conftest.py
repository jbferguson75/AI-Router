"""Test-only in-memory Chroma stand-in; tests never open the user's database."""

from __future__ import annotations

import math
import sys
import types

import pytest


class FakeCollection:
    def __init__(self, name):
        self.name = name
        self.records = {}

    def clear(self):
        self.records.clear()

    def count(self):
        return len(self.records)

    def add(self, ids, documents, embeddings, metadatas=None):
        metadatas = metadatas or [None] * len(ids)
        lengths = {len(ids), len(documents), len(embeddings), len(metadatas)}
        if len(lengths) != 1:
            raise ValueError("All Chroma add fields must have equal lengths")
        for record_id, document, embedding, metadata in zip(
            ids, documents, embeddings, metadatas
        ):
            if record_id in self.records:
                raise ValueError(f"Duplicate record id: {record_id}")
            self.records[record_id] = {
                "document": document,
                "embedding": list(embedding),
                "metadata": metadata,
            }

    def get(self, ids=None, where=None, include=None, limit=None, offset=0):
        rows = list(self.records.items())
        if ids is not None:
            wanted = set(ids)
            rows = [(key, value) for key, value in rows if key in wanted]
        if where:
            rows = [
                (key, value)
                for key, value in rows
                if all((value["metadata"] or {}).get(k) == v for k, v in where.items())
            ]
        rows = rows[offset: offset + limit if limit is not None else None]
        requested = set(include or ["documents", "metadatas"])
        result = {"ids": [key for key, _ in rows]}
        if "documents" in requested:
            result["documents"] = [value["document"] for _, value in rows]
        if "metadatas" in requested:
            result["metadatas"] = [value["metadata"] for _, value in rows]
        if "embeddings" in requested:
            result["embeddings"] = [value["embedding"] for _, value in rows]
        return result

    def query(
        self,
        query_embeddings,
        n_results=10,
        include=None,
        where=None,
        **_kwargs,
    ):
        requested = set(include or ["documents", "metadatas", "distances"])
        output = {"ids": [], "distances": []}
        if "documents" in requested:
            output["documents"] = []
        if "metadatas" in requested:
            output["metadatas"] = []
        if "embeddings" in requested:
            output["embeddings"] = []

        for query in query_embeddings:
            rows = list(self.records.items())
            if where:
                rows = [
                    (key, value)
                    for key, value in rows
                    if all(
                        (value["metadata"] or {}).get(k) == v
                        for k, v in where.items()
                    )
                ]
            rows.sort(
                key=lambda item: self._distance(query, item[1]["embedding"])
            )
            rows = rows[:n_results]
            output["ids"].append([key for key, _ in rows])
            output["distances"].append(
                [self._distance(query, value["embedding"]) for _, value in rows]
            )
            if "documents" in requested:
                output["documents"].append(
                    [value["document"] for _, value in rows]
                )
            if "metadatas" in requested:
                output["metadatas"].append(
                    [value["metadata"] for _, value in rows]
                )
            if "embeddings" in requested:
                output["embeddings"].append(
                    [value["embedding"] for _, value in rows]
                )
        return output

    @staticmethod
    def _distance(left, right):
        if len(left) != len(right):
            return math.inf
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))

    def delete(self, ids=None, where=None):
        if ids is not None:
            for record_id in ids:
                self.records.pop(record_id, None)
        elif where:
            matching = [
                record_id
                for record_id, value in self.records.items()
                if all((value["metadata"] or {}).get(k) == v for k, v in where.items())
            ]
            for record_id in matching:
                self.records.pop(record_id, None)


class FakePersistentClient:
    clients = {}

    def __new__(cls, path, *args, **kwargs):
        key = str(path)
        if key not in cls.clients:
            instance = super().__new__(cls)
            instance.collections = {}
            cls.clients[key] = instance
        return cls.clients[key]

    def get_or_create_collection(self, name, **_kwargs):
        if name not in self.collections:
            self.collections[name] = FakeCollection(name)
        return self.collections[name]

    def get_collection(self, name, **_kwargs):
        return self.collections[name]


fake_chromadb = types.ModuleType("chromadb")
fake_chromadb.PersistentClient = FakePersistentClient
sys.modules["chromadb"] = fake_chromadb


@pytest.fixture(autouse=True)
def isolate_local_state(tmp_path, monkeypatch):
    for client in FakePersistentClient.clients.values():
        for collection in client.collections.values():
            collection.clear()

    # Import only after substituting chromadb so imports never open ./chroma_db.
    import index_status
    import ingest

    monkeypatch.setattr(index_status, "STATUS_FILE", tmp_path / "index-status.json")
    monkeypatch.setattr(ingest, "HASH_CACHE_FILE", tmp_path / "hash-cache.json")
    monkeypatch.setattr(ingest, "INDEX_LOCK_FILE", tmp_path / "index-write.lock")
    ingest.hash_cache.clear()
    yield


@pytest.fixture
def document_collection():
    return FakePersistentClient("./chroma_db").get_or_create_collection("documents")


@pytest.fixture
def memory_collection():
    return FakePersistentClient("./chroma_db").get_or_create_collection("memories")
