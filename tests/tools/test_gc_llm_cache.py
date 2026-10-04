"""Regression tests for the orphaned LLM cache GC.

The sweep deletes data, so the tests pin the two halves that decide what may
go: the candidate filter (a row must name a chunk) and the liveness check (the
chunk must be gone). A row whose chunk still exists must survive even when no
chunk lists it — deleting it would only re-bill the next reprocess — and a
deletion that reports success while the rows are still present must raise.
"""

import types

import pytest

from lightrag.tools.gc_llm_cache import gc_orphan_llm_cache

pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def _stub_key_enumeration(monkeypatch):
    """Route the sweep's fake KVs through the shared enumeration seam.

    The real per-backend scans belong to the rebuild tool's tests; this tool
    only has to consume the seam, which
    ``test_key_enumeration_delegates_to_the_shared_backend_scan`` pins.
    """
    import lightrag.tools.rebuild_vdb as rebuild_vdb

    async def fake_enumerate(kv):
        return list(kv.rows)

    monkeypatch.setattr(rebuild_vdb, "enumerate_kv_keys", fake_enumerate)


class FakeCacheKV:
    """Cache namespace with just enough surface for the sweep."""

    def __init__(self, rows, *, delete_removes=True):
        self.rows = dict(rows)
        self.delete_removes = delete_removes
        self.deleted: list[str] = []
        self.commits = 0

    async def get_by_ids(self, ids):
        return [self.rows.get(key) for key in ids]

    async def filter_keys(self, keys):
        return {key for key in keys if key not in self.rows}

    async def delete(self, ids):
        self.deleted.extend(ids)
        if self.delete_removes:
            for key in ids:
                self.rows.pop(key, None)

    async def index_done_callback(self):
        self.commits += 1


class FakeChunksKV:
    """Text chunks namespace reduced to the liveness question."""

    def __init__(self, chunk_ids):
        self.chunk_ids = set(chunk_ids)
        self.filter_calls: list[set[str]] = []

    async def filter_keys(self, keys):
        self.filter_calls.append(set(keys))
        return {key for key in keys if key not in self.chunk_ids}


def _rag(cache, chunks):
    return types.SimpleNamespace(llm_response_cache=cache, text_chunks=chunks)


def _extract_row(chunk_id):
    return {
        "return": "entities...",
        "cache_type": "extract",
        "chunk_id": chunk_id,
        "original_prompt": "chunk text",
        "queryparam": None,
    }


def _query_row():
    return {
        "return": "answer",
        "cache_type": "query",
        "chunk_id": None,
        "original_prompt": "question",
        "queryparam": {},
    }


async def test_orphan_is_reported_and_a_dry_run_deletes_nothing():
    cache = FakeCacheKV(
        {
            "default:extract:aaa": _extract_row("chunk-gone"),
            "default:extract:bbb": _extract_row("chunk-alive"),
        }
    )
    chunks = FakeChunksKV({"chunk-alive"})

    report = await gc_orphan_llm_cache(_rag(cache, chunks))

    assert report["cache_rows"] == 2
    assert report["chunk_scoped_rows"] == 2
    assert report["orphan_rows"] == 1
    assert report["live_rows"] == 1
    assert report["deleted_rows"] == 0
    assert report["orphan_keys_sample"] == ["default:extract:aaa"]
    assert cache.deleted == []


async def test_apply_deletes_only_the_orphans_and_commits():
    cache = FakeCacheKV(
        {
            "default:extract:aaa": _extract_row("chunk-gone"),
            "default:extract:bbb": _extract_row("chunk-alive"),
            "default:query:ccc": _query_row(),
        }
    )
    chunks = FakeChunksKV({"chunk-alive"})

    report = await gc_orphan_llm_cache(_rag(cache, chunks), apply=True)

    assert report["deleted_rows"] == 1
    assert cache.deleted == ["default:extract:aaa"]
    assert "default:extract:aaa" not in cache.rows
    assert set(cache.rows) == {"default:extract:bbb", "default:query:ccc"}
    # The delete is only durable once the storage commits.
    assert cache.commits == 1


async def test_query_rows_are_never_candidates():
    """Query/keywords rows carry no chunk_id; another tool owns them."""
    cache = FakeCacheKV(
        {"local:query:aaa": _query_row(), "local:keywords:bbb": _query_row()}
    )
    chunks = FakeChunksKV(set())

    report = await gc_orphan_llm_cache(_rag(cache, chunks), apply=True)

    assert report["unscoped_rows"] == 2
    assert report["orphan_rows"] == 0
    assert cache.deleted == []
    # Nothing chunk-scoped was seen, so liveness was never asked.
    assert chunks.filter_calls == []


async def test_row_with_live_chunk_survives_even_when_no_chunk_lists_it():
    """The row's chunk exists, so its document is alive; leave the row alone."""
    cache = FakeCacheKV({"default:extract:aaa": _extract_row("chunk-alive")})
    chunks = FakeChunksKV({"chunk-alive"})  # no llm_cache_list anywhere

    report = await gc_orphan_llm_cache(_rag(cache, chunks), apply=True)

    assert report["orphan_rows"] == 0
    assert cache.deleted == []
    assert "default:extract:aaa" in cache.rows


async def test_empty_or_non_string_chunk_id_is_not_a_candidate():
    cache = FakeCacheKV(
        {
            "default:extract:aaa": _extract_row(""),
            "default:extract:bbb": _extract_row(None),
        }
    )
    chunks = FakeChunksKV(set())

    report = await gc_orphan_llm_cache(_rag(cache, chunks), apply=True)

    assert report["unscoped_rows"] == 2
    assert cache.deleted == []


async def test_a_delete_that_did_not_stick_raises():
    cache = FakeCacheKV(
        {"default:extract:aaa": _extract_row("chunk-gone")}, delete_removes=False
    )
    chunks = FakeChunksKV(set())

    with pytest.raises(RuntimeError, match="still present"):
        await gc_orphan_llm_cache(_rag(cache, chunks), apply=True)


async def test_batching_preserves_every_orphan():
    rows = {f"default:extract:{i}": _extract_row(f"chunk-{i}") for i in range(7)}
    rows["default:extract:live"] = _extract_row("chunk-live")
    cache = FakeCacheKV(rows)
    chunks = FakeChunksKV({"chunk-live"})

    report = await gc_orphan_llm_cache(_rag(cache, chunks), apply=True, batch_size=2)

    assert report["orphan_rows"] == 7
    assert report["deleted_rows"] == 7
    assert set(cache.deleted) == {f"default:extract:{i}" for i in range(7)}
    assert chunks.filter_calls and all(len(call) <= 2 for call in chunks.filter_calls)


async def test_missing_storages_raise():
    chunks = FakeChunksKV(set())
    with pytest.raises(ValueError, match="cache storage"):
        await gc_orphan_llm_cache(
            types.SimpleNamespace(llm_response_cache=None, text_chunks=chunks)
        )

    cache = FakeCacheKV({})
    with pytest.raises(ValueError, match="text chunks storage"):
        await gc_orphan_llm_cache(
            types.SimpleNamespace(llm_response_cache=cache, text_chunks=None)
        )


async def test_non_positive_batch_size_raises():
    cache = FakeCacheKV({})
    chunks = FakeChunksKV(set())
    with pytest.raises(ValueError, match="batch_size"):
        await gc_orphan_llm_cache(_rag(cache, chunks), batch_size=0)


async def test_key_enumeration_delegates_to_the_shared_backend_scan(monkeypatch):
    """The sweep must not grow its own copy of the per-backend scans."""
    calls: list[object] = []

    async def fake_enumerate(kv):
        calls.append(kv)
        return ["default:extract:aaa"]

    import lightrag.tools.rebuild_vdb as rebuild_vdb

    monkeypatch.setattr(rebuild_vdb, "enumerate_kv_keys", fake_enumerate)

    cache = FakeCacheKV({"default:extract:aaa": _extract_row("chunk-gone")})
    chunks = FakeChunksKV(set())

    report = await gc_orphan_llm_cache(_rag(cache, chunks))

    assert calls == [cache]
    assert report["orphan_rows"] == 1
