"""Shared pytest fixtures for the MoLeJEPA test suite."""

from __future__ import annotations

from typing import Any

import pytest

import mole_jepa.registry as _reg_mod

# ---------------------------------------------------------------------------
# In-memory Supabase fake
# ---------------------------------------------------------------------------


class _FakeResult:
    def __init__(self, data: list[dict]) -> None:
        self.data = data


class _FakeQuery:
    """Fluent query builder backed by a shared in-memory dict."""

    def __init__(self, store: dict[str, Any]) -> None:
        self._store = store
        self._op = "select"
        self._filters: dict[str, Any] = {}
        self._cols = "*"
        self._order_col: str | None = None
        self._upsert_row: dict[str, Any] | None = None

    def select(self, cols: str = "*") -> _FakeQuery:
        self._cols = cols
        return self

    def eq(self, col: str, val: Any) -> _FakeQuery:
        self._filters[col] = val
        return self

    def order(self, col: str) -> _FakeQuery:
        self._order_col = col
        return self

    def upsert(self, row: dict, on_conflict: str | None = None) -> _FakeQuery:
        self._op = "upsert"
        self._upsert_row = row
        return self

    def delete(self) -> _FakeQuery:
        self._op = "delete"
        return self

    def execute(self) -> _FakeResult:
        if self._op == "upsert":
            assert self._upsert_row is not None
            self._store[self._upsert_row["name"]] = dict(self._upsert_row)
            return _FakeResult([self._upsert_row])

        if self._op == "delete":
            to_del = [
                k
                for k, r in self._store.items()
                if all(r.get(c) == v for c, v in self._filters.items())
            ]
            for k in to_del:
                del self._store[k]
            return _FakeResult([])

        # select
        rows = [dict(r) for r in self._store.values()]
        for col, val in self._filters.items():
            rows = [r for r in rows if r.get(col) == val]
        if self._order_col:
            rows.sort(key=lambda r: r[self._order_col])
        if self._cols != "*":
            wanted = [c.strip() for c in self._cols.split(",")]
            rows = [{k: r[k] for k in wanted if k in r} for r in rows]
        return _FakeResult(rows)


class FakeSupabaseClient:
    """Minimal in-memory Supabase client for unit tests."""

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}

    def table(self, _name: str) -> _FakeQuery:
        return _FakeQuery(self._store)


@pytest.fixture(autouse=True)
def fake_supabase(monkeypatch: pytest.MonkeyPatch) -> FakeSupabaseClient:
    """Replace the module-level Supabase client with a fresh in-memory fake.

    ``autouse=True`` means every test gets a clean, isolated registry
    without needing to request this fixture explicitly.
    """
    client = FakeSupabaseClient()
    monkeypatch.setattr(_reg_mod, "_sb_client", client)
    return client
