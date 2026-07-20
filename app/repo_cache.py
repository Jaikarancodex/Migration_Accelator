"""Streamlit-cached wrappers around MigrationRepo disk reads.

Streamlit reruns the entire script top-to-bottom on every interaction, and
`st.tabs()` does not skip inactive tabs' code — so a single click can trigger
the same workflow's ir.json being parsed several times over (once per tab
that has an independent object selector). Every wrapper here is keyed on the
relevant file/directory's mtime, so a cache entry is used exactly as long as
the underlying file is unchanged — edits made through the app (which all go
through MigrationRepo's own write methods) invalidate it automatically,
never serving stale data.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from ingest.alteryx.ir import Workflow
from repo.metadata import ObjectMetadata
from repo.store import MigrationRepo

_MAX_ENTRIES = 128


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return -1.0


@st.cache_data(show_spinner=False, max_entries=_MAX_ENTRIES)
def _read_workflow(root: str, name: str, mtime_key: float) -> Workflow:
    return MigrationRepo(root).read_workflow(name)


def read_workflow(repo: MigrationRepo, name: str) -> Workflow:
    path = repo.root / name / "ir.json"
    return _read_workflow(str(repo.root), name, _mtime(path))


@st.cache_data(show_spinner=False, max_entries=_MAX_ENTRIES)
def _read_metadata(root: str, name: str, mtime_key: float) -> ObjectMetadata:
    return MigrationRepo(root).read_metadata(name)


def read_metadata(repo: MigrationRepo, name: str) -> ObjectMetadata:
    path = repo.root / name / "metadata.json"
    return _read_metadata(str(repo.root), name, _mtime(path))


@st.cache_data(show_spinner=False, max_entries=_MAX_ENTRIES)
def _list_object_names(root: str, mtime_key: float) -> list[str]:
    return MigrationRepo(root).list_object_names()


def list_object_names(repo: MigrationRepo) -> list[str]:
    return _list_object_names(str(repo.root), _mtime(repo.root))


def list_metadata(repo: MigrationRepo) -> list[ObjectMetadata]:
    return [read_metadata(repo, name) for name in list_object_names(repo)]


@st.cache_data(show_spinner=False, max_entries=_MAX_ENTRIES)
def _list_macro_names(root: str, mtime_key: float) -> list[str]:
    return MigrationRepo(root).list_macro_names()


def list_macro_names(repo: MigrationRepo) -> list[str]:
    return _list_macro_names(str(repo.root), _mtime(repo.macro_dir))


def all_macros(repo: MigrationRepo) -> dict[str, Workflow]:
    return {name: read_workflow_macro(repo, name) for name in list_macro_names(repo)}


@st.cache_data(show_spinner=False, max_entries=_MAX_ENTRIES)
def _read_macro(root: str, key: str, mtime_key: float) -> Workflow:
    return MigrationRepo(root).read_macro(key)


def read_workflow_macro(repo: MigrationRepo, key: str) -> Workflow:
    path = repo.macro_dir / f"{key.lower()}.json"
    return _read_macro(str(repo.root), key, _mtime(path))
