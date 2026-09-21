from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

_PATH_GRANT_TTL_SECONDS = 4 * 60 * 60
_PATH_GRANT_MAX_ITEMS = 256


def _grant_store(app) -> dict[str, dict[str, Any]]:
    store = getattr(app.state, "path_grants", None)
    if store is None:
        store = {}
        app.state.path_grants = store
    return store


def _cleanup_grants(store: dict[str, dict[str, Any]]) -> None:
    now = time.monotonic()

    expired = [
        grant_id
        for grant_id, entry in store.items()
        if float(entry.get("expires_at", 0.0)) <= now
    ]
    for grant_id in expired:
        store.pop(grant_id, None)

    if len(store) <= _PATH_GRANT_MAX_ITEMS:
        return

    ordered = sorted(
        store.items(),
        key=lambda item: float(item[1].get("expires_at", 0.0)),
    )
    for grant_id, _entry in ordered[: len(store) - _PATH_GRANT_MAX_ITEMS]:
        store.pop(grant_id, None)


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return candidate.resolve()


def path_is_within(path: Path, root: Path) -> bool:
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    try:
        resolved_path.relative_to(resolved_root)
        return True
    except ValueError:
        return False


def grant_file_access(app, path: str | Path) -> str:
    resolved = _resolve_path(path)
    if not resolved.exists() or not resolved.is_file():
        raise ValueError(f"Selected path is not a file: {resolved}")

    store = _grant_store(app)
    _cleanup_grants(store)

    grant_id = uuid.uuid4().hex
    store[grant_id] = {
        "kind": "file",
        "path": str(resolved),
        "expires_at": time.monotonic() + _PATH_GRANT_TTL_SECONDS,
    }
    return grant_id


def require_file_grant(app, path: str | Path, grant_id: Any) -> Path:
    resolved = _resolve_path(path)
    token = str(grant_id or "").strip()

    if not token:
        raise PermissionError(
            "External file access requires selecting the file with Browse."
        )

    store = _grant_store(app)
    _cleanup_grants(store)

    entry = store.get(token)
    if entry is None or entry.get("kind") != "file":
        raise PermissionError(
            "The file access grant is missing or expired. Select the file again."
        )

    if entry.get("path") != str(resolved):
        raise PermissionError(
            "The file access grant does not match the requested path."
        )

    return resolved


def resolve_read_file(
    app,
    path: str | Path,
    grant_id: Any = None,
    *,
    managed_roots: tuple[Path, ...] = (),
) -> Path:
    resolved = _resolve_path(path)

    for root in managed_roots:
        if path_is_within(resolved, root):
            return resolved

    return require_file_grant(app, resolved, grant_id)
