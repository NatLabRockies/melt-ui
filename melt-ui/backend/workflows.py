from __future__ import annotations

import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()


BASE_DIR = Path(__file__).resolve().parent.parent
USER_WORKFLOWS_DIR = BASE_DIR / "frontend" / "workflows" / "user_workflow"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _ensure_dir() -> None:
    USER_WORKFLOWS_DIR.mkdir(parents=True, exist_ok=True)


def _slugify(name: str) -> str:
    base = (name or "").strip().lower()
    base = re.sub(r"[^a-z0-9_-]+", "-", base)
    base = re.sub(r"-+", "-", base).strip("-")
    return base or "workflow"


def _safe_id(value: str) -> str:
    cleaned = _slugify(value)
    if not cleaned:
        raise ValueError("Invalid workflow id")
    return cleaned


def _workflow_path(workflow_id: str) -> Path:
    safe = _safe_id(workflow_id)
    return (USER_WORKFLOWS_DIR / f"{safe}.json").resolve()


def _is_within_user_dir(path: Path) -> bool:
    try:
        return path.is_relative_to(USER_WORKFLOWS_DIR.resolve())
    except AttributeError:
        # Python <3.9 fallback (not expected, but safe)
        return str(path).startswith(str(USER_WORKFLOWS_DIR.resolve()))


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


@router.get("/workflows/user")
async def list_user_workflows() -> JSONResponse:
    _ensure_dir()

    items: list[dict[str, Any]] = []
    for path in sorted(USER_WORKFLOWS_DIR.glob("*.json")):
        try:
            data = _read_json(path)
        except Exception:
            continue

        stat = path.stat()
        workflow_id = data.get("id") or path.stem
        created_at = data.get("createdAt")
        updated_at = (
            data.get("updatedAt")
            or datetime.fromtimestamp(stat.st_mtime, UTC).isoformat()
        )

        items.append(
            {
                "id": str(workflow_id),
                "source": data.get("source", "user"),
                "name": data.get("name") or path.stem,
                "description": data.get("description", ""),
                "tags": data.get("tags", []),
                "createdAt": created_at,
                "updatedAt": updated_at,
                "file": path.name,
            }
        )

    items.sort(key=lambda x: x.get("updatedAt") or "", reverse=True)
    return JSONResponse(
        {
            "items": items,
            "directory": str(USER_WORKFLOWS_DIR),
        }
    )


@router.get("/workflows/user/{workflow_id}")
async def get_user_workflow(workflow_id: str) -> JSONResponse:
    _ensure_dir()

    try:
        path = _workflow_path(workflow_id)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    if not _is_within_user_dir(path):
        return JSONResponse(status_code=400, content={"error": "Invalid workflow path"})

    if not path.exists():
        return JSONResponse(status_code=404, content={"error": "Workflow not found"})

    try:
        payload = _read_json(path)
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"error": f"Failed to read workflow: {e}"}
        )

    return JSONResponse(payload)


@router.post("/workflows/user/save")
async def save_user_workflow(request: Request) -> JSONResponse:
    _ensure_dir()

    body = await request.json()
    workflow_id = body.get("id")
    name = (body.get("name") or "").strip()
    description = body.get("description") or ""
    tags = body.get("tags") or []
    graph = body.get("graph")
    view = body.get("view") or {}
    overwrite = bool(body.get("overwrite", False))

    if graph is None:
        return JSONResponse(
            status_code=400, content={"error": "Missing workflow graph payload"}
        )

    if not workflow_id:
        workflow_id = (
            _slugify(name) if name else f"workflow-{int(datetime.now().timestamp())}"
        )

    try:
        workflow_id = _safe_id(str(workflow_id))
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    path = _workflow_path(workflow_id)
    if not _is_within_user_dir(path):
        return JSONResponse(status_code=400, content={"error": "Invalid workflow path"})

    existing: dict[str, Any] = {}
    if path.exists():
        if not overwrite:
            return JSONResponse(
                status_code=409, content={"error": "Workflow already exists"}
            )
        try:
            existing = _read_json(path)
        except Exception:
            existing = {}

    created_at = existing.get("createdAt") or _now_iso()
    updated_at = _now_iso()

    record: dict[str, Any] = {
        "id": workflow_id,
        "source": "user",
        "name": name or existing.get("name") or workflow_id,
        "description": description,
        "tags": tags if isinstance(tags, list) else [],
        "createdAt": created_at,
        "updatedAt": updated_at,
        "graph": graph,
        "view": view if isinstance(view, dict) else {},
    }

    try:
        _write_json(path, record)
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"error": f"Failed to write workflow: {e}"}
        )

    return JSONResponse(
        {
            "ok": True,
            "id": workflow_id,
            "file": path.name,
            "path": str(path),
            "record": {
                "id": workflow_id,
                "source": "user",
                "name": record["name"],
                "description": record["description"],
                "tags": record["tags"],
                "createdAt": record["createdAt"],
                "updatedAt": record["updatedAt"],
            },
        }
    )


@router.post("/workflows/user/{workflow_id}/rename")
async def rename_user_workflow(workflow_id: str, request: Request) -> JSONResponse:
    _ensure_dir()

    body = await request.json()
    new_name = (body.get("name") or "").strip()
    description = body.get("description")

    if not new_name:
        return JSONResponse(
            status_code=400, content={"error": "Missing new workflow name"}
        )

    path = _workflow_path(workflow_id)
    if not path.exists():
        return JSONResponse(status_code=404, content={"error": "Workflow not found"})

    try:
        payload = _read_json(path)
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"error": f"Failed to read workflow: {e}"}
        )

    payload["name"] = new_name
    if description is not None:
        payload["description"] = description
    payload["updatedAt"] = _now_iso()

    try:
        _write_json(path, payload)
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"error": f"Failed to rename workflow: {e}"}
        )

    return JSONResponse({"ok": True, "id": workflow_id})


@router.delete("/workflows/user/{workflow_id}")
async def delete_user_workflow(workflow_id: str) -> JSONResponse:
    _ensure_dir()

    path = _workflow_path(workflow_id)
    if not path.exists():
        return JSONResponse(status_code=404, content={"error": "Workflow not found"})

    try:
        os.remove(path)
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"error": f"Failed to delete workflow: {e}"}
        )

    return JSONResponse({"ok": True, "id": workflow_id})
