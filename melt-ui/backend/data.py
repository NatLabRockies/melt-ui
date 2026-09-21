import json
import time
from pathlib import Path
from typing import Optional
from uuid import uuid4

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from ptmelt.utils.preprocessing import get_normalizers
from sklearn.datasets import make_blobs, make_regression

router = APIRouter()


APP_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = APP_ROOT / "datasets"
DATASET_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR = DATASET_DIR
_SUPPORTED_DATA_EXTENSIONS = {".xlsx", ".csv"}

# naive in-memory index... good enough for v1...
_UPLOAD_INDEX: dict[str, dict] = {}
_UPLOAD_TTL_SEC = 60 * 60  # 1 hour


def _dataset_meta_path(dataset_id: str) -> Path:
    return DATASET_DIR / f"{dataset_id}.meta.json"


def _write_dataset_meta(meta: dict):
    meta_path = _dataset_meta_path(meta["dataset_id"])
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _read_dataset_meta(dataset_id: str) -> dict | None:
    meta_path = _dataset_meta_path(dataset_id)
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _index_dataset(meta: dict):
    _UPLOAD_INDEX[meta["dataset_id"]] = {
        "path": str(meta["path"]),
        "created": float(meta.get("created", time.time())),
        "dataset_id": meta["dataset_id"],
        "file_type": meta.get("file_type", "xlsx"),
    }


def _resolve_dataset(dataset_id_or_file_id: str | None) -> tuple[Path, dict]:
    if not dataset_id_or_file_id:
        raise HTTPException(
            status_code=400,
            detail="Missing dataset_id/file_id. Please upload a dataset.",
        )

    idx_meta = _UPLOAD_INDEX.get(dataset_id_or_file_id)
    if idx_meta:
        path = Path(idx_meta["path"])
        if path.exists():
            meta = _read_dataset_meta(dataset_id_or_file_id) or {
                "dataset_id": dataset_id_or_file_id,
                "path": str(path),
                "file_type": idx_meta.get("file_type", "xlsx"),
                "created": idx_meta.get("created", time.time()),
            }
            return path, meta

    meta = _read_dataset_meta(dataset_id_or_file_id)
    if not meta:
        raise HTTPException(
            status_code=400,
            detail="Missing/expired dataset_id/file_id. Please re-upload the dataset.",
        )

    path = Path(meta.get("path", ""))
    if not path.exists():
        raise HTTPException(
            status_code=400,
            detail="Uploaded dataset not found on server. Please re-upload.",
        )

    _index_dataset(meta)
    return path, meta


def _read_preview(
    path: Path, file_type: str, sheet_name: str | None, preview_rows: int
):
    file_type = (file_type or "xlsx").lower()

    if file_type == "xlsx":
        xls = pd.ExcelFile(path, engine="openpyxl")
        sheet_names = list(xls.sheet_names)
        chosen_sheet = sheet_name if (sheet_name in sheet_names) else sheet_names[0]
        df_preview = pd.read_excel(
            path,
            sheet_name=chosen_sheet,
            nrows=int(preview_rows),
            engine="openpyxl",
        )
        return df_preview, sheet_names, chosen_sheet

    if file_type == "csv":
        df_preview = pd.read_csv(path, nrows=int(preview_rows))
        return df_preview, [], ""

    raise HTTPException(status_code=400, detail=f"Unsupported file_type: {file_type}")


def _parse_json_if_needed(value):
    if isinstance(value, str):
        return json.loads(value)
    return value


def _split_index_from_name(split: str) -> int:
    split_l = str(split or "train").lower()
    if split_l == "train":
        return 0
    if split_l == "val":
        return 1
    if split_l == "test":
        return 2
    raise ValueError("split must be one of: train, val, test")


def _select_split_if_needed(value, split: str):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return value
    if not all(isinstance(item, (list, tuple)) for item in value):
        return value
    idx = _split_index_from_name(split)
    return value[idx]


def _coerce_2d(value, name: str, split: str):
    if value is None:
        return None

    parsed = _parse_json_if_needed(value)
    selected = _select_split_if_needed(parsed, split)
    arr = np.asarray(selected)

    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 1D or 2D after split selection.")
    return arr


def _coerce_1d(value, name: str, split: str):
    if value is None:
        return None

    parsed = _parse_json_if_needed(value)
    selected = _select_split_if_needed(parsed, split)
    arr = np.asarray(selected).reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D after split selection.")
    return arr


def _default_columns(prefix: str, width: int):
    return [f"{prefix}{i}" for i in range(width)]


def _coerce_columns(cols, width: int, prefix: str):
    if cols is None:
        return _default_columns(prefix, width)
    if isinstance(cols, str):
        parsed = _parse_json_if_needed(cols)
        cols = parsed
    if not isinstance(cols, list):
        raise ValueError(f"{prefix}_cols must be a list when provided.")
    if len(cols) != width:
        raise ValueError(
            f"{prefix}_cols length ({len(cols)}) does not match width ({width})."
        )
    return [str(c) for c in cols]


def _avoid_overwrite(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    i = 1
    while True:
        cand = parent / f"{stem}_{i}{suffix}"
        if not cand.exists():
            return cand
        i += 1


@router.post("/save_tabular_data")
async def save_tabular_data(request: Request):
    body = await request.json()

    fmt = str(body.get("format", "xlsx")).lower().strip()
    if fmt not in {"xlsx", "csv"}:
        return JSONResponse(
            status_code=400,
            content={"error": "format must be one of: xlsx, csv"},
        )

    split = str(body.get("split", "train"))
    sheet_name = str(body.get("sheet_name", "data"))[:31] or "data"
    include_index = bool(body.get("include_index", False))
    overwrite = bool(body.get("overwrite", False))

    try:
        x = _coerce_2d(body.get("x"), "x", split=split)
        y = _coerce_2d(body.get("y"), "y", split=split)
        labels = _coerce_1d(body.get("labels"), "labels", split=split)
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    if x is None and y is None and labels is None:
        return JSONResponse(
            status_code=400,
            content={"error": "At least one of x, y, labels must be provided."},
        )

    n_rows = None
    for arr in (x, y, labels):
        if arr is None:
            continue
        if n_rows is None:
            n_rows = int(arr.shape[0])
        elif int(arr.shape[0]) != n_rows:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "x, y, and labels must have matching row counts after split selection."
                },
            )

    frames = []
    meta = {
        "split": split,
        "included": [],
    }

    try:
        if x is not None:
            x_cols = _coerce_columns(body.get("x_cols"), x.shape[1], "x")
            frames.append(pd.DataFrame(x, columns=x_cols))
            meta["included"].append("x")

        if y is not None:
            y_cols = _coerce_columns(body.get("y_cols"), y.shape[1], "y")
            frames.append(pd.DataFrame(y, columns=y_cols))
            meta["included"].append("y")

        if labels is not None:
            label_col = str(body.get("label_col", "label"))
            frames.append(pd.DataFrame({label_col: labels}))
            meta["included"].append("labels")

        df = pd.concat(frames, axis=1)
        dupes = df.columns[df.columns.duplicated()].tolist()
        if dupes:
            return JSONResponse(
                status_code=400,
                content={
                    "error": f"Duplicate output column names detected: {dupes}. Provide explicit x_cols/y_cols/label_col."
                },
            )
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    save_dir_raw = str(body.get("save_dir", "saved_data")).strip() or "saved_data"
    save_dir = Path(save_dir_raw)
    if not save_dir.is_absolute():
        save_dir = Path.cwd() / save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    filename_raw = str(body.get("filename", "")).strip()
    suffix = f".{fmt}"
    if filename_raw:
        filename = filename_raw
        if not filename.lower().endswith(suffix):
            filename = f"{filename}{suffix}"
    else:
        ts = int(time.time())
        filename = f"tabular_data_{ts}_{uuid4().hex[:8]}{suffix}"

    out_path = save_dir / filename
    if not overwrite:
        out_path = _avoid_overwrite(out_path)

    try:
        if fmt == "csv":
            df.to_csv(out_path, index=include_index)
        else:
            with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=include_index)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to save output file: {e}"},
        )

    return JSONResponse(
        content={
            "path": str(out_path),
            "format": fmt,
            "shape": {
                "rows": int(df.shape[0]),
                "cols": int(df.shape[1]),
            },
            "metadata": {
                **meta,
                "columns": [str(c) for c in df.columns.tolist()],
                "sheet_name": sheet_name if fmt == "xlsx" else None,
            },
        }
    )


def _cleanup_uploads():
    now = time.time()
    expired = [
        fid
        for fid, meta in _UPLOAD_INDEX.items()
        if now - meta["created"] > _UPLOAD_TTL_SEC
    ]
    for fid in expired:
        try:
            path = Path(_UPLOAD_INDEX[fid]["path"])
            if path.parent != DATASET_DIR:
                path.unlink(missing_ok=True)
        except Exception:
            pass
        _UPLOAD_INDEX.pop(fid, None)


@router.get("/dataset_library")
async def dataset_library():
    datasets = []
    for meta_path in sorted(DATASET_DIR.glob("*.meta.json"), reverse=True):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            datasets.append(
                {
                    "dataset_id": meta.get("dataset_id"),
                    "file_name": meta.get("file_name"),
                    "file_type": meta.get("file_type"),
                    "created": meta.get("created"),
                }
            )
        except Exception:
            continue

    return {"datasets": datasets}


def _validate_headers(columns):
    # Robust header sanitization for real-world CSV/XLSX files:
    # - blank/Unnamed headers become auto names
    # - duplicate headers are disambiguated with numeric suffixes
    cleaned = []
    counts: dict[str, int] = {}

    for i, c in enumerate(columns):
        s = "" if c is None else str(c).strip()
        if not s or s.lower().startswith("unnamed"):
            s = f"col_{i}"

        n = counts.get(s, 0)
        if n == 0:
            name = s
        else:
            name = f"{s}__{n + 1}"
        counts[s] = n + 1
        cleaned.append(name)

    return cleaned


@router.post("/excel_inspect")
async def excel_inspect(
    file: UploadFile = File(...),
    sheet_name: str | None = Form(None),
    preview_rows: int = Form(50),
):
    _cleanup_uploads()

    raw_name = file.filename or "dataset"
    suffix = Path(raw_name).suffix.lower()
    if suffix not in _SUPPORTED_DATA_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only .xlsx and .csv files are supported in this node.",
        )

    dataset_id = str(uuid4())
    path = DATASET_DIR / f"{dataset_id}{suffix}"
    file_type = suffix[1:]

    # save upload
    with path.open("wb") as f:
        f.write(await file.read())

    created = time.time()
    meta = {
        "dataset_id": dataset_id,
        "file_name": raw_name,
        "file_type": file_type,
        "path": str(path),
        "created": created,
    }
    _write_dataset_meta(meta)
    _index_dataset(meta)

    try:
        df_preview, sheet_names, chosen_sheet = _read_preview(
            path,
            file_type=file_type,
            sheet_name=sheet_name,
            preview_rows=preview_rows,
        )

        cols = _validate_headers(df_preview.columns)
        df_preview.columns = cols  # apply cleaned names

        dtypes = {c: str(t) for c, t in df_preview.dtypes.items()}

        return {
            "file_id": dataset_id,
            "dataset_id": dataset_id,
            "file_name": file.filename,
            "file_type": file_type,
            "sheet_names": sheet_names,
            "sheet_name": chosen_sheet,
            "columns": cols,
            "dtypes": dtypes,
            "preview_rows": df_preview.to_dict(orient="records"),
        }
    except HTTPException:
        # re-raise header validation errors
        raise
    except Exception as e:
        # cleanup bad upload
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            _dataset_meta_path(dataset_id).unlink(missing_ok=True)
        except Exception:
            pass
        _UPLOAD_INDEX.pop(dataset_id, None)
        raise HTTPException(status_code=400, detail=f"Failed to read dataset file: {e}")


@router.post("/excel_inspect_by_id")
async def excel_inspect_by_id(request: Request):
    body = await request.json()
    _cleanup_uploads()

    dataset_id = body.get("dataset_id") or body.get("file_id")
    sheet_name = body.get("sheet_name")
    preview_rows = int(body.get("preview_rows", 50))

    path, meta = _resolve_dataset(dataset_id)
    file_type = str(meta.get("file_type", "xlsx")).lower()

    try:
        df_preview, sheet_names, chosen_sheet = _read_preview(
            path,
            file_type=file_type,
            sheet_name=sheet_name,
            preview_rows=preview_rows,
        )

        cols = _validate_headers(df_preview.columns)
        df_preview.columns = cols
        dtypes = {c: str(t) for c, t in df_preview.dtypes.items()}

        return {
            "file_id": dataset_id,
            "dataset_id": dataset_id,
            "file_type": file_type,
            "sheet_names": sheet_names,
            "sheet_name": chosen_sheet,
            "columns": cols,
            "dtypes": dtypes,
            "preview_rows": df_preview.to_dict(orient="records"),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to inspect sheet: {e}")


@router.post("/excel_build_xy")
async def excel_build_xy(request: Request):
    body = await request.json()
    _cleanup_uploads()

    dataset_id = body.get("dataset_id") or body.get("file_id")
    sheet_name = body.get("sheet_name")
    x_cols = body.get("x_cols") or []
    y_cols = body.get("y_cols") or []
    na_strategy = str(body.get("na_strategy", "drop")).lower()

    path, meta = _resolve_dataset(dataset_id)
    file_type = str(meta.get("file_type", "xlsx")).lower()

    if not x_cols:
        raise HTTPException(status_code=400, detail="No feature columns selected.")
    if not y_cols:
        raise HTTPException(status_code=400, detail="No target columns selected.")

    if not isinstance(x_cols, list) or not isinstance(y_cols, list):
        raise HTTPException(
            status_code=400, detail="x_cols and y_cols must be lists of column names."
        )

    if set(x_cols) & set(y_cols):
        raise HTTPException(
            status_code=400, detail="Inputs (X) and targets (y) cannot overlap."
        )

    try:
        if file_type == "xlsx":
            df = pd.read_excel(path, sheet_name=sheet_name or 0, engine="openpyxl")
        elif file_type == "csv":
            df = pd.read_csv(path)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported dataset file_type: {file_type}",
            )

        df.columns = _validate_headers(df.columns)

        missing = [c for c in (x_cols + y_cols) if c not in df.columns]
        if missing:
            raise HTTPException(
                status_code=400, detail=f"Selected columns not found: {missing}"
            )

        X = df[x_cols]
        Y = df[y_cols]

        combined = pd.concat([X, Y], axis=1)
        total_rows = int(combined.shape[0])
        warnings = []

        if na_strategy == "drop":
            cleaned = combined.dropna()
        elif na_strategy == "error":
            if bool(combined.isna().any().any()):
                raise HTTPException(
                    status_code=400,
                    detail="Selected columns contain missing values. Choose a missing-value strategy.",
                )
            cleaned = combined
        else:
            raise HTTPException(
                status_code=400,
                detail="na_strategy must be one of: drop, error",
            )

        dropped_rows = int(total_rows - cleaned.shape[0])
        if dropped_rows > 0:
            warnings.append(
                f"Dropped {dropped_rows} rows with missing values in selected columns."
            )

        combined = cleaned
        X = combined[x_cols]
        Y = combined[y_cols]

        return {
            "x_data": X.values.tolist(),
            "y_data": Y.values.tolist(),
            "shape": {
                "x": [int(X.shape[0]), int(X.shape[1])],
                "y": [int(Y.shape[0]), int(Y.shape[1])],
            },
            "x_cols": x_cols,
            "y_cols": y_cols,
            "dataset_id": dataset_id,
            "file_type": file_type,
            "stats": {
                "total_rows": total_rows,
                "used_rows": int(combined.shape[0]),
                "dropped_rows": dropped_rows,
            },
            "warnings": warnings,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/regression_data")
async def regression_data(request: Request):
    """
    Generate surrogate regression data using sklearn.datasets.make_regression.
    """
    body = await request.json()

    n_samples = int(body.get("n_samples", 1000))
    n_features = int(body.get("n_features", 10))
    n_informative = int(body.get("n_informative", 5))
    n_targets = int(body.get("n_targets", 3))
    noise = float(body.get("noise", 1.0))
    random_state_raw = int(body.get("random_state", 42))

    random_state = None
    if random_state_raw is not None:
        try:
            random_state = int(random_state_raw)
        except (TypeError, ValueError):
            random_state = 42  # default if invalid

    try:
        x, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_targets=n_targets,
            noise=noise,
            random_state=random_state,
        )
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    return JSONResponse(
        content={
            "x": x.tolist(),
            "y": y.tolist(),
            # "x": x,
            # "y": y,
            "shape": {"x": list(x.shape), "y": list(y.shape)},
        }
    )


@router.post("/time_series_regression_data")
async def time_series_regression_data(request: Request):
    """Generate synthetic multivariate time-series regression data (2D tabular time axis)."""
    body = await request.json()

    n_samples = int(body.get("n_samples", 2000))
    n_features = int(body.get("n_features", 4))
    season_period = int(body.get("season_period", 24))
    noise = float(body.get("noise", 0.05))
    trend_strength = float(body.get("trend_strength", 0.001))
    random_state = int(body.get("random_state", 42))

    if n_samples < 10:
        return JSONResponse(
            status_code=400, content={"error": "n_samples must be >= 10"}
        )
    if n_features < 1:
        return JSONResponse(
            status_code=400, content={"error": "n_features must be >= 1"}
        )

    rng = np.random.default_rng(random_state)
    t = np.arange(n_samples, dtype=np.float32)

    x = np.zeros((n_samples, n_features), dtype=np.float32)
    for j in range(n_features):
        amp = 0.5 + 0.5 * (j + 1) / n_features
        phase = rng.uniform(0, 2 * np.pi)
        seasonal = amp * np.sin(2 * np.pi * t / max(season_period, 2) + phase)
        trend = trend_strength * (j + 1) * t
        ar = np.zeros(n_samples, dtype=np.float32)
        eps = rng.normal(0, noise, size=n_samples).astype(np.float32)
        for i in range(1, n_samples):
            ar[i] = 0.75 * ar[i - 1] + eps[i]
        x[:, j] = seasonal + trend + ar

    weights = rng.normal(0, 1.0, size=(n_features,)).astype(np.float32)
    y = (x @ weights).reshape(-1, 1)
    y += 0.2 * np.sin(2 * np.pi * t / max(season_period, 2)).reshape(-1, 1)
    y += rng.normal(0, noise, size=(n_samples, 1)).astype(np.float32)

    return JSONResponse(
        content={
            "x": x.tolist(),
            "y": y.tolist(),
            "shape": {
                "x": [int(x.shape[0]), int(x.shape[1])],
                "y": [int(y.shape[0]), int(y.shape[1])],
            },
            "meta": {
                "season_period": season_period,
                "trend_strength": trend_strength,
                "noise": noise,
            },
        }
    )


@router.post("/build_sequence_windows")
async def build_sequence_windows(request: Request):
    """Build sliding sequence windows for temporal supervised training."""
    body = await request.json()

    x_raw = body.get("x_data") or body.get("x")
    y_raw = body.get("y_data") or body.get("y")
    seq_length = int(body.get("seq_length", 24))
    stride = int(body.get("stride", 1))
    target_offset = int(body.get("target_offset", 0))

    if x_raw is None or y_raw is None:
        return JSONResponse(
            status_code=400, content={"error": "x_data and y_data are required."}
        )
    if seq_length < 2:
        return JSONResponse(
            status_code=400, content={"error": "seq_length must be >= 2."}
        )
    if stride < 1:
        return JSONResponse(status_code=400, content={"error": "stride must be >= 1."})

    x = np.asarray(x_raw, dtype=np.float32)
    y = np.asarray(y_raw, dtype=np.float32)

    if x.ndim != 2:
        return JSONResponse(
            status_code=400, content={"error": "x_data must be 2D [samples, features]."}
        )
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if y.ndim != 2:
        return JSONResponse(
            status_code=400, content={"error": "y_data must be 1D or 2D."}
        )
    if x.shape[0] != y.shape[0]:
        return JSONResponse(
            status_code=400,
            content={"error": "x_data and y_data sample sizes must match."},
        )

    n = x.shape[0]
    max_start = n - seq_length - target_offset
    if max_start < 0:
        return JSONResponse(
            status_code=400,
            content={
                "error": "Not enough samples for requested seq_length and target_offset."
            },
        )

    x_seq = []
    y_seq = []
    lengths = []
    for start in range(0, max_start + 1, stride):
        end = start + seq_length
        target_idx = end - 1 + target_offset
        x_seq.append(x[start:end])
        y_seq.append(y[target_idx])
        lengths.append(seq_length)

    x_seq = np.asarray(x_seq, dtype=np.float32)
    y_seq = np.asarray(y_seq, dtype=np.float32)
    lengths = np.asarray(lengths, dtype=np.int64)

    return JSONResponse(
        content={
            "x_seq": x_seq.tolist(),
            "y_seq": y_seq.tolist(),
            "lengths": lengths.tolist(),
            "shape": {
                "x_seq": [int(v) for v in x_seq.shape],
                "y_seq": [int(v) for v in y_seq.shape],
                "lengths": [int(v) for v in lengths.shape],
            },
        }
    )


@router.post("/vae_synthetic_data")
async def vae_synthetic_data(request: Request):
    """Generate clustered synthetic feature data for VAE workflows."""
    body = await request.json()

    n_samples = int(body.get("n_samples", 2000))
    n_features = int(body.get("n_features", 2))
    centers = int(body.get("centers", 5))
    cluster_std = float(body.get("cluster_std", 1.0))
    random_state = int(body.get("random_state", 42))

    if n_features < 2:
        return JSONResponse(
            status_code=400,
            content={
                "error": "n_features must be >= 2 for meaningful latent clustering."
            },
        )

    try:
        x, labels = make_blobs(
            n_samples=n_samples,
            n_features=n_features,
            centers=centers,
            cluster_std=cluster_std,
            random_state=random_state,
        )
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})

    return JSONResponse(
        content={
            "x": x.tolist(),
            "labels": labels.tolist(),
            "shape": {
                "x": [int(x.shape[0]), int(x.shape[1])],
                "labels": [int(labels.shape[0])],
            },
            "meta": {
                "centers": centers,
                "cluster_std": cluster_std,
            },
        }
    )


@router.post("/prepare_temporal_evaluation_data")
async def prepare_temporal_evaluation_data(request: Request):
    """
    Prepare raw 2D x/y data for temporal model evaluation without retraining.

    Performs the same split -> apply_external_scalers -> window pipeline as the
    temporal trainer, but accepts already-fitted scalers (e.g. from scaler_info.txt
    / Load Model node) instead of re-fitting on training data.

    Response format matches /evaluate_temporal_supervised_model expectations:
        x_data        - JSON string [x_train_3D, x_val_3D, x_test_3D]  (unscaled)
        y_data        - JSON string [y_train,    y_val,    y_test   ]  (unscaled truth)
        x_data_scaled - JSON string [x_train_3D, x_val_3D, x_test_3D]  (scaled, model input)
        y_data_scaled - JSON string [y_train,    y_val,    y_test   ]  (scaled)
        shape         - shape info of the windowed train split
        splits        - row counts for train/val/test after windowing
    """
    from .temporal_trainers import prepare_sequences_from_raw
    from .training_common import split_train_val_test_temporal
    from .utils import scaler_from_dict

    body = await request.json()

    x_raw = body.get("x") or body.get("x_data")
    y_raw = body.get("y") or body.get("y_data")

    if x_raw is None or y_raw is None:
        return JSONResponse(status_code=400, content={"error": "x and y are required."})

    val_size = float(body.get("val_size", 0.1))
    test_size = float(body.get("test_size", 0.1))
    seq_length = int(body.get("seq_length", 60))
    seq_to_one = bool(body.get("seq_to_one", True))

    if val_size <= 0 or test_size <= 0 or val_size + test_size >= 1.0:
        return JSONResponse(
            status_code=400,
            content={"error": "val_size and test_size must be > 0 and their sum < 1."},
        )
    if seq_length < 1:
        return JSONResponse(
            status_code=400, content={"error": "seq_length must be >= 1."}
        )

    try:
        x = np.asarray(x_raw, dtype=np.float64)
        y = np.asarray(y_raw, dtype=np.float64)
    except Exception as e:
        return JSONResponse(
            status_code=400, content={"error": f"Failed to parse x/y: {e}"}
        )

    if x.ndim != 2:
        return JSONResponse(
            status_code=400,
            content={
                "error": f"x must be 2D [samples, features], got shape {list(x.shape)}."
            },
        )
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if y.ndim != 2:
        return JSONResponse(
            status_code=400,
            content={"error": f"y must be 1D or 2D, got shape {list(y.shape)}."},
        )
    if x.shape[0] != y.shape[0]:
        return JSONResponse(
            status_code=400,
            content={"error": "x and y must have the same number of samples."},
        )

    try:
        x_tr, x_va, x_te, y_tr, y_va, y_te = split_train_val_test_temporal(
            x=x, y=y, val_size=val_size, test_size=test_size
        )
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Split failed: {e}"})

    split_lengths = {
        "train": int(x_tr.shape[0]),
        "val": int(x_va.shape[0]),
        "test": int(x_te.shape[0]),
    }
    too_short = [k for k, v in split_lengths.items() if v < seq_length]
    if too_short:
        return JSONResponse(
            status_code=400,
            content={
                "error": (
                    f"Splits {too_short} have fewer rows than seq_length={seq_length}. "
                    f"Raw split lengths: {split_lengths}."
                )
            },
        )

    # Apply external scalers if provided, otherwise identity (no scaling)
    x_scaler: Optional[object] = None
    y_scaler: Optional[object] = None
    x_norm_dict = body.get("x_normalizer") or {}
    y_norm_dict = body.get("y_normalizer") or {}

    if x_norm_dict:
        try:
            x_scaler = scaler_from_dict(x_norm_dict)
        except Exception as e:
            return JSONResponse(
                status_code=400, content={"error": f"Bad x_normalizer: {e}"}
            )
    if y_norm_dict:
        try:
            y_scaler = scaler_from_dict(y_norm_dict)
        except Exception as e:
            return JSONResponse(
                status_code=400, content={"error": f"Bad y_normalizer: {e}"}
            )

    def _sx(a: np.ndarray) -> np.ndarray:
        return x_scaler.transform(a) if x_scaler is not None else a

    def _sy(a: np.ndarray) -> np.ndarray:
        return y_scaler.transform(a) if y_scaler is not None else a

    x_tr_s = _sx(x_tr)
    x_va_s = _sx(x_va)
    x_te_s = _sx(x_te)
    y_tr_s = _sy(y_tr)
    y_va_s = _sy(y_va)
    y_te_s = _sy(y_te)

    # Window each split independently (notebook-parity: no cross-split leakage)
    try:
        X_tr, Y_tr = prepare_sequences_from_raw(x_tr, y_tr, seq_length, seq_to_one)
        X_va, Y_va = prepare_sequences_from_raw(x_va, y_va, seq_length, seq_to_one)
        X_te, Y_te = prepare_sequences_from_raw(x_te, y_te, seq_length, seq_to_one)
        X_tr_s, Y_tr_s = prepare_sequences_from_raw(
            x_tr_s, y_tr_s, seq_length, seq_to_one
        )
        X_va_s, Y_va_s = prepare_sequences_from_raw(
            x_va_s, y_va_s, seq_length, seq_to_one
        )
        X_te_s, Y_te_s = prepare_sequences_from_raw(
            x_te_s, y_te_s, seq_length, seq_to_one
        )
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"error": f"Windowing failed: {e}"}
        )

    windowed_counts = {
        "train": int(X_tr.shape[0]),
        "val": int(X_va.shape[0]),
        "test": int(X_te.shape[0]),
    }

    return JSONResponse(
        content={
            "x_data": json.dumps([X_tr.tolist(), X_va.tolist(), X_te.tolist()]),
            "y_data": json.dumps([Y_tr.tolist(), Y_va.tolist(), Y_te.tolist()]),
            "x_data_scaled": json.dumps(
                [X_tr_s.tolist(), X_va_s.tolist(), X_te_s.tolist()]
            ),
            "y_data_scaled": json.dumps(
                [Y_tr_s.tolist(), Y_va_s.tolist(), Y_te_s.tolist()]
            ),
            "shape": {
                "x_data": [int(v) for v in X_tr.shape],
                "y_data": [int(v) for v in Y_tr.shape],
                "x_data_scaled": [int(v) for v in X_tr_s.shape],
            },
            "splits": windowed_counts,
        }
    )
