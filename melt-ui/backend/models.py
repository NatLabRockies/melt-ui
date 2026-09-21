from __future__ import annotations

import inspect
import json
import os
import re
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from safetensors import safe_open
from safetensors.torch import load_file as safetensors_load_file
from safetensors.torch import save_file as safetensors_save_file

from .path_access import grant_file_access, resolve_read_file

router = APIRouter()

SAVED_MODELS_DIR = Path(__file__).resolve().parents[1] / "saved_models"


async def _maybe_await(x):
    return await x if inspect.isawaitable(x) else x


async def _ensure_model_store(app) -> Any:
    """Ensure app.state.model_store exists. Reuses attach_model_store from trainers.py if available."""
    if getattr(app.state, "model_store", None) is not None:
        return app.state.model_store

    # Try to attach the ModelStore used by trainers
    try:
        from .trainers import attach_model_store  # type: ignore

        attach_model_store(app, max_items=10, ttl_seconds=3600)
        return app.state.model_store
    except Exception:
        # Fall back to a plain dict store
        app.state.model_store = {}
        return app.state.model_store


async def _model_store_get(model_store: Any, model_id: str) -> dict | None:
    if model_store is None:
        return None
    getter = getattr(model_store, "get", None)
    if getter is None:
        return None
    entry = getter(model_id)
    return await _maybe_await(entry)


async def _model_store_set(model_store: Any, model_id: str, entry: dict) -> None:
    setter = getattr(model_store, "set", None)
    if callable(setter):
        maybe = setter(model_id, entry)
        await _maybe_await(maybe)
        return

    # dict fallback
    if isinstance(model_store, dict):
        model_store[model_id] = entry
        return

    raise RuntimeError("Unsupported model_store type (no .set and not a dict)")


def _as_cpu_contiguous_state_dict(
    state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Ensure tensors are CPU + contiguous, as required by safetensors."""
    out: dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if not isinstance(v, torch.Tensor):
            continue
        t = v.detach()
        if t.device.type != "cpu":
            t = t.to("cpu")
        if not t.is_contiguous():
            t = t.contiguous()
        out[str(k)] = t
    return out


def _infer_in_out_from_state_dict(
    state: Mapping[str, torch.Tensor],
) -> tuple[int | None, int | None]:
    """Best-effort inference of input/output dims from common weight shapes."""
    in_dim: int | None = None
    out_dim: int | None = None

    # Pick first and last 2D weights
    weights_2d = [
        (k, v) for k, v in state.items() if isinstance(v, torch.Tensor) and v.ndim == 2
    ]
    if weights_2d:
        _, first = weights_2d[0]
        _, last = weights_2d[-1]
        try:
            in_dim = int(first.shape[1])
        except Exception:
            in_dim = None
        try:
            out_dim = int(last.shape[0])
        except Exception:
            out_dim = None

    return in_dim, out_dim


def _find_tensor_by_suffix(state, suffix: str):
    """Find a tensor whose key exactly equals suffix or ends with '.' + suffix."""
    for k, v in state.items():
        ks = str(k)
        if ks == suffix or ks.endswith("." + suffix):
            return v
    return None


def _coerce_meta_value(raw: Any) -> Any:
    if raw is None:
        return None
    if isinstance(raw, (bool, int, float, list, dict)):
        return raw

    text = str(raw).strip()
    if text == "":
        return ""

    lo = text.lower()
    if lo in {"true", "false"}:
        return lo == "true"

    try:
        if re.fullmatch(r"[-+]?\d+", text):
            return int(text)
        if re.fullmatch(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", text) or re.fullmatch(
            r"[-+]?\d+(?:[eE][-+]?\d+)", text
        ):
            return float(text)
    except Exception:
        pass

    if (text.startswith("[") and text.endswith("]")) or (
        text.startswith("{") and text.endswith("}")
    ):
        try:
            return json.loads(text)
        except Exception:
            pass

    return text


def _as_float_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, list):
        out: list[float] = []
        for item in value:
            try:
                out.append(float(item))
            except Exception:
                return None
        return out
    if isinstance(value, str):
        txt = value.strip()
        if txt.startswith("[") and txt.endswith("]"):
            try:
                parsed = json.loads(txt)
                return _as_float_list(parsed)
            except Exception:
                pass
        parts = [p.strip() for p in txt.split(",") if p.strip()]
        if not parts:
            return None
        out = []
        for p in parts:
            try:
                out.append(float(p))
            except Exception:
                return None
        return out
    return None


def _build_standard_scaler_dict(
    mean_vals: list[float], scale_vals: list[float]
) -> dict:
    var_vals = [float(s) ** 2 for s in scale_vals]
    n_features = int(len(mean_vals))
    return {
        "class": "StandardScaler",
        "module": "sklearn.preprocessing._data",
        "params": {"copy": True, "with_mean": True, "with_std": True},
        "attributes": {
            "mean_": mean_vals,
            "scale_": scale_vals,
            "var_": var_vals,
            "n_features_in_": n_features,
            "n_samples_seen_": 1,
        },
    }


def _parse_scaler_info_text(text: str) -> dict[str, dict]:
    parsed_raw: dict[str, dict[str, list[float]]] = {
        "x_normalizer": {},
        "y_normalizer": {},
    }
    line_re = re.compile(r"^(x_normalizer|y_normalizer)\s+(mean|scale)\s*:\s*(.+)$")

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = line_re.match(line)
        if not m:
            continue
        key, field, raw_value = m.group(1), m.group(2), m.group(3)
        vals = _as_float_list(_coerce_meta_value(raw_value))
        if vals:
            parsed_raw[key][field] = vals

    out: dict[str, dict] = {}
    for key in ("x_normalizer", "y_normalizer"):
        mean_vals = parsed_raw[key].get("mean")
        scale_vals = parsed_raw[key].get("scale")
        if mean_vals and scale_vals and len(mean_vals) == len(scale_vals):
            out[key] = _build_standard_scaler_dict(mean_vals, scale_vals)
    return out


def _parse_metadata_scalars(meta_map: dict[str, str], target: dict[str, Any]) -> None:
    keys = [
        "architecture",
        "num_features",
        "num_outputs",
        "width",
        "depth",
        "rnn_type",
        "head_type",
        "node_list",
        "activation_function",
        "output_activation",
        "dropout_rate",
        "batch_norm",
        "l1_reg",
        "l2_reg",
        "num_mixtures",
        "random_state",
        "seq_length",
        "seq_to_one",
    ]
    for k in keys:
        if k in target:
            continue
        if k not in meta_map:
            continue
        val = _coerce_meta_value(meta_map[k])
        if val is not None:
            target[k] = val

    # PT-MELT notebook safetensors use different key names
    _NOTEBOOK_ALIASES = {
        "n_features": "num_features",
        "n_targets": "num_outputs",
        "act_fun": "activation_function",
    }
    for src_key, dst_key in _NOTEBOOK_ALIASES.items():
        if dst_key not in target and src_key in meta_map:
            val = _coerce_meta_value(meta_map[src_key])
            if val is not None:
                target[dst_key] = val

    # Map model_type string to architecture tag
    if "architecture" not in target and "model_type" in meta_map:
        _MODEL_TYPE_MAP = {
            "recurrentneuralnetwork": "rnn",
            "artificialneuralnetwork": "ann",
            "residualneuralnetwork": "resnet",
            "bayesianneuralnetwork": "bnn",
            "variationalautoencoder": "vae",
        }
        normalized = (
            str(meta_map["model_type"]).lower().replace("_", "").replace(" ", "")
        )
        arch = _MODEL_TYPE_MAP.get(normalized)
        if arch:
            target["architecture"] = arch


def _infer_rnn_init_from_state_dict(
    state: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    recurrent_keys = [
        k for k in state.keys() if "weight_ih_l" in str(k) or "weight_hh_l" in str(k)
    ]
    if not recurrent_keys:
        return out

    out["architecture"] = "rnn"

    layer_idxs = []
    for key in recurrent_keys:
        m = re.search(r"_l(\d+)", str(key))
        if m:
            try:
                layer_idxs.append(int(m.group(1)))
            except Exception:
                pass
    if layer_idxs:
        out["depth"] = max(layer_idxs) + 1

    w_ih0 = _find_tensor_by_suffix(state, "weight_ih_l0")
    w_hh0 = _find_tensor_by_suffix(state, "weight_hh_l0")

    if isinstance(w_ih0, torch.Tensor) and w_ih0.ndim == 2:
        out["num_features"] = int(w_ih0.shape[1])

    if isinstance(w_hh0, torch.Tensor) and w_hh0.ndim == 2:
        hidden_size = int(w_hh0.shape[1])
        out["width"] = hidden_size
        gate_rows = int(w_hh0.shape[0])
        if hidden_size > 0:
            gate_ratio = gate_rows // hidden_size
            if gate_ratio == 4:
                out["rnn_type"] = "lstm"
            elif gate_ratio == 3:
                out["rnn_type"] = "gru"
            else:
                out["rnn_type"] = "rnn"

    if "rnn_type" not in out:
        joined = " ".join(recurrent_keys).lower()
        out["rnn_type"] = "gru" if "gru" in joined else "lstm"

    out.setdefault("head_type", "last")

    # Infer output size from the last non-recurrent, non-attention 2D weight.
    output_weights = [
        (k, v)
        for k, v in state.items()
        if isinstance(v, torch.Tensor)
        and v.ndim == 2
        and "weight_ih_l" not in str(k)
        and "weight_hh_l" not in str(k)
        and "pool_head" not in str(k)
        and "attn" not in str(k)
    ]
    if output_weights:
        _, last_w = output_weights[-1]
        out["num_outputs"] = int(last_w.shape[0])

    return out


def _infer_num_mixtures_from_state_dict(state) -> int:
    """Infer num_mixtures: if MDN output layers (mean/log_var/mix_coeffs) exist -> 1."""
    keys_str = " ".join(str(k) for k in state.keys())
    if (
        "mean_layer" in keys_str
        and "log_var_layer" in keys_str
        and "mix_coeffs_layer" in keys_str
    ):
        return 1
    return 0


# def _extract_model_init(
#     model_obj: Any,
#     model_meta: Mapping[str, Any],
#     state_dict: Mapping[str, torch.Tensor],
# ) -> Dict[str, Any]:
#     """Create a compact init config that load can use."""
#     init_cfg: Dict[str, Any] = dict(model_meta or {})

#     # Normalize common key names
#     arch = (init_cfg.get("architecture") or init_cfg.get("arch") or "").lower()
#     if arch:
#         init_cfg["architecture"] = arch

#     # Try to capture explicit num_features/num_outputs
#     if "num_features" not in init_cfg:
#         for k in ("n_features", "in_features", "input_dim", "num_inputs"):
#             if k in init_cfg:
#                 init_cfg["num_features"] = init_cfg[k]
#                 break
#     if "num_outputs" not in init_cfg:
#         for k in ("n_outputs", "out_features", "output_dim", "num_targets"):
#             if k in init_cfg:
#                 init_cfg["num_outputs"] = init_cfg[k]
#                 break

#     # Heuristic fallback from state_dict
#     if "num_features" not in init_cfg or "num_outputs" not in init_cfg:
#         in_dim, out_dim = _infer_in_out_from_state_dict(state_dict)
#         init_cfg.setdefault("num_features", in_dim)
#         init_cfg.setdefault("num_outputs", out_dim)

#     return init_cfg


def _extract_model_init(
    model_obj: Any,
    model_meta: Mapping[str, Any],
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    """Create a compact init config that load can use."""
    init_cfg: dict[str, Any] = {}

    # Start with model_meta (trainer already populates most hparams)
    for k in [
        "architecture",
        "node_list",
        "width",
        "depth",
        "rnn_type",
        "head_type",
        "latent_dims",
        "encoder_node_list",
        "decoder_node_list",
        "activation_function",
        "output_activation",
        "dropout_rate",
        "batch_norm",
        "l1_reg",
        "l2_reg",
        "num_mixtures",
        "random_state",
    ]:
        if k in model_meta:
            init_cfg[k] = model_meta[k]

    # Try common attribute names
    for cand in ["num_features", "n_features", "in_features", "input_dim", "d_in"]:
        if hasattr(model_obj, cand):
            try:
                init_cfg["num_features"] = int(getattr(model_obj, cand))
                break
            except Exception:
                pass

    for cand in ["num_outputs", "n_outputs", "out_features", "output_dim", "d_out"]:
        if hasattr(model_obj, cand):
            try:
                init_cfg["num_outputs"] = int(getattr(model_obj, cand))
                break
            except Exception:
                pass

    # Heuristic fallback from state_dict
    if "num_features" not in init_cfg or "num_outputs" not in init_cfg:
        in_dim, out_dim = _infer_in_out_from_state_dict(state_dict)
        init_cfg.setdefault("num_features", in_dim)
        init_cfg.setdefault("num_outputs", out_dim)

    return init_cfg


# def _build_model_from_init(init_cfg: Mapping[str, Any]) -> Any:
#     """Reconstruct a PTMELT model using init_cfg."""

#     # Import lazily so the module can load even if ptmelt isn't installed
#     try:
#         from ptmelt.models import (  # type: ignore
#             ArtificialNeuralNetwork,
#             BayesianNeuralNetwork,
#             ResidualNeuralNetwork,
#         )
#     except Exception as e:
#         raise ImportError(
#             "Could not import ptmelt.models. Ensure the backend environment has 'ptmelt' installed."
#         ) from e

#     arch = (init_cfg.get("architecture") or "").lower()
#     num_features = init_cfg.get("num_features")
#     num_outputs = init_cfg.get("num_outputs")

#     if num_features is None or num_outputs is None:
#         raise ValueError("Missing num_features/num_outputs; cannot reconstruct model.")

#     common_kwargs = {
#         "num_features": int(num_features),
#         "num_outputs": int(num_outputs),
#         "node_list": init_cfg.get("node_list", []),
#         "act_fun": init_cfg.get("activation_function", "relu"),
#         "output_activation": init_cfg.get("output_activation", "linear"),
#         "dropout": float(init_cfg.get("dropout_rate", 0.0) or 0.0),
#         "batch_norm": bool(init_cfg.get("batch_norm", False)),
#         "l1_reg": float(init_cfg.get("l1_reg", 0.0) or 0.0),
#         "l2_reg": float(init_cfg.get("l2_reg", 0.0) or 0.0),
#         "num_mixtures": int(init_cfg.get("num_mixtures", 0) or 0),
#     }

#     if arch == "ann":
#         return ArtificialNeuralNetwork(**common_kwargs)
#     if arch == "resnet":
#         return ResidualNeuralNetwork(**common_kwargs)
#     if arch == "bnn":
#         return BayesianNeuralNetwork(**common_kwargs)


#     # Default to ANN if unknown
#     return ArtificialNeuralNetwork(**common_kwargs)
def _build_model_from_init(init_cfg: Mapping[str, Any]) -> Any:
    """Reconstruct a PTMELT model using init_cfg."""

    arch = (init_cfg.get("architecture") or "").lower()
    num_features = init_cfg.get("num_features")
    num_outputs = init_cfg.get("num_outputs")

    if num_features is None or num_outputs is None:
        raise ValueError(
            "Missing num_features/num_outputs in saved metadata; cannot reconstruct model reliably."
        )

    # Import the same classes trainers.py uses.
    try:
        from ptmelt.models import (  # type: ignore
            ArtificialNeuralNetwork,
            BayesianNeuralNetwork,
            RecurrentNeuralNetwork,
            ResidualNeuralNetwork,
            TemporalTransformerNetwork,
            VariationalAutoencoder,
        )
    except Exception as e:
        raise ImportError(
            "Could not import ptmelt.models. Ensure the backend environment has 'ptmelt' installed."
        ) from e

    common_kwargs = {
        "num_features": int(num_features),
        "num_outputs": int(num_outputs),
        "node_list": init_cfg.get("node_list", []),
        "act_fun": init_cfg.get("activation_function", "relu"),
        "output_activation": init_cfg.get("output_activation", "linear"),
        "dropout": float(init_cfg.get("dropout_rate", 0.0) or 0.0),
        "batch_norm": bool(init_cfg.get("batch_norm", False)),
        "l1_reg": float(init_cfg.get("l1_reg", 0.0) or 0.0),
        "l2_reg": float(init_cfg.get("l2_reg", 0.0) or 0.0),
        "num_mixtures": int(init_cfg.get("num_mixtures", 0) or 0),
    }

    print(f"Common kwargs: {common_kwargs}")

    if arch == "ann":
        print("Trying to return ANN...")
        return ArtificialNeuralNetwork(**common_kwargs)
    if arch == "resnet":
        return ResidualNeuralNetwork(**common_kwargs)
    if arch == "bnn":
        return BayesianNeuralNetwork(**common_kwargs)
    if arch == "rnn":
        rnn_kwargs = {
            **common_kwargs,
            "width": int(init_cfg.get("width", 32) or 32),
            "depth": int(init_cfg.get("depth", 2) or 2),
            "rnn_type": str(init_cfg.get("rnn_type", "lstm")).lower(),
            "head_type": str(init_cfg.get("head_type", "last")).lower(),
        }
        # RNN constructors generally use width/depth, not node_list
        rnn_kwargs.pop("node_list", None)
        return RecurrentNeuralNetwork(**rnn_kwargs)
    if arch in {"temporal_transformer", "transformer"}:
        transformer_kwargs = {
            **common_kwargs,
            "width": int(init_cfg.get("width", 64) or 64),
            "depth": int(init_cfg.get("depth", 2) or 2),
            "head_type": str(init_cfg.get("head_type", "last")).lower(),
            "num_heads": int(init_cfg.get("num_heads", 4) or 4),
            "ff_dim": (
                None
                if int(init_cfg.get("ff_dim", 0) or 0) == 0
                else int(init_cfg.get("ff_dim", 0) or 0)
            ),
            "max_seq_len": int(init_cfg.get("max_seq_len", 2048) or 2048),
            "use_causal_mask": bool(init_cfg.get("use_causal_mask", False)),
        }
        transformer_kwargs.pop("node_list", None)
        return TemporalTransformerNetwork(**transformer_kwargs)
    if arch == "vae":
        vae_kwargs = {
            **common_kwargs,
            "num_outputs": int(init_cfg.get("num_outputs") or num_features),
            "latent_dims": int(init_cfg.get("latent_dims", 8) or 8),
            "encoder_node_list": init_cfg.get("encoder_node_list", [32, 16]),
            "decoder_node_list": init_cfg.get("decoder_node_list", [16, 32]),
            "num_mixtures": int(init_cfg.get("num_mixtures", 1) or 1),
        }
        return VariationalAutoencoder(**vae_kwargs)

    # If unknown, try to default to ANN
    return ArtificialNeuralNetwork(**common_kwargs)


def _resolve_path(save_dir: str, filename: str) -> Path:
    save_dir = save_dir or "saved_models"
    base = Path(save_dir)
    if not base.is_absolute():
        base = (Path.cwd() / base).resolve()

    fname = filename or "model.safetensors"
    if not fname.endswith(".safetensors"):
        fname += ".safetensors"

    # Prevent path separators in filename
    fname = os.path.basename(fname)

    return (base / fname).resolve()


def _avoid_overwrite(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return path.with_name(f"{stem}_{ts}{path.suffix}")


@router.post("/save_model")
async def save_model(request: Request):

    body = await request.json()

    model_dict = body.get("model") or {}
    # model_id = model_ref.get("model_id")
    # model_meta = model_ref.get("model_meta") or {}
    save_dir = str(body.get("save_dir") or "saved_models")
    filename = body.get("filename")  # optional
    overwrite = bool(body.get("overwrite", False))
    include_history = bool(body.get("include_history", False))

    # upack the model which is actually model_id and model_meta dict
    if isinstance(model_dict, dict) and "model_id" in model_dict:
        model_id = model_dict["model_id"]
        print(f"Model ID: {model_id}")
        model_store = getattr(request.app.state, "model_store", {})
        model_dict = model_store.get(model_id)
        if inspect.isawaitable(model_dict):
            model_dict = await model_dict
        if model_dict is None:
            return {"error": f"Model with id {model_id} not found."}
    else:
        return {"error": "Invalid model format. Expected a dict with 'model_id'."}

    # get the model from the model_dict
    model = model_dict["model"]
    model_metadata = model_dict["model_meta"]

    # if not model_id:
    #     return JSONResponse(
    #         status_code=400, content={"error": "Missing model.model_id"}
    #     )

    # model_store = await _ensure_model_store(request.app)
    # entry = await _model_store_get(model_store, str(model_id))
    # if not entry:
    #     return JSONResponse(
    #         status_code=404,
    #         content={"error": f"Model id '{model_id}' not found in model store."},
    #     )

    # model_obj = entry.get("model")
    # if model_obj is None:
    #     return JSONResponse(
    #         status_code=404,
    #         content={"error": f"Model entry for '{model_id}' has no 'model' object."},
    #     )

    # # Produce state dict (CPU + contiguous)
    # try:
    #     raw_state = model_obj.state_dict()
    # except Exception as e:
    #     return JSONResponse(
    #         status_code=500, content={"error": f"Failed to get state_dict: {e}"}
    #     )

    # if not isinstance(raw_state, Mapping):
    #     return JSONResponse(
    #         status_code=400, content={"error": "state_dict() did not return a mapping."}
    #     )

    # state = _as_cpu_contiguous_state_dict(raw_state)

    # Prepare output path
    out_path = _resolve_path(save_dir, str(filename or f"{model_id}.safetensors"))
    if not overwrite:
        out_path = _avoid_overwrite(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Build metadata (string->string)
    saved_at = datetime.now(UTC).isoformat()
    init_cfg = _extract_model_init(model, model_metadata, model.state_dict())

    meta_map: dict[str, str] = {
        "format": "ptmelt.safetensors.v1",
        "saved_at": saved_at,
        "model_id": str(model_id),
        # "model_class": f"{model.__class__.__module__}.{model.__class__.__name__}",
        "model_meta": json.dumps(model_metadata, ensure_ascii=False),
        "model_init": json.dumps(init_cfg, ensure_ascii=False),
        "pytorch_version": getattr(torch, "__version__", "unknown"),
    }

    print(f"Meta map: {meta_map}")

    if include_history and hasattr(model, "history"):
        try:
            history = model_dict.get("history", getattr(model, "history", None))
            if history is not None:
                meta_map["history"] = json.dumps(history, ensure_ascii=False)
        except Exception:
            pass

    print("Trying to save...")
    print(f"outpath: {out_path}")
    # Save tensors
    try:
        safetensors_save_file(model.state_dict(), str(out_path), metadata=meta_map)  # type: ignore[misc]
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"error": f"Failed to save safetensors: {e}"}
        )

    return JSONResponse(content={"path": str(out_path), "metadata": meta_map})


# @router.post("/load_model")
# async def load_model(request: Request):

#     body = await request.json()
#     path_str = body.get("path")
#     requested_model_id = body.get("model_id")  # optional
#     strict = bool(body.get("strict", True))

#     if not path_str:
#         return JSONResponse(status_code=400, content={"error": "Missing 'path'."})

#     path = Path(str(path_str)).expanduser()
#     if not path.is_absolute():
#         path = (Path.cwd() / path).resolve()

#     if not path.exists():
#         return JSONResponse(
#             status_code=404, content={"error": f"File not found: {path}"}
#         )

#     # Read metadata via safe_open, then load tensors
#     try:
#         with safe_open(str(path), framework="pt", device="cpu") as f:  # type: ignore[misc]
#             md = {}
#             # Some versions expose metadata() in the Python bindings; be defensive.
#             if hasattr(f, "metadata") and callable(getattr(f, "metadata")):
#                 try:
#                     md = f.metadata() or {}
#                 except Exception:
#                     md = {}
#             meta_map = dict(md)
#     except Exception as e:
#         return JSONResponse(
#             status_code=400,
#             content={"error": f"Failed to read safetensors metadata: {e}"},
#         )

#     try:
#         state = safetensors_load_file(str(path), device="cpu")  # type: ignore[misc]
#     except Exception as e:
#         return JSONResponse(
#             status_code=400,
#             content={"error": f"Failed to load safetensors tensors: {e}"},
#         )

#     # Parse JSON-encoded metadata fields
#     model_meta_str = meta_map.get("model_meta", "{}")
#     model_init_str = meta_map.get("model_init", "{}")

#     try:
#         model_meta = json.loads(model_meta_str) if model_meta_str else {}
#     except Exception:
#         model_meta = {}

#     try:
#         init_cfg = json.loads(model_init_str) if model_init_str else {}
#     except Exception:
#         init_cfg = {}

#     # If init_cfg missing, fall back to model_meta
#     if not init_cfg:
#         init_cfg = dict(model_meta or {})

#     load_warnings: Dict[str, Any] = {}

#     # Rebuild model and load weights
#     try:
#         model_obj = _build_model_from_init(init_cfg)
#         # PTMELT models often require build() before load_state_dict
#         build_fn = getattr(model_obj, "build", None)
#         if callable(build_fn):
#             build_fn()
#         missing_keys, unexpected_keys = model_obj.load_state_dict(state, strict=strict)
#         if missing_keys or unexpected_keys:
#             load_warnings["missing_keys"] = list(missing_keys)
#             load_warnings["unexpected_keys"] = list(unexpected_keys)
#     except Exception as e:
#         return JSONResponse(
#             status_code=500, content={"error": f"Failed to reconstruct/load model: {e}"}
#         )

#     # Determine model id and store
#     new_model_id = str(
#         requested_model_id or meta_map.get("model_id") or path.stem or uuid.uuid4().hex
#     )
#     # Avoid collisions
#     model_store = await _ensure_model_store(request.app)
#     existing = await _model_store_get(model_store, new_model_id)
#     if existing:
#         new_model_id = f"{new_model_id}_{uuid.uuid4().hex[:8]}"

#     entry = {
#         "model": model_obj,
#         "model_meta": model_meta,
#         "loaded_from": str(path),
#         "loaded_at": datetime.now(timezone.utc).isoformat(),
#         "load_warnings": load_warnings,
#     }

#     # Restore optional history
#     if "history" in meta_map:
#         try:
#             entry["history"] = json.loads(meta_map["history"])
#         except Exception:
#             entry["history"] = meta_map["history"]

#     try:
#         await _model_store_set(model_store, new_model_id, entry)
#     except Exception as e:
#         return JSONResponse(
#             status_code=500, content={"error": f"Failed to store loaded model: {e}"}
#         )

#     return JSONResponse(
#         content={
#             "model_id": new_model_id,
#             "model_meta": model_meta,
#             "path": str(path),
#             "metadata": meta_map,
#             "load_warnings": load_warnings,
#         }
#     )


@router.post("/load_model")
async def load_model(request: Request):
    body = await request.json()

    # err = _require_safetensors()
    # if err is not None:
    #     return err

    requested_model_id = body.get("model_id")  # optional override
    strict = bool(body.get("strict", True))
    scaler_info_path = body.get("scaler_info_path")
    auto_load_scaler_info = bool(body.get("auto_load_scaler_info", True))

    # Accept an explicit model path.
    path_in = body.get("path") or body.get("file_path")

    print(f"Path in: {path_in}")

    if not path_in:
        return JSONResponse(
            status_code=400,
            content={"error": "Provide 'path' or 'file_path'."},
        )

    try:
        path = resolve_read_file(
            request.app,
            str(path_in),
            body.get("path_grant"),
            managed_roots=(SAVED_MODELS_DIR,),
        )
    except PermissionError as exc:
        return JSONResponse(status_code=403, content={"error": str(exc)})

    if not path.exists() or not path.is_file():
        return JSONResponse(
            status_code=404, content={"error": f"File not found: {path}"}
        )

    # Read metadata (header) via safe_open, then load tensors via safetensors.torch
    try:
        with safe_open(str(path), framework="pt", device="cpu") as f:  # type: ignore[misc]
            md = {}
            if hasattr(f, "metadata") and callable(f.metadata):
                try:
                    md = f.metadata() or {}
                except Exception:
                    md = {}
            meta_map: dict[str, str] = {str(k): str(v) for k, v in dict(md).items()}
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": f"Failed to read safetensors metadata: {e}"},
        )

    try:
        tensors = safetensors_load_file(str(path), device="cpu")  # type: ignore[misc]
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": f"Failed to load safetensors tensors: {e}"},
        )

    # print(f"Tensors: {tensors}")

    # Parse metadata
    model_meta_str = meta_map.get("model_meta", "{}")
    model_init_str = meta_map.get("model_init", "{}")

    try:
        model_meta = json.loads(model_meta_str) if model_meta_str else {}
    except Exception:
        model_meta = {}
    if not isinstance(model_meta, dict):
        model_meta = {}

    try:
        init_cfg = json.loads(model_init_str) if model_init_str else {}
    except Exception:
        init_cfg = {}
    if not isinstance(init_cfg, dict):
        init_cfg = {}

    # If init_cfg missing, fall back to model_meta
    if not init_cfg:
        init_cfg = dict(model_meta)

    _parse_metadata_scalars(meta_map, model_meta)
    _parse_metadata_scalars(meta_map, init_cfg)

    rnn_guess = _infer_rnn_init_from_state_dict(tensors)
    if "architecture" not in init_cfg and "architecture" in rnn_guess:
        init_cfg["architecture"] = rnn_guess["architecture"]
    for key in [
        "num_features",
        "num_outputs",
        "width",
        "depth",
        "rnn_type",
        "head_type",
    ]:
        if key not in init_cfg and key in rnn_guess:
            init_cfg[key] = rnn_guess[key]

    if "architecture" not in init_cfg and "architecture" in model_meta:
        init_cfg["architecture"] = model_meta["architecture"]

    if "num_features" not in init_cfg or "num_outputs" not in init_cfg:
        in_dim, out_dim = _infer_in_out_from_state_dict(tensors)
        if "num_features" not in init_cfg and in_dim is not None:
            init_cfg["num_features"] = in_dim
        if "num_outputs" not in init_cfg and out_dim is not None:
            init_cfg["num_outputs"] = out_dim

    # Infer num_mixtures from MDN output layer structure when absent in metadata
    if "num_mixtures" not in init_cfg:
        init_cfg["num_mixtures"] = _infer_num_mixtures_from_state_dict(tensors)

    x_normalizer = {}
    y_normalizer = {}
    scaler_payload = None

    if scaler_info_path:
        try:
            scaler_path = resolve_read_file(
                request.app,
                str(scaler_info_path),
                body.get("scaler_info_grant"),
                managed_roots=(SAVED_MODELS_DIR,),
            )
        except PermissionError as exc:
            return JSONResponse(status_code=403, content={"error": str(exc)})

        if scaler_path.exists() and scaler_path.is_file():
            try:
                scaler_payload = _parse_scaler_info_text(scaler_path.read_text())
            except Exception:
                scaler_payload = None
    elif auto_load_scaler_info:
        sidecar_candidates = [
            path.with_name("scaler_info.txt"),
            path.with_name(f"{path.stem}_scaler_info.txt"),
        ]
        for candidate in sidecar_candidates:
            try:
                candidate = resolve_read_file(
                    request.app,
                    candidate,
                    body.get("scaler_info_grant"),
                    managed_roots=(SAVED_MODELS_DIR,),
                )
            except PermissionError:
                continue

            if candidate.exists() and candidate.is_file():
                try:
                    scaler_payload = _parse_scaler_info_text(candidate.read_text())
                    break
                except Exception:
                    scaler_payload = None

    if isinstance(scaler_payload, dict):
        x_normalizer = scaler_payload.get("x_normalizer") or {}
        y_normalizer = scaler_payload.get("y_normalizer") or {}

    if y_normalizer and "y_data_is_scaled" not in model_meta:
        model_meta["y_data_is_scaled"] = True

    print(f"model meta: {model_meta_str}")
    print(f"model init: {model_init_str}")
    print(f"init cfg: {init_cfg}")

    # Build model + load weights
    try:
        model_obj = _build_model_from_init(init_cfg)
        print(f"Returned model obj: {model_obj}")
        # Some PTMELT models expose a build() method; call it if present.
        build_fn = getattr(model_obj, "build", None)
        if callable(build_fn):
            build_fn()

        model_keys = set(model_obj.state_dict().keys())
        state_filtered = {k: v for k, v in tensors.items() if k in model_keys}

        missing, unexpected = model_obj.load_state_dict(state_filtered, strict=strict)
        # # Torch may return a namedtuple-like object or a (missing, unexpected) tuple.
        # missing = getattr(res, "missing_keys", None)
        # unexpected = getattr(res, "unexpected_keys", None)

        print(f"missing: {missing}")
        print(f"unexpected: {unexpected}")
        # if (
        #     missing is None
        #     and unexpected is None
        #     and isinstance(res, tuple)
        #     and len(res) == 2
        # ):
        #     missing, unexpected = res
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"error": f"Failed to reconstruct/load model: {e}"}
        )

    # print(f"succesfully loaded: {res}")

    # Store in model_store under a new id
    model_store = await _ensure_model_store(request.app)
    new_model_id = str(
        requested_model_id or meta_map.get("model_id") or uuid.uuid4().hex
    )

    # Avoid collisions
    try:
        existing = await _model_store_get(model_store, new_model_id)
        if existing is not None:
            new_model_id = f"{new_model_id}_{uuid.uuid4().hex[:8]}"
    except Exception:
        new_model_id = f"{new_model_id}_{uuid.uuid4().hex[:8]}"

    entry = {
        "model": model_obj,
        "model_meta": model_meta,
        "x_normalizer": x_normalizer,
        "y_normalizer": y_normalizer,
        "loaded_from": str(path),
        "load_warnings": {
            "missing_keys": list(missing) if isinstance(missing, (list, tuple)) else [],
            "unexpected_keys": (
                list(unexpected) if isinstance(unexpected, (list, tuple)) else []
            ),
            "strict": strict,
        },
    }

    # Restore history if present
    if "history" in meta_map:
        try:
            entry["history"] = json.loads(meta_map["history"])
        except Exception:
            pass

    try:
        await _model_store_set(model_store, new_model_id, entry)
    except Exception as e:
        return JSONResponse(
            status_code=500, content={"error": f"Failed to store loaded model: {e}"}
        )

    return JSONResponse(
        content={
            "model_id": new_model_id,
            "model_meta": model_meta,
            "path": str(path),
            "metadata": meta_map,
            "load_warnings": entry.get("load_warnings", {}),
            "x_normalizer": x_normalizer,
            "y_normalizer": y_normalizer,
        }
    )


@router.post("/browse_file")
async def browse_file(request: Request):
    """Open a native OS file picker on the server and return the selected path."""
    import asyncio
    import sys

    body = await request.json()
    title = str(body.get("title", "Select File"))
    filetypes_raw = body.get("filetypes", [["All files", "*.*"]])
    initial_dir = str(body.get("initial_dir", str(Path.home())))

    ft_repr = repr([(str(a), str(b)) for a, b in filetypes_raw])
    script = (
        "import tkinter as tk; from tkinter import filedialog; "
        "root = tk.Tk(); root.withdraw(); root.wm_attributes('-topmost', True); "
        f"p = filedialog.askopenfilename(title={repr(title)}, filetypes={ft_repr}, initialdir={repr(initial_dir)}); "
        "print(p or '', end='')"
    )

    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            script,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=120.0)
        selected = stdout.decode("utf-8", errors="replace").strip()
        if not selected:
            return JSONResponse(content={"path": None, "cancelled": True})

        try:
            selected_path = Path(selected).expanduser().resolve()
            grant_id = grant_file_access(request.app, selected_path)
        except ValueError as exc:
            return JSONResponse(status_code=400, content={"error": str(exc)})

        return JSONResponse(
            content={
                "path": str(selected_path),
                "grant_id": grant_id,
                "cancelled": False,
            }
        )
    except TimeoutError:
        return JSONResponse(
            status_code=408, content={"error": "File dialog timed out."}
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Browse failed: {e}"})

    # body = await request.json()

    # path_str = body.get("path")
    # requested_model_id = body.get("model_id")  # optional override
    # strict = bool(body.get("strict", True))

    # print(f"path: {path_str}")

    # if not path_str:
    #     return JSONResponse(status_code=400, content={"error": "Missing 'path'."})

    # path = Path(str(path_str)).expanduser()
    # if not path.is_absolute():
    #     path = (Path.cwd() / path).resolve()

    # if not path.exists():
    #     return JSONResponse(
    #         status_code=404, content={"error": f"File not found: {path}"}
    #     )

    # # ---- read metadata + tensors ----
    # try:
    #     # with safe_open(str(path), framework="pt", device="cpu") as f:  # type: ignore[misc]
    #     #     md = {}
    #     #     if hasattr(f, "metadata") and callable(getattr(f, "metadata")):
    #     #         try:
    #     #             md = f.metadata() or {}
    #     #         except Exception:
    #     #             md = {}
    #     #     meta_map: Dict[str, str] = dict(md)
    #     with safe_open(path, framework="pt") as f:
    #         meta = f.metadata()
    #         tensor_names = list(f.keys())

    #         meta_map: Dict[str, str] = dict(meta)
    # except Exception as e:
    #     return JSONResponse(
    #         status_code=400,
    #         content={"error": f"Failed to read safetensors metadata: {e}"},
    #     )

    # try:
    #     state = safetensors_load_file(str(path), device="cpu")  # type: ignore[misc]
    # except Exception as e:
    #     return JSONResponse(
    #         status_code=400,
    #         content={"error": f"Failed to load safetensors tensors: {e}"},
    #     )

    # # ---- parse required metadata ----
    # model_meta_str = meta_map.get("model_meta")
    # model_init_str = meta_map.get("model_init")

    # if not model_meta_str:
    #     return JSONResponse(
    #         status_code=400,
    #         content={"error": "Missing 'model_meta' in safetensors metadata."},
    #     )
    # if not model_init_str:
    #     return JSONResponse(
    #         status_code=400,
    #         content={"error": "Missing 'model_init' in safetensors metadata."},
    #     )

    # try:
    #     model_meta = json.loads(model_meta_str) if model_meta_str else {}
    # except Exception as e:
    #     return JSONResponse(
    #         status_code=400, content={"error": f"Failed to parse model_meta JSON: {e}"}
    #     )

    # try:
    #     init_cfg = json.loads(model_init_str) if model_init_str else {}
    # except Exception as e:
    #     return JSONResponse(
    #         status_code=400, content={"error": f"Failed to parse model_init JSON: {e}"}
    #     )

    # if not isinstance(model_meta, dict):
    #     model_meta = {}
    # if not isinstance(init_cfg, dict):
    #     init_cfg = {}

    # # ---- reconstruct model + load weights ----
    # load_warnings: Dict[str, Any] = {}
    # try:
    #     print("Trying to build the model...")
    #     model_obj = _build_model_from_init(init_cfg)

    #     print("Model built...")

    #     build_fn = getattr(model_obj, "build", None)
    #     if callable(build_fn):
    #         build_fn()

    #     res = model_obj.load_state_dict(state, strict=strict)

    #     # handle both torch return styles
    #     missing_keys = getattr(res, "missing_keys", None)
    #     unexpected_keys = getattr(res, "unexpected_keys", None)
    #     if (
    #         missing_keys is None
    #         and unexpected_keys is None
    #         and isinstance(res, tuple)
    #         and len(res) == 2
    #     ):
    #         missing_keys, unexpected_keys = res

    #     if missing_keys:
    #         load_warnings["missing_keys"] = list(missing_keys)
    #     if unexpected_keys:
    #         load_warnings["unexpected_keys"] = list(unexpected_keys)

    # except Exception as e:
    #     return JSONResponse(
    #         status_code=500,
    #         content={"error": f"Failed to reconstruct/load model: {e}"},
    #     )

    # # ---- store in model_store (same style as save_model) ----
    # model_store = getattr(request.app.state, "model_store", {})
    # new_model_id = str(
    #     requested_model_id or meta_map.get("model_id") or path.stem or uuid.uuid4().hex
    # )

    # # avoid collisions
    # try:
    #     existing = model_store.get(new_model_id)
    #     if inspect.isawaitable(existing):
    #         existing = await existing
    #     if existing is not None:
    #         new_model_id = f"{new_model_id}_{uuid.uuid4().hex[:8]}"
    # except Exception:
    #     new_model_id = f"{new_model_id}_{uuid.uuid4().hex[:8]}"

    # entry = {
    #     "model": model_obj,
    #     "model_meta": model_meta,
    #     "loaded_from": str(path),
    #     "loaded_at": datetime.now(timezone.utc).isoformat(),
    #     "load_warnings": load_warnings,
    # }

    # # optional history
    # if "history" in meta_map:
    #     try:
    #         entry["history"] = json.loads(meta_map["history"])
    #     except Exception:
    #         entry["history"] = meta_map["history"]

    # try:
    #     setter = getattr(model_store, "set", None)
    #     if callable(setter):
    #         maybe = setter(new_model_id, entry)
    #         if inspect.isawaitable(maybe):
    #             await maybe
    #     elif isinstance(model_store, dict):
    #         model_store[new_model_id] = entry
    #     else:
    #         return JSONResponse(
    #             status_code=500,
    #             content={
    #                 "error": "Unsupported model_store type (no .set and not a dict)."
    #             },
    #         )
    # except Exception as e:
    #     return JSONResponse(
    #         status_code=500,
    #         content={"error": f"Failed to store loaded model: {e}"},
    #     )

    # # return a handle in the same shape other nodes expect
    # return JSONResponse(
    #     content={
    #         "model": {"model_id": new_model_id, "model_meta": model_meta},
    #         "model_id": new_model_id,
    #         "model_meta": model_meta,
    #         "path": str(path),
    #         "metadata": meta_map,
    #         "load_warnings": load_warnings,
    #     }
    # )
