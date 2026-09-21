import asyncio
import json
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from .temporal_trainers import prepare_sequences_from_raw
from .training_common import (
    HYPERPARAMETER_ALIASES,
    HYPERPARAMETER_ALLOWLISTS,
    TrainingCancelledError,
    clear_training_run,
    make_dataloaders,
    register_training_run,
    scale_splits,
    split_train_val_test,
    split_train_val_test_temporal,
)

router = APIRouter()

TUNING_RESULTS_DIR = Path(__file__).resolve().parents[1] / "tuning_results"

_ARCH_TO_PTMELT = {
    "ann": "ann",
    "resnet": "resnet",
    "bnn": "bnn",
    "rnn": "rnn",
    "temporal_rnn": "rnn",
    "temporal_transformer": "temporal_transformer",
    "transformer": "temporal_transformer",
    "vae": "vae",
}

_TRAINER_FAMILY_ALIASES = {
    "static": "static",
    "static_supervised": "static",
    "supervised": "static",
    "ann": "static",
    "resnet": "static",
    "bnn": "static",
    "rnn": "temporal_rnn",
    "temporal": "temporal_rnn",
    "temporal_rnn": "temporal_rnn",
    "temporal_supervised": "temporal_rnn",
    "temporal_transformer": "temporal_transformer",
    "transformer": "temporal_transformer",
    "vae": "vae",
}

_PARAMETER_TEMPLATES = {
    "common": [
        {
            "key": "learning_rate",
            "label": "Learning Rate",
            "category": "Optimization",
            "default": {"type": "choice", "values": [1e-3, 3e-4, 1e-4]},
            "supported_types": ["choice", "loguniform", "uniform", "fixed"],
            "description": "Optimizer learning rate.",
        },
        {
            "key": "dropout",
            "label": "Dropout",
            "category": "Regularization",
            "default": {"type": "choice", "values": [0.0, 0.1, 0.2]},
            "supported_types": ["choice", "uniform", "fixed"],
            "description": "Dropout probability used by PT-MELT model blocks.",
        },
        {
            "key": "l2_reg",
            "label": "L2 Regularization",
            "category": "Regularization",
            "default": {
                "type": "loguniform",
                "low": 1e-6,
                "high": 1e-2,
                "include_zero": True,
            },
            "supported_types": ["choice", "loguniform", "qloguniform", "fixed"],
            "allow_zero": True,
            "zero_label": "No regularization",
            "description": "L2 weight regularization strength.",
        },
        {
            "key": "batch_size",
            "label": "Batch Size",
            "category": "Optimization",
            "default": {"type": "choice", "values": [16, 32, 64]},
            "value_kind": "int",
            "min": 1,
            "supported_types": ["choice", "randint", "fixed"],
            "description": "Training batch size.",
        },
        {
            "key": "num_mixtures",
            "label": "Num Mixtures",
            "category": "Output",
            "default": {"type": "choice", "values": [0, 1, 3]},
            "vae_default": {"type": "choice", "values": [1, 3, 5]},
            "value_kind": "int",
            "min": 0,
            "supported_types": ["choice", "randint", "fixed"],
            "description": "Number of mixture-density output components. VAE searches use at least one mixture.",
        },
    ],
    "static": [
        {
            "key": "width",
            "label": "Width",
            "category": "Architecture",
            "default": {"type": "choice", "values": [16, 32, 64]},
            "supported_types": ["choice", "randint", "fixed"],
            "description": "Uniform hidden width when node_list is not tuned.",
        },
        {
            "key": "depth",
            "label": "Depth",
            "category": "Architecture",
            "default": {"type": "choice", "values": [1, 2, 3]},
            "supported_types": ["choice", "randint", "fixed"],
            "description": "Number of hidden layers when node_list is not tuned.",
        },
        {
            "key": "node_list",
            "label": "Node List",
            "category": "Architecture",
            "default": {"type": "choice", "values": [[8, 8], [16, 16], [32, 32]]},
            "supported_types": ["choice", "independent_layer_choice", "fixed"],
            "description": "Explicit hidden layer widths.",
            "supports_independent_layers": True,
            "layer_grouping_by_architecture": {"ann": "single", "bnn": "single"},
        },
    ],
    "temporal_rnn": [
        {
            "key": "width",
            "label": "Width",
            "category": "Architecture",
            "default": {"type": "choice", "values": [16, 32, 64]},
            "supported_types": ["choice", "randint", "fixed"],
            "description": "RNN hidden size.",
        },
        {
            "key": "depth",
            "label": "Depth",
            "category": "Architecture",
            "default": {"type": "choice", "values": [1, 2, 3]},
            "supported_types": ["choice", "randint", "fixed"],
            "description": "Number of recurrent layers.",
        },
        {
            "key": "node_list",
            "label": "Paired Layer Widths",
            "category": "Architecture",
            "default": {"type": "choice", "values": [[16, 16], [32, 32]]},
            "supported_types": ["independent_layer_choice", "fixed"],
            "description": "Paired RNN layer widths (each pair has matching widths).",
            "supports_independent_layers": True,
            "layer_grouping_by_architecture": {"rnn": "paired_equal"},
        },
        {
            "key": "rnn_type",
            "label": "RNN Type",
            "category": "Temporal",
            "default": {"type": "choice", "values": ["lstm", "gru", "rnn"]},
            "supported_types": ["choice", "fixed"],
            "description": "Recurrent cell type.",
        },
        {
            "key": "head_type",
            "label": "Head Type",
            "category": "Temporal",
            "default": {"type": "choice", "values": ["last", "mean", "max", "attn"]},
            "supported_types": ["choice", "fixed"],
            "description": "Sequence pooling head.",
        },
    ],
    "temporal_transformer": [
        {
            "key": "width",
            "label": "Model Width",
            "category": "Architecture",
            "default": {"type": "choice", "values": [32, 64, 128]},
            "supported_types": ["choice", "fixed"],
            "description": "Transformer model dimension. Must be even and divisible by num_heads.",
        },
        {
            "key": "depth",
            "label": "Depth",
            "category": "Architecture",
            "default": {"type": "choice", "values": [1, 2, 3]},
            "supported_types": ["choice", "randint", "fixed"],
            "description": "Number of encoder layers.",
        },
        {
            "key": "num_heads",
            "label": "Attention Heads",
            "category": "Architecture",
            "default": {"type": "choice", "values": [2, 4, 8]},
            "supported_types": ["choice", "fixed"],
            "description": "Number of attention heads.",
        },
        {
            "key": "head_type",
            "label": "Pooling Head",
            "category": "Temporal",
            "default": {"type": "choice", "values": ["last", "mean", "max", "attn"]},
            "supported_types": ["choice", "fixed"],
            "description": "Sequence pooling head.",
        },
    ],
    "vae": [
        {
            "key": "latent_dims",
            "label": "Latent Dims",
            "category": "VAE",
            "default": {"type": "choice", "values": [2, 4, 8, 16]},
            "supported_types": ["choice", "randint", "fixed"],
            "description": "Latent space dimensionality.",
        },
        {
            "key": "encoder_node_list",
            "label": "Encoder Nodes",
            "category": "VAE",
            "default": {"type": "choice", "values": [[32, 16], [64, 32], [128, 64]]},
            "supported_types": ["choice", "fixed"],
            "description": "Encoder hidden layer widths.",
        },
        {
            "key": "decoder_node_list",
            "label": "Decoder Nodes",
            "category": "VAE",
            "default": {"type": "choice", "values": [[16, 32], [32, 64], [64, 128]]},
            "supported_types": ["choice", "fixed"],
            "description": "Decoder hidden layer widths.",
        },
    ],
}


def _normalize_trainer_family(value: Any) -> str:
    key = str(value or "static").strip().lower().replace("-", "_")
    family = _TRAINER_FAMILY_ALIASES.get(key, key)
    if family not in {"static", "temporal_rnn", "temporal_transformer", "vae"}:
        raise ValueError(
            "trainer_family must be one of: static, temporal_rnn, "
            "temporal_transformer, vae."
        )
    return family


def _normalize_architecture(value: Any, trainer_family: str) -> str:
    if trainer_family == "temporal_rnn":
        return "rnn"
    if trainer_family == "temporal_transformer":
        return "temporal_transformer"
    if trainer_family == "vae":
        return "vae"

    key = str(value or "ann").strip().lower().replace("-", "_")
    arch = _ARCH_TO_PTMELT.get(key, key)
    if arch not in {"ann", "resnet", "bnn"}:
        raise ValueError("static trainer architecture must be ann, resnet, or bnn.")
    return arch


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "to_dict"):
        try:
            return _json_safe(value.to_dict(orient="records"))
        except TypeError:
            return _json_safe(value.to_dict())
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


def _load_ptmelt_hpo():
    try:
        from ptmelt.utils.hp_tuning import run_hyperparameter_tuning
    except Exception as exc:
        raise RuntimeError(
            "Could not import ptmelt.utils.hp_tuning.run_hyperparameter_tuning. "
            "Install a PT-MELT version containing the new HPO API."
        ) from exc
    return run_hyperparameter_tuning


def _build_ray_value(spec: Any) -> Any:
    if not isinstance(spec, Mapping):
        if isinstance(spec, list):
            from ray import tune

            return tune.choice(spec)
        return spec

    spec_type = str(spec.get("type", "value")).strip().lower()
    if spec_type in {"value", "constant", "fixed"}:
        return spec.get("value")

    # Grouped architecture specs are expanded in _build_search_space.
    if spec_type in {"independent_layer_choice", "layer_structure"}:
        return spec  # Return as-is; will be expanded in _build_search_space

    from ray import tune

    if spec_type == "choice":
        return tune.choice(spec.get("values", []))
    if spec_type == "uniform":
        return tune.uniform(float(spec["low"]), float(spec["high"]))
    if spec_type == "loguniform":
        include_zero, zero_probability = _resolve_zero_options(spec)
        if include_zero:
            return tune.sample_from(
                lambda _: _sample_log_with_optional_zero(
                    low=float(spec["low"]),
                    high=float(spec["high"]),
                    include_zero=True,
                    zero_probability=zero_probability,
                )
            )
        return tune.loguniform(float(spec["low"]), float(spec["high"]))
    if spec_type == "randint":
        return tune.randint(int(spec["low"]), int(spec["high"]))
    if spec_type == "quniform":
        return tune.quniform(float(spec["low"]), float(spec["high"]), float(spec["q"]))
    if spec_type == "qloguniform":
        include_zero, zero_probability = _resolve_zero_options(spec)
        if include_zero:
            return tune.sample_from(
                lambda _: _sample_log_with_optional_zero(
                    low=float(spec["low"]),
                    high=float(spec["high"]),
                    include_zero=True,
                    zero_probability=zero_probability,
                    q=float(spec["q"]),
                )
            )
        return tune.qloguniform(
            float(spec["low"]), float(spec["high"]), float(spec["q"])
        )

    raise ValueError(f"Unsupported search-space value type '{spec_type}'.")


def _expand_independent_layers(
    spec: Mapping[str, Any],
    grouping: str,
) -> Dict[str, Any]:
    """Expand independent_layer_choice spec into separate Ray tune dimensions per layer/pair."""
    max_depth = int(spec.get("max_depth", 1))
    values = spec.get("values", [])
    if max_depth < 1:
        raise ValueError("max_depth must be >= 1")
    if not values:
        raise ValueError("values list must not be empty")

    expanded = {"__max_depth": max_depth, "__grouping": grouping}

    from ray import tune

    if grouping == "single":
        # ANN/BNN: independent layer widths
        for i in range(max_depth):
            # First layer cannot be zero
            layer_values = [v for v in values if v != 0] if i == 0 else values
            if not layer_values:
                raise ValueError(
                    "At least one positive width is required for layer tuning"
                )
            expanded[f"layer_{i}_width"] = tune.choice(layer_values)
    elif grouping == "paired_equal":
        # RNN: paired layers with equal width
        for i in range(max_depth):
            pair_values = [v for v in values if v != 0] if i == 0 else values
            if not pair_values:
                raise ValueError(
                    "At least one positive width is required for pair tuning"
                )
            expanded[f"rnn_pair_{i}_width"] = tune.choice(pair_values)
    else:
        raise ValueError(f"Unsupported grouping: {grouping}")

    return expanded


def _expand_layer_structure(spec: Mapping[str, Any]) -> Dict[str, Any]:
    """Expand grouped architecture spec into Ray tune dimensions."""
    mode = str(spec.get("mode", "independent")).strip().lower()
    grouping = str(spec.get("grouping", "single")).strip().lower()
    max_depth = int(spec.get("max_depth", 1))
    widths = [int(v) for v in spec.get("widths", [])]

    if mode not in {"shared", "independent"}:
        raise ValueError("architecture mode must be 'shared' or 'independent'")
    if max_depth < 1:
        raise ValueError("max_depth must be >= 1")
    if not widths:
        raise ValueError("widths must contain at least one value")
    if any(v < 0 for v in widths):
        raise ValueError("all widths must be non-negative integers")
    if not any(v > 0 for v in widths):
        raise ValueError("at least one positive width is required")

    expanded: Dict[str, Any] = {
        "__architecture": True,
        "__arch_mode": mode,
        "__grouping": grouping,
        "__max_depth": max_depth,
    }

    from ray import tune

    if mode == "shared":
        positive_widths = [v for v in widths if v > 0]
        if not positive_widths:
            raise ValueError("shared mode requires positive widths")
        expanded["architecture_width"] = tune.choice(positive_widths)
        expanded["architecture_depth"] = tune.randint(1, max_depth + 1)
        return expanded

    # independent mode
    if grouping in {"single", "paired_equal"}:
        independent_spec = {
            "max_depth": max_depth,
            "values": widths,
            "grouping": grouping,
        }
        expanded.update(_expand_independent_layers(independent_spec, grouping))
        return expanded

    raise ValueError(f"Unsupported grouping: {grouping}")


def _normalize_sampled_architecture(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize grouped sampled architecture keys back to node_list."""
    result = dict(config)
    if not result.get("__architecture"):
        return result

    mode = str(result.pop("__arch_mode", "")).strip().lower()
    grouping = str(result.get("__grouping", "single")).strip().lower()

    # Remove the grouped architecture key from trainer-facing params.
    result.pop("architecture", None)
    result.pop("__architecture", None)
    # Grouped sampled architecture should override stale fixed architecture fields.
    result.pop("node_list", None)
    result.pop("width", None)
    result.pop("depth", None)

    if mode == "shared":
        width = result.pop("architecture_width", None)
        depth = result.pop("architecture_depth", None)
        result.pop("__grouping", None)
        result.pop("__max_depth", None)
        if width is not None and depth is not None:
            width_i = int(width)
            depth_i = int(depth)
            if width_i > 0 and depth_i > 0:
                result["node_list"] = [width_i] * depth_i
        return result

    if mode == "independent":
        normalized = _normalize_sampled_layers(result, grouping)
        normalized.pop("architecture", None)
        return normalized

    return result


def _normalize_sampled_layers(
    config: Mapping[str, Any],
    grouping: str,
) -> Dict[str, Any]:
    """Normalize sampled layer/pair keys back to node_list."""
    result = dict(config)
    max_depth = result.pop("__max_depth", None)
    result_grouping = result.pop("__grouping", None)

    if max_depth is None:
        return result

    if grouping == "single":
        # Extract layer widths and filter out zeros
        layers = []
        for i in range(max_depth):
            key = f"layer_{i}_width"
            width = result.pop(key, None)
            if width is not None and width != 0:
                layers.append(int(width))
        if layers:
            result["node_list"] = layers
    elif grouping == "paired_equal":
        # Extract pair widths and create paired node_list
        pairs = []
        pair_widths = []
        for i in range(max_depth):
            key = f"rnn_pair_{i}_width"
            width = result.pop(key, None)
            if width is not None and width != 0:
                pairs.append(int(width))
                pair_widths.append(int(width))
                pairs.append(int(width))  # Both members of pair have same width
        if pairs:
            result["node_list"] = pairs
            result["pair_widths"] = pair_widths  # For reference if needed

    return result


def _build_search_space(raw_search_space: Any) -> Dict[str, Any]:
    if raw_search_space in (None, ""):
        return {}
    if isinstance(raw_search_space, str):
        raw_search_space = json.loads(raw_search_space)
    if not isinstance(raw_search_space, Mapping):
        raise ValueError("search_space must be a JSON object.")

    search_space = {}
    for key, value in raw_search_space.items():
        key_str = str(key)

        if (
            key_str == "architecture"
            and isinstance(value, Mapping)
            and str(value.get("type", "")).strip().lower() == "layer_structure"
        ):
            expanded = _expand_layer_structure(value)
            for exp_key, exp_val in expanded.items():
                if not exp_key.startswith("__"):
                    search_space[exp_key] = _build_ray_value(exp_val)
                else:
                    search_space[exp_key] = exp_val
            continue

        # Check if this is an independent_layer_choice grouped spec
        if (
            isinstance(value, Mapping)
            and str(value.get("type", "")).strip().lower() == "independent_layer_choice"
        ):
            grouping = str(value.get("grouping", "single")).strip().lower()
            expanded = _expand_independent_layers(value, grouping)
            # Add all expanded dimensions to search space
            for exp_key, exp_val in expanded.items():
                if not exp_key.startswith("__"):
                    ray_val = _build_ray_value(exp_val)
                    search_space[exp_key] = ray_val
                else:
                    search_space[exp_key] = exp_val
        else:
            search_space[key_str] = _build_ray_value(value)

    return search_space


def _sample_log_with_optional_zero(
    *,
    low: float,
    high: float,
    include_zero: bool,
    zero_probability: float = 0.2,
    q: Optional[float] = None,
) -> float:
    # Keep logarithmic sampling strictly positive; zero is a separate discrete option.
    if include_zero and np.random.random() < float(zero_probability):
        return 0.0
    sampled = float(np.exp(np.random.uniform(np.log(low), np.log(high))))
    if q is not None:
        # q rounds sampled positive values in ordinary value space (not log-ratio space).
        quantized = round(sampled / q) * q
        sampled = float(min(max(quantized, low), high))
    return sampled


def _resolve_zero_options(spec: Mapping[str, Any]) -> tuple[bool, float]:
    include_zero = bool(spec.get("include_zero", False))
    if not include_zero:
        return False, 0.0
    raw_probability = spec.get("zero_probability", 0.2)
    return True, float(raw_probability)


def _normalize_log_spec_fields(spec: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = dict(spec)
    spec_type = str(normalized.get("type", "value")).strip().lower()
    if spec_type not in {"loguniform", "qloguniform"}:
        return normalized

    include_zero, zero_probability = _resolve_zero_options(normalized)
    normalized["include_zero"] = include_zero
    normalized["zero_probability"] = zero_probability if include_zero else 0.0
    return normalized


def _template_map_for_family(
    trainer_family: str, model_architecture: Any = None
) -> Dict[str, Mapping[str, Any]]:
    payload = _template_payload(trainer_family, model_architecture)
    return {str(template["key"]): template for template in payload["templates"]}


def _validate_single_spec(
    key: str, raw_spec: Any, template: Mapping[str, Any]
) -> Optional[str]:
    if not isinstance(raw_spec, Mapping):
        return None

    spec_type = str(raw_spec.get("type", "value")).strip().lower()

    # Handle grouped architecture validation.
    if spec_type == "layer_structure":
        if key != "architecture":
            return f"{key}: layer_structure is only supported for architecture"

        mode = str(raw_spec.get("mode", "")).strip().lower()
        if mode not in {"shared", "independent"}:
            return f"{key}: mode must be 'shared' or 'independent'"

        try:
            max_depth = int(raw_spec.get("max_depth", 0))
        except Exception:
            return f"{key}: max_depth must be an integer"
        if max_depth < 1:
            return f"{key}: max_depth must be >= 1"

        widths = raw_spec.get("widths")
        if not isinstance(widths, list) or not widths:
            return f"{key}: widths must be a non-empty array"
        try:
            int_widths = [int(v) for v in widths]
        except Exception:
            return f"{key}: all widths must be integers"
        if any(v < 0 for v in int_widths):
            return f"{key}: all widths must be non-negative integers"
        if not any(v > 0 for v in int_widths):
            return f"{key}: at least one positive width is required"

        grouping = str(raw_spec.get("grouping", "single")).strip().lower()
        if grouping not in {"single", "paired_equal"}:
            return f"{key}: grouping must be 'single' or 'paired_equal'"

        if mode == "shared" and any(v == 0 for v in int_widths):
            return f"{key}: shared mode cannot include zero in widths"

        if mode == "independent":
            zero_behavior = str(raw_spec.get("zero_behavior", "omit")).strip().lower()
            if zero_behavior != "omit":
                return f"{key}: zero_behavior must be 'omit' in independent mode"
        return None

    # Handle independent_layer_choice validation
    if spec_type == "independent_layer_choice":
        if key != "node_list":
            return f"{key}: independent_layer_choice is only supported for node_list"

        try:
            max_depth = int(raw_spec.get("max_depth", 0))
        except Exception:
            return f"{key}: max_depth must be an integer"
        if max_depth < 1:
            return f"{key}: max_depth must be >= 1"

        values = raw_spec.get("values")
        if not isinstance(values, list) or not values:
            return f"{key}: values must be a non-empty array"

        # All values must be non-negative integers
        try:
            int_values = [int(v) for v in values]
            if any(v < 0 for v in int_values):
                return f"{key}: all width values must be non-negative integers"
        except Exception:
            return f"{key}: all width values must be integers"

        # At least one positive width required
        if not any(v > 0 for v in int_values):
            return f"{key}: at least one positive width is required"

        grouping = str(raw_spec.get("grouping", "single")).strip().lower()
        if grouping not in {"single", "paired_equal"}:
            return f"{key}: grouping must be 'single' or 'paired_equal'"

        zero_behavior = str(raw_spec.get("zero_behavior", "omit")).strip().lower()
        if zero_behavior != "omit":
            return f"{key}: zero_behavior must be 'omit'"

        return None

    include_zero = raw_spec.get("include_zero", False)
    if "include_zero" in raw_spec and not isinstance(include_zero, bool):
        return f"{key}: include_zero must be boolean when provided"

    if "zero_probability" in raw_spec:
        try:
            zero_probability = float(raw_spec.get("zero_probability"))
        except Exception:
            return f"{key}: zero_probability must be numeric"
        if zero_probability < 0.0 or zero_probability > 1.0:
            return f"{key}: zero_probability must be between 0 and 1"

    allow_zero = bool(template.get("allow_zero", False))
    if include_zero:
        if spec_type not in {"loguniform", "qloguniform"}:
            return f"{key}: include_zero is only valid for logarithmic search types"
        if not allow_zero:
            return f"{key} does not permit zero in its logarithmic search space"

    if spec_type in {"loguniform", "qloguniform"}:
        try:
            low = float(raw_spec["low"])
            high = float(raw_spec["high"])
        except Exception:
            return f"{key}: low/high must be numeric"
        if low <= 0:
            return f"{key}: logarithmic low must be > 0"
        if high <= low:
            return f"{key}: high must be greater than low"

    return None


def _template_payload(
    trainer_family: str, model_architecture: Any = None
) -> Dict[str, Any]:
    family = _normalize_trainer_family(trainer_family)
    arch_type = _normalize_architecture(model_architecture, family)
    templates = list(_PARAMETER_TEMPLATES["common"]) + list(
        _PARAMETER_TEMPLATES.get(family, [])
    )
    if family == "static" and arch_type == "resnet":
        templates = [
            template
            for template in templates
            if template["key"] not in {"width", "node_list"}
        ] + [
            {
                "key": "node_list",
                "label": "Residual Node List",
                "category": "Architecture",
                "default": {"type": "choice", "values": [[2, 2], [4, 4], [8, 8]]},
                "supported_types": ["choice", "fixed"],
                "description": "ResNet block endpoints must preserve residual-add dimensions.",
            }
        ]
    if family == "vae":
        templates = [
            (
                {
                    **template,
                    "default": template.get("vae_default", template["default"]),
                    "min": max(int(template.get("min", 0)), 1),
                }
                if template["key"] == "num_mixtures"
                else template
            )
            for template in templates
        ]
    architecture_tuning = {
        "supported": False,
        "modes": [],
        "grouping": None,
        "depth_label": "Maximum depth",
        "default_max_depth": 4,
        "default_widths": [8, 16, 32, 64],
    }

    if family == "static" and arch_type in {"ann", "bnn"}:
        architecture_tuning = {
            "supported": True,
            "modes": ["shared", "independent"],
            "grouping": "single",
            "depth_label": "Maximum depth",
            "default_max_depth": 4,
            "default_widths": [0, 8, 16, 32, 64],
        }
    elif family == "temporal_rnn":
        architecture_tuning = {
            "supported": True,
            "modes": ["shared", "independent"],
            "grouping": "paired_equal",
            "depth_label": "Maximum pairs",
            "default_max_depth": 3,
            "default_widths": [0, 8, 16, 32, 64],
        }

    return {
        "trainer_family": family,
        "model_architecture": arch_type,
        "templates": templates,
        "default_search_space": {
            template["key"]: template["default"]
            for template in templates
            if template["key"] in {"learning_rate", "dropout", "num_mixtures"}
        },
        "categories": sorted({template["category"] for template in templates}),
        "architecture_tuning": architecture_tuning,
    }


def _validate_search_space_spec(
    trainer_family: str, raw_search_space: Any, model_architecture: Any = None
) -> Dict[str, Any]:
    family = _normalize_trainer_family(trainer_family)
    arch = _normalize_architecture(model_architecture, family)
    if isinstance(raw_search_space, str):
        raw_search_space = json.loads(raw_search_space)
    if not isinstance(raw_search_space, Mapping) or not raw_search_space:
        return {
            "ok": False,
            "errors": ["search_space must be a non-empty JSON object."],
            "warnings": [],
            "normalized": {},
        }

    allowed = HYPERPARAMETER_ALLOWLISTS.get(family, set())
    template_map = _template_map_for_family(family, model_architecture)
    errors = []
    warnings = []
    normalized = {}

    # Check for conflicting tuning specifications.
    has_grouped_architecture = False
    has_legacy_arch_tuning = False
    for raw_key, raw_spec in raw_search_space.items():
        key = HYPERPARAMETER_ALIASES.get(str(raw_key), str(raw_key))
        if (
            key == "architecture"
            and isinstance(raw_spec, Mapping)
            and str(raw_spec.get("type", "")).strip().lower() == "layer_structure"
        ):
            has_grouped_architecture = True
        if key in {"width", "depth", "node_list"}:
            has_legacy_arch_tuning = True

    if has_grouped_architecture and has_legacy_arch_tuning:
        return {
            "ok": False,
            "errors": [
                "Remove Width, Depth, and Node List tunables before enabling grouped architecture tuning."
            ],
            "warnings": [],
            "normalized": {},
        }

    for raw_key, raw_spec in raw_search_space.items():
        key = HYPERPARAMETER_ALIASES.get(str(raw_key), str(raw_key))
        if key not in allowed:
            warnings.append(
                f"'{raw_key}' is not tunable for trainer_family='{family}'."
            )
            continue
        spec_error = _validate_single_spec(key, raw_spec, template_map.get(key, {}))
        if spec_error:
            errors.append(spec_error)
            continue
        try:
            if (
                key == "architecture"
                and isinstance(raw_spec, Mapping)
                and str(raw_spec.get("type", "")).strip().lower() == "layer_structure"
            ):
                mode = str(raw_spec.get("mode", "")).strip().lower()
                grouping = str(raw_spec.get("grouping", "single")).strip().lower()
                if family == "static":
                    if arch not in {"ann", "bnn"}:
                        errors.append(
                            "architecture: grouped architecture tuning is only supported for ann/bnn in static trainer"
                        )
                        continue
                    if grouping != "single":
                        errors.append("architecture: ann/bnn require grouping='single'")
                        continue
                elif family == "temporal_rnn":
                    if grouping != "paired_equal":
                        errors.append(
                            "architecture: temporal_rnn requires grouping='paired_equal'"
                        )
                        continue
                else:
                    errors.append(
                        f"architecture: grouped architecture tuning is not supported for {family}"
                    )
                    continue

                if mode == "shared" and any(
                    int(v) == 0 for v in raw_spec.get("widths", [])
                ):
                    errors.append("architecture: shared mode widths cannot include 0")
                    continue

                normalized[key] = dict(raw_spec)
                continue

            if (
                isinstance(raw_spec, Mapping)
                and str(raw_spec.get("type", "")).strip().lower()
                == "independent_layer_choice"
            ):
                # Special handling for independent_layer_choice
                grouping = str(raw_spec.get("grouping", "single")).strip().lower()

                # Check architecture support
                if family == "static":
                    if arch not in {"ann", "bnn"} or grouping != "single":
                        errors.append(
                            f"{key}: independent layer choice requires ann or bnn with single grouping"
                        )
                        continue
                elif family == "temporal_rnn":
                    if arch != "rnn" or grouping != "paired_equal":
                        errors.append(f"{key}: RNN requires paired_equal grouping")
                        continue
                else:
                    errors.append(
                        f"{key}: independent layer choice not supported for {family}"
                    )
                    continue

                # Don't try to build ray values yet, just validate structure
                normalized_spec = dict(raw_spec)
                normalized[key] = normalized_spec
            else:
                normalized_spec = _normalize_log_spec_fields(raw_spec)
                _build_ray_value(normalized_spec)
                normalized[key] = normalized_spec
        except Exception as exc:
            errors.append(f"{raw_key}: {exc}")

    return {
        "ok": not errors and bool(normalized),
        "errors": errors,
        "warnings": warnings,
        "normalized": normalized,
    }


def _error_response(
    exc: Exception,
    *,
    status_code: int = 400,
    step: Optional[str] = None,
    debug: bool = False,
) -> JSONResponse:
    content = {
        "error": str(exc),
        "exc_type": type(exc).__name__,
    }
    if step:
        content["step"] = step
    if debug:
        content["traceback"] = traceback.format_exc()
    return JSONResponse(status_code=status_code, content=content)


def _to_float(body: Mapping[str, Any], key: str, default: float) -> float:
    return float(body.get(key, default))


def _to_int(body: Mapping[str, Any], key: str, default: int) -> int:
    return int(body.get(key, default))


def _build_base_config(
    body: Mapping[str, Any],
    trainer_family: str,
    arch_type: str,
    num_features: int,
    num_outputs: int,
) -> Dict[str, Any]:
    base_config: Dict[str, Any] = {
        "arch_type": arch_type,
        "num_features": int(num_features),
        "num_outputs": int(num_outputs),
        "epochs": _to_int(body, "max_epochs", _to_int(body, "num_epochs", 10)),
        "loss_fn": body.get("loss_function", body.get("loss_fn", "mse")),
        "optimizer": body.get("optimizer", "adam"),
        "learning_rate": _to_float(body, "learning_rate", 1e-3),
        "act_fun": body.get("activation_function", body.get("act_fun", "relu")),
        "output_activation": body.get("output_activation", "linear"),
        "dropout": _to_float(body, "dropout_rate", _to_float(body, "dropout", 0.0)),
        "batch_norm": bool(body.get("batch_norm", False)),
        "l1_reg": _to_float(body, "l1_reg", 0.0),
        "l2_reg": _to_float(body, "l2_reg", 0.0),
        "num_mixtures": _to_int(body, "num_mixtures", 0),
        "seed": body.get("random_state", body.get("seed")),
    }

    if trainer_family == "static":
        node_list = body.get("node_list")
        if node_list:
            base_config["node_list"] = node_list
        else:
            base_config["width"] = _to_int(body, "width", 32)
            base_config["depth"] = _to_int(body, "depth", 2)
        if arch_type == "resnet":
            base_config["layers_per_block"] = _to_int(body, "layers_per_block", 2)
        return base_config

    if trainer_family == "temporal_rnn":
        base_config.update(
            {
                "width": _to_int(body, "width", 32),
                "depth": _to_int(body, "depth", 2),
                "rnn_type": body.get("rnn_type", "lstm"),
                "head_type": body.get("head_type", "last"),
            }
        )
        return base_config

    if trainer_family == "temporal_transformer":
        base_config.update(
            {
                "width": _to_int(body, "width", 64),
                "depth": _to_int(body, "depth", 2),
                "head_type": body.get("head_type", "last"),
                "num_heads": _to_int(body, "num_heads", 4),
                "ff_dim": (
                    None
                    if _to_int(body, "ff_dim", 0) == 0
                    else _to_int(body, "ff_dim", 0)
                ),
                "max_seq_len": _to_int(body, "max_seq_len", 2048),
                "use_causal_mask": bool(body.get("use_causal_mask", False)),
            }
        )
        return base_config

    base_config.update(
        {
            "latent_dims": _to_int(body, "latent_dims", 8),
            "encoder_node_list": body.get("encoder_node_list", [32, 16]),
            "decoder_node_list": body.get("decoder_node_list", [16, 32]),
            "num_mixtures": max(_to_int(body, "num_mixtures", 1), 1),
        }
    )
    return base_config


def _trainer_hyperparameters_from_config(
    config: Mapping[str, Any], trainer_family: str
) -> Dict[str, Any]:
    values = dict(config)

    # Handle grouped architecture tuning first.
    values = _normalize_sampled_architecture(values)

    # Handle grouped independent layer/pair tuning
    grouping = values.get("__grouping")
    if grouping:
        values = _normalize_sampled_layers(values, grouping)

    # Legacy single-layer expansion (backward compat for max_depth without __grouping)
    max_depth = values.get("max_depth")
    if max_depth is not None and "node_list" not in values:
        node_list = []
        for layer_index in range(int(max_depth)):
            width = int(values.get(f"layer_{layer_index}_width", 0) or 0)
            if width > 0:
                node_list.append(width)
        if node_list:
            values["node_list"] = node_list

    allowed = HYPERPARAMETER_ALLOWLISTS.get(trainer_family, set())
    trainer_params = {}
    for key, value in values.items():
        normalized_key = HYPERPARAMETER_ALIASES.get(str(key), str(key))
        if normalized_key in allowed:
            trainer_params[normalized_key] = value
    return trainer_params


_EMITTED_HYPERPARAMETER_EXCLUDED_KEYS = {
    "epochs",
    "max_epochs",
    "num_epochs",
}


def _strip_emitted_epoch_hyperparameters(
    values: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    if not isinstance(values, Mapping):
        return {}

    stripped: Dict[str, Any] = {}
    for key, value in values.items():
        normalized_key = HYPERPARAMETER_ALIASES.get(str(key), str(key))
        if normalized_key in _EMITTED_HYPERPARAMETER_EXCLUDED_KEYS:
            continue
        if str(key) in _EMITTED_HYPERPARAMETER_EXCLUDED_KEYS:
            continue
        stripped[str(key)] = value
    return stripped


def _prepare_static_data(body: Mapping[str, Any]):
    x = np.asarray(body.get("x"))
    y_raw = body.get("y")
    if y_raw is None:
        raise ValueError("Missing y input for supervised HPO.")
    y = np.asarray(y_raw)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    splits = split_train_val_test(
        x=x,
        y=y,
        val_size=_to_float(body, "val_size", 0.1),
        test_size=_to_float(body, "test_size", 0.1),
        random_state=_to_int(body, "random_state", 42),
    )
    return x, y, splits


def _prepare_temporal_data(body: Mapping[str, Any]):
    x_raw = body.get("x") if body.get("x") is not None else body.get("x_seq")
    y_raw = body.get("y") if body.get("y") is not None else body.get("y_seq")
    if x_raw is None or y_raw is None:
        raise ValueError("Missing temporal x/y inputs for HPO.")

    x = np.asarray(x_raw)
    y = np.asarray(y_raw)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    if x.shape[0] != y.shape[0]:
        raise ValueError("x and y must have matching sample counts.")

    seq_length = _to_int(body, "seq_length", 60)
    seq_to_one = bool(body.get("seq_to_one", True))
    if seq_length < 1:
        raise ValueError("seq_length must be >= 1.")

    if x.ndim == 2:
        if not seq_to_one:
            raise ValueError("Raw 2D temporal HPO currently requires seq_to_one=true.")
        raw_splits = split_train_val_test_temporal(
            x=x,
            y=y,
            val_size=_to_float(body, "val_size", 0.1),
            test_size=_to_float(body, "test_size", 0.1),
        )
        x_train_raw, x_val_raw, x_test_raw, y_train_raw, y_val_raw, y_test_raw = (
            raw_splits
        )
        for name, split in {
            "train": x_train_raw,
            "val": x_val_raw,
            "test": x_test_raw,
        }.items():
            if split.shape[0] < seq_length:
                raise ValueError(
                    f"Not enough {name} samples for seq_length={seq_length}."
                )

        scaled_raw = scale_splits(
            x_train=x_train_raw,
            x_val=x_val_raw,
            x_test=x_test_raw,
            y_train=y_train_raw,
            y_val=y_val_raw,
            y_test=y_test_raw,
            normalizer_type=body.get("norm_type", "none"),
        )
        (
            x_train_raw_scaled,
            x_val_raw_scaled,
            x_test_raw_scaled,
            y_train_raw_scaled,
            y_val_raw_scaled,
            y_test_raw_scaled,
            x_normalizer,
            y_normalizer,
        ) = scaled_raw

        x_train, y_train = prepare_sequences_from_raw(
            x_train_raw, y_train_raw, seq_length, seq_to_one
        )
        x_val, y_val = prepare_sequences_from_raw(
            x_val_raw, y_val_raw, seq_length, seq_to_one
        )
        x_test, y_test = prepare_sequences_from_raw(
            x_test_raw, y_test_raw, seq_length, seq_to_one
        )
        x_train_scaled, y_train_scaled = prepare_sequences_from_raw(
            x_train_raw_scaled, y_train_raw_scaled, seq_length, seq_to_one
        )
        x_val_scaled, y_val_scaled = prepare_sequences_from_raw(
            x_val_raw_scaled, y_val_raw_scaled, seq_length, seq_to_one
        )
        x_test_scaled, y_test_scaled = prepare_sequences_from_raw(
            x_test_raw_scaled, y_test_raw_scaled, seq_length, seq_to_one
        )
        return (
            (x_train, x_val, x_test, y_train, y_val, y_test),
            (
                x_train_scaled,
                x_val_scaled,
                x_test_scaled,
                y_train_scaled,
                y_val_scaled,
                y_test_scaled,
                x_normalizer,
                y_normalizer,
            ),
        )

    if x.ndim != 3:
        raise ValueError("Temporal HPO expects x to be 2D raw or 3D windowed data.")
    if x.shape[1] != seq_length:
        raise ValueError(
            f"Input sequence length mismatch: got {x.shape[1]}, expected {seq_length}."
        )

    x_train, x_val, x_test, y_train, y_val, y_test = split_train_val_test_temporal(
        x=x,
        y=y,
        val_size=_to_float(body, "val_size", 0.1),
        test_size=_to_float(body, "test_size", 0.1),
    )
    boundary_gap = max(0, seq_length - 1)
    if boundary_gap > 0:
        if x_val.shape[0] <= boundary_gap or x_test.shape[0] <= boundary_gap:
            raise ValueError(
                "Not enough pre-windowed sequences after applying temporal boundary gap."
            )
        x_val = x_val[boundary_gap:]
        y_val = y_val[boundary_gap:]
        x_test = x_test[boundary_gap:]
        y_test = y_test[boundary_gap:]

    scaled = scale_splits(
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        normalizer_type=body.get("norm_type", "none"),
    )
    return ((x_train, x_val, x_test, y_train, y_val, y_test), scaled)


def _make_hpo_dataloaders(
    body: Mapping[str, Any], trainer_family: str, run_id: str, app
):
    if trainer_family == "vae":
        x = np.asarray(body.get("x"))
        if x.ndim < 2:
            raise ValueError("VAE HPO input x must be at least 2D.")
        y = x.copy()
        splits = split_train_val_test(
            x=x,
            y=y,
            val_size=_to_float(body, "val_size", 0.1),
            test_size=_to_float(body, "test_size", 0.1),
            random_state=_to_int(body, "random_state", 42),
        )
        x_train, x_val, x_test, y_train, y_val, y_test = splits
        scaled = scale_splits(
            x_train=x_train,
            x_val=x_val,
            x_test=x_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            normalizer_type=body.get("norm_type", "none"),
        )
    elif trainer_family == "static":
        _, _, splits = _prepare_static_data(body)
        x_train, x_val, x_test, y_train, y_val, y_test = splits
        scaled = scale_splits(
            x_train=x_train,
            x_val=x_val,
            x_test=x_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            normalizer_type=body.get("norm_type", "none"),
        )
    else:
        splits, scaled = _prepare_temporal_data(body)
        x_train, x_val, x_test, y_train, y_val, y_test = splits

    (
        x_train_scaled,
        x_val_scaled,
        _x_test_scaled,
        y_train_scaled,
        y_val_scaled,
        _y_test_scaled,
        _x_normalizer,
        _y_normalizer,
    ) = scaled
    train_dl, val_dl = make_dataloaders(
        x_train_scaled=x_train_scaled,
        y_train_scaled=y_train_scaled,
        x_val_scaled=x_val_scaled,
        y_val_scaled=y_val_scaled,
        batch_size=_to_int(body, "batch_size", 32),
        shuffle=bool(body.get("shuffle", True)),
    )
    return train_dl, val_dl, (x_train_scaled, y_train_scaled)


def _default_search_space(trainer_family: str = "static") -> Dict[str, Any]:
    search_space = {
        "learning_rate": {"type": "choice", "values": [1e-3, 3e-4, 1e-4]},
        "dropout": {"type": "choice", "values": [0.0, 0.1, 0.2]},
        "num_mixtures": {"type": "choice", "values": [0, 1, 3]},
    }
    if trainer_family == "vae":
        search_space["num_mixtures"] = {"type": "choice", "values": [1, 3, 5]}
    return search_space


def _artifact_path(tuning_id: str) -> Path:
    return TUNING_RESULTS_DIR / f"{tuning_id}.json"


def _write_artifact(tuning_id: str, payload: Mapping[str, Any]) -> Path:
    TUNING_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = _artifact_path(tuning_id)
    path.write_text(json.dumps(_json_safe(payload), indent=2), encoding="utf-8")
    return path


@router.post("/melt_hyperparameter_tuner")
async def melt_hyperparameter_tuner(request: Request):
    try:
        body = await request.json()
    except Exception as exc:
        return _error_response(exc, step="parse_request_json", debug=True)

    debug_errors = bool(body.get("debug_errors", False))
    run_id = register_training_run(
        request.app,
        str(body.get("training_run_id") or "").strip() or None,
    )

    try:
        step = "load_ptmelt_hpo"
        run_hyperparameter_tuning = _load_ptmelt_hpo()
        step = "normalize_trainer_family"
        trainer_family = _normalize_trainer_family(body.get("trainer_family"))
        step = "normalize_architecture"
        arch_type = _normalize_architecture(
            body.get("model_architecture", body.get("arch_type")), trainer_family
        )
        step = "prepare_dataloaders"
        train_dl, val_dl, shape_data = _make_hpo_dataloaders(
            body, trainer_family, run_id, request.app
        )
        x_train_scaled, y_train_scaled = shape_data
        num_features = x_train_scaled.shape[-1]
        num_outputs = y_train_scaled.shape[1] if y_train_scaled.ndim > 1 else 1
        base_config = _build_base_config(
            body=body,
            trainer_family=trainer_family,
            arch_type=arch_type,
            num_features=num_features,
            num_outputs=num_outputs,
        )
        search_space_spec = body.get(
            "search_space", _default_search_space(trainer_family)
        )
        step = "validate_search_space"
        validation = _validate_search_space_spec(
            trainer_family,
            search_space_spec,
            body.get("model_architecture", body.get("arch_type")),
        )
        if not validation.get("ok", False):
            raise ValueError(
                "; ".join(validation.get("errors") or ["Invalid search_space"])
            )
        search_space_spec = validation.get("normalized", search_space_spec)
        step = "build_search_space"
        search_space = _build_search_space(search_space_spec)
        step_kwargs = dict(body.get("step_kwargs") or {})
        if trainer_family == "temporal_rnn":
            step_kwargs.setdefault("suffix_crop", bool(body.get("suffix_crop", False)))
            step_kwargs.setdefault(
                "min_length", _to_int(body, "suffix_crop_min_length", 32)
            )

        step = "run_hyperparameter_tuning"
        result = await asyncio.to_thread(
            run_hyperparameter_tuning,
            train_dl=train_dl,
            val_dl=val_dl,
            search_space=search_space,
            base_config=base_config,
            metric=body.get("metric", "val_loss"),
            mode=body.get("mode", "min"),
            num_samples=_to_int(body, "num_samples", 10),
            resources=body.get("resources"),
            scheduler=body.get("scheduler", "asha"),
            search_alg=body.get("search_alg", "optuna"),
            ray_init_kwargs=body.get("ray_init_kwargs"),
            storage_path=body.get("storage_path"),
            name=body.get("name"),
            max_concurrent=body.get("max_concurrent"),
            checkpoint_interval=_to_int(body, "checkpoint_interval", 1),
            scheduler_kwargs=body.get("scheduler_kwargs"),
            search_alg_kwargs=body.get("search_alg_kwargs"),
            tune_config_kwargs=body.get("tune_config_kwargs"),
            run_config_kwargs=body.get("run_config_kwargs"),
            device=body.get("device"),
            step_kwargs=step_kwargs,
            return_raw_results=False,
        )

        step = "serialize_result"
        tuning_id = str(body.get("tuning_id") or uuid.uuid4().hex)
        emitted_best_hyperparameters = _strip_emitted_epoch_hyperparameters(
            result.best_hyperparameters
        )
        emitted_trainer_hyperparameters = _strip_emitted_epoch_hyperparameters(
            _trainer_hyperparameters_from_config(result.best_config, trainer_family)
        )
        response_payload = {
            "tuning_id": tuning_id,
            "trainer_family": trainer_family,
            "best_hyperparameters": emitted_best_hyperparameters,
            "trainer_hyperparameters": emitted_trainer_hyperparameters,
            "best_config": result.best_config,
            "metric_details": result.metric_details,
            "trial_history": result.trial_history,
            "search_space": search_space_spec,
            "base_config": base_config,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        if bool(body.get("autosave", True)):
            path = _write_artifact(tuning_id, response_payload)
            response_payload["artifact_path"] = str(path)

        return JSONResponse(content=_json_safe(response_payload))
    except TrainingCancelledError:
        return JSONResponse(
            status_code=200,
            content={
                "cancelled": True,
                "run_id": run_id,
                "error": "Hyperparameter tuning cancelled by user.",
            },
        )
    except Exception as exc:
        return _error_response(exc, step=locals().get("step"), debug=debug_errors)
    finally:
        clear_training_run(request.app, run_id)


@router.post("/melt_hpo_parameter_templates")
async def melt_hpo_parameter_templates(request: Request):
    body = await request.json()
    try:
        payload = _template_payload(
            body.get("trainer_family", "static"),
            body.get("model_architecture", body.get("arch_type")),
        )
        return JSONResponse(content=_json_safe(payload))
    except Exception as exc:
        return _error_response(exc, step="build_parameter_templates")


@router.post("/validate_hpo_search_space")
async def validate_hpo_search_space(request: Request):
    body = await request.json()
    try:
        payload = _validate_search_space_spec(
            body.get("trainer_family", "static"),
            body.get("search_space", {}),
            body.get("model_architecture", body.get("arch_type")),
        )
        return JSONResponse(content=_json_safe(payload))
    except Exception as exc:
        return _error_response(exc, step="validate_search_space")


@router.post("/save_hyperparameters")
async def save_hyperparameters(request: Request):
    body = await request.json()
    tuning_id = str(body.get("tuning_id") or uuid.uuid4().hex)
    payload = body.get("tuning_result") or body.get("hyperparameters") or body
    path = _write_artifact(
        tuning_id,
        {
            "tuning_id": tuning_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "payload": payload,
        },
    )
    return JSONResponse(content={"tuning_id": tuning_id, "artifact_path": str(path)})


@router.post("/load_hyperparameters")
async def load_hyperparameters(request: Request):
    body = await request.json()
    path_raw = body.get("path") or body.get("artifact_path")
    if not path_raw:
        tuning_id = str(body.get("tuning_id") or "").strip()
        if not tuning_id:
            return JSONResponse(
                status_code=400,
                content={"error": "Provide path/artifact_path or tuning_id."},
            )
        path = _artifact_path(tuning_id)
    else:
        path = Path(str(path_raw)).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()

    if not path.exists() or not path.is_file():
        return JSONResponse(
            status_code=404, content={"error": f"File not found: {path}"}
        )

    payload = json.loads(path.read_text(encoding="utf-8"))
    best_hyperparameters = payload.get("best_hyperparameters")
    if best_hyperparameters is None and isinstance(payload.get("payload"), Mapping):
        best_hyperparameters = payload["payload"].get("best_hyperparameters")
    best_hyperparameters = _strip_emitted_epoch_hyperparameters(best_hyperparameters)
    return JSONResponse(
        content={
            "tuning_result": payload,
            "best_hyperparameters": best_hyperparameters or {},
            "artifact_path": str(path),
        }
    )
