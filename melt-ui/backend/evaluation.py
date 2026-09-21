import inspect
import json
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from ptmelt.utils.evaluation import ensemble_predictions, make_predictions
from ptmelt.utils.statistics import compute_metrics, compute_rmse, compute_rsquared
from ptmelt.utils.visualization import (
    point_cloud_plot,
    point_cloud_plot_with_uncertainty,
)

from .utils import convert_fig_to_image, scaler_from_dict

router = APIRouter()


def _default_ensemble_size() -> int:
    try:
        default = inspect.signature(ensemble_predictions).parameters["n_iter"].default
        value = int(default)
        return value if value >= 2 else 100
    except Exception:
        return 100


def _slice_output(arr: np.ndarray, output_index: int) -> np.ndarray:
    if arr.ndim == 1:
        if output_index not in (0, -1):
            raise ValueError("output_index out of range for 1D outputs")
        return arr
    if output_index < 0 or output_index >= arr.shape[1]:
        raise ValueError(
            f"output_index out of range. Got {output_index}, "
            f"valid range is [0, {arr.shape[1] - 1}]"
        )
    return arr[:, output_index]


def _validate_uq_shapes(mean: np.ndarray, std: np.ndarray, truth: np.ndarray) -> None:
    if mean.shape != truth.shape:
        raise ValueError(
            f"UQ mean/truth shape mismatch: mean {mean.shape}, truth {truth.shape}"
        )
    if std.shape != mean.shape:
        raise ValueError(
            f"UQ std/mean shape mismatch: std {std.shape}, mean {mean.shape}"
        )
    if not np.all(np.isfinite(std)):
        raise ValueError("UQ standard deviations contain non-finite values")
    if np.any(std < 0):
        raise ValueError("UQ standard deviations must be non-negative")


def _native_uq_predict(model, x_split):
    pred = make_predictions(
        model,
        x_split,
        y_normalizer=None,
        unnormalize=False,
        training=False,
    )
    if isinstance(pred, tuple) and len(pred) >= 2 and pred[1] is not None:
        return np.asarray(pred[0]), np.asarray(pred[1])
    raise ValueError("This model does not provide built-in uncertainty estimates.")


def _ensemble_uq_predict(model, x_split, ensemble_size):
    # training=True keeps dropout layers active during inference (Monte Carlo Dropout).
    # This is required for epistemic uncertainty estimation via stochastic forward passes.
    mean, std = ensemble_predictions(
        model,
        x_split,
        y_normalizer=None,
        unnormalize=False,
        n_iter=int(ensemble_size),
        training=True,
    )
    mean = np.asarray(mean)
    std = np.asarray(std)
    if np.allclose(std, 0.0):
        raise ValueError("This model does not support stochastic ensemble prediction.")
    return mean, std


def _total_uq_predict(model, x_split, ensemble_size):
    aleatoric_mean, aleatoric_std = _native_uq_predict(model, x_split)
    epistemic_mean, epistemic_std = _ensemble_uq_predict(model, x_split, ensemble_size)

    if aleatoric_std.shape != epistemic_std.shape:
        raise ValueError(
            "Aleatoric and epistemic uncertainties are incompatible for total uncertainty."
        )
    if aleatoric_mean.shape != epistemic_mean.shape:
        raise ValueError(
            "Aleatoric and epistemic prediction means are incompatible for total uncertainty."
        )

    total_std = np.sqrt(np.square(aleatoric_std) + np.square(epistemic_std))
    # Use epistemic mean so the point estimate is consistent with the stochastic pass.
    return epistemic_mean, total_std


def _parse_bool_or_none(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if v in {"0", "false", "f", "no", "n", "off"}:
            return False
    return None


def _parse_split_triplet(data, field_name: str):
    parsed = data
    if isinstance(data, str):
        try:
            parsed = json.loads(data)
        except Exception:
            raise ValueError(f"Invalid {field_name} JSON.")

    if not isinstance(parsed, (list, tuple)) or len(parsed) != 3:
        raise ValueError(
            f"Invalid {field_name} format. Expected train/val/test arrays."
        )

    return [np.asarray(arr) for arr in parsed]


def _deserialize_normalizer(normalizer_payload, stored_normalizer=None):
    if hasattr(normalizer_payload, "inverse_transform"):
        return normalizer_payload
    if isinstance(normalizer_payload, dict) and normalizer_payload:
        return scaler_from_dict(normalizer_payload)
    if hasattr(stored_normalizer, "inverse_transform"):
        return stored_normalizer
    if isinstance(stored_normalizer, dict) and stored_normalizer:
        return scaler_from_dict(stored_normalizer)
    return None


def _transform_x_split_for_model(x_split: np.ndarray, x_normalizer):
    if x_normalizer is None:
        return x_split
    if x_split.ndim == 2:
        return x_normalizer.transform(x_split)
    if x_split.ndim == 3:
        n, t, f = x_split.shape
        x_flat = x_split.reshape(-1, f)
        x_flat_scaled = x_normalizer.transform(x_flat)
        return x_flat_scaled.reshape(n, t, f)
    return x_normalizer.transform(x_split)


def _inverse_y_preserve_shape(values: np.ndarray, y_normalizer):
    if y_normalizer is None:
        return values
    arr = np.asarray(values)
    if arr.ndim == 1:
        inv = y_normalizer.inverse_transform(arr.reshape(-1, 1))
        return inv.reshape(-1)
    return y_normalizer.inverse_transform(arr)


def _inverse_std_preserve_shape(std_values: np.ndarray, y_normalizer):
    if y_normalizer is None:
        return std_values
    std_arr = np.asarray(std_values)
    zeros = np.zeros_like(std_arr)
    try:
        inv_std = np.abs(
            _inverse_y_preserve_shape(std_arr, y_normalizer)
            - _inverse_y_preserve_shape(zeros, y_normalizer)
        )
        return inv_std
    except Exception:
        return std_arr


@router.post("/evaluate_supervised_model")
async def evaluate_supervised_model(request: Request):
    """
    Make predictions for a supervised model using provided input data.
    """
    body = await request.json()
    print(f"Received evaluation request with body keys: {list(body.keys())}")

    model_dict = body.get("model")
    # x_data = np.asarray(body.get("x_data"))
    # y_data = np.asarray(body.get("y_data"))
    x_data = body.get("x_data")
    y_data = body.get("y_data")
    x_normalizer_payload = body.get("x_normalizer", None)
    y_normalizer = body.get("y_normalizer", None)
    unnormalize = bool(body.get("unnormalize", True))
    x_data_is_scaled_input = _parse_bool_or_none(body.get("x_data_is_scaled_input"))
    y_data_is_scaled_input = _parse_bool_or_none(body.get("y_data_is_scaled_input"))
    evaluation_mode = str(body.get("evaluation_mode", "deterministic")).strip().lower()
    ensemble_size = int(body.get("ensemble_size", _default_ensemble_size()))
    output_index_raw = body.get("output_index", None)

    if evaluation_mode not in {
        "deterministic",
        "aleatoric",
        "epistemic",
        "total",
    }:
        return JSONResponse(
            status_code=400,
            content={
                "error": "evaluation_mode must be one of: deterministic, aleatoric, epistemic, total"
            },
        )

    if output_index_raw is not None:
        try:
            output_index = int(output_index_raw)
        except Exception:
            return JSONResponse(
                status_code=400,
                content={"error": "output_index must be an integer when provided."},
            )
    else:
        output_index = None

    if ensemble_size < 2:
        return JSONResponse(
            status_code=400,
            content={"error": "ensemble_size must be >= 2."},
        )

    # upack the model which is actually model_id and model_meta dict
    if isinstance(model_dict, dict) and "model_id" in model_dict:
        model_id = model_dict["model_id"]
        print(f"Model ID: {model_id}")
        if not model_id:
            return JSONResponse(
                status_code=400,
                content={"error": "model.model_id is missing or null."},
            )
        model_store = getattr(request.app.state, "model_store", {})
        model_dict = model_store.get(model_id)
        if inspect.isawaitable(model_dict):
            model_dict = await model_dict
        if model_dict is None:
            return JSONResponse(
                status_code=404,
                content={"error": f"Model with id {model_id} not found."},
            )
    else:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid model format. Expected a dict with 'model_id'."},
        )

    try:
        x_train, x_val, x_test = _parse_split_triplet(x_data, "x_data")
        y_train, y_val, y_test = _parse_split_triplet(y_data, "y_data")
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})

    # get the model from the model_dict
    model = model_dict["model"]
    model_metadata = model_dict.get("model_meta", {}) or {}
    stored_x_normalizer = model_dict.get("x_normalizer")
    stored_y_normalizer = model_dict.get("y_normalizer")

    x_normalizer = _deserialize_normalizer(x_normalizer_payload, stored_x_normalizer)
    y_normalizer = _deserialize_normalizer(y_normalizer, stored_y_normalizer)

    x_data_is_scaled = (
        x_data_is_scaled_input
        if x_data_is_scaled_input is not None
        else bool(model_metadata.get("x_data_is_scaled", False))
    )
    y_data_is_scaled = (
        y_data_is_scaled_input
        if y_data_is_scaled_input is not None
        else bool(model_metadata.get("y_data_is_scaled", False))
    )

    if str(model_metadata.get("architecture", "")).lower() == "vae":
        return JSONResponse(
            status_code=400,
            content={
                "error": "VAE models are self-supervised and are not supported by /evaluate_supervised_model."
            },
        )

    print(f"unpack x_data shapes: {x_train.shape}, {x_val.shape}, {x_test.shape}")
    print(f"unpack y_data shapes: {y_train.shape}, {y_val.shape}, {y_test.shape}")

    print(f"model: {model}")

    if x_normalizer is not None and not x_data_is_scaled:
        x_train_model = _transform_x_split_for_model(x_train, x_normalizer)
        x_val_model = _transform_x_split_for_model(x_val, x_normalizer)
        x_test_model = _transform_x_split_for_model(x_test, x_normalizer)
    else:
        x_train_model, x_val_model, x_test_model = x_train, x_val, x_test

    splits = {
        "train": (x_train_model, y_train),
        "validation": (x_val_model, y_val),
        "test": (x_test_model, y_test),
    }

    prediction_bundle = {}
    uq_method_used = None

    if evaluation_mode == "deterministic":
        for split_name, (x_split, y_split) in splits.items():
            pred = make_predictions(
                model,
                x_split,
                y_normalizer=None,
                unnormalize=False,
                training=False,
            )
            pred_mean = np.asarray(pred[0] if isinstance(pred, tuple) else pred)
            prediction_bundle[split_name] = {
                "mean": pred_mean,
                "std": None,
                "truth": np.asarray(y_split),
            }
    else:

        def compute_aleatoric_all():
            computed = {}
            for split_name, (x_split, y_split) in splits.items():
                mean, std = _native_uq_predict(model, x_split)
                computed[split_name] = {
                    "mean": mean,
                    "std": std,
                    "truth": np.asarray(y_split),
                }
            return computed

        def compute_epistemic_all():
            computed = {}
            for split_name, (x_split, y_split) in splits.items():
                mean, std = _ensemble_uq_predict(model, x_split, ensemble_size)
                computed[split_name] = {
                    "mean": mean,
                    "std": std,
                    "truth": np.asarray(y_split),
                }
            return computed

        def compute_total_all():
            computed = {}
            for split_name, (x_split, y_split) in splits.items():
                mean, std = _total_uq_predict(model, x_split, ensemble_size)
                computed[split_name] = {
                    "mean": mean,
                    "std": std,
                    "truth": np.asarray(y_split),
                }
            return computed

        if evaluation_mode == "aleatoric":
            try:
                prediction_bundle = compute_aleatoric_all()
                uq_method_used = "aleatoric"
            except ValueError as exc:
                return JSONResponse(status_code=400, content={"error": str(exc)})
        elif evaluation_mode == "epistemic":
            try:
                prediction_bundle = compute_epistemic_all()
                uq_method_used = "epistemic"
            except ValueError as exc:
                return JSONResponse(status_code=400, content={"error": str(exc)})
        else:  # total
            try:
                prediction_bundle = compute_total_all()
                uq_method_used = "total"
            except ValueError as exc:
                return JSONResponse(status_code=400, content={"error": str(exc)})

    max_targets = (
        prediction_bundle["train"]["mean"].shape[1]
        if prediction_bundle["train"]["mean"].ndim > 1
        else 1
    )

    if output_index is None:
        output_indices = list(range(max_targets))
    else:
        if output_index < 0 or output_index >= max_targets:
            return JSONResponse(
                status_code=400,
                content={
                    "error": (
                        f"output_index out of range. Got {output_index}, "
                        f"valid range is [0, {max_targets - 1}]"
                    )
                },
            )
        output_indices = [output_index]

    print(f"Finished predictions, preparing plots...")

    # Create a 1x3 subplot for training, validation, and test data
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Define markers and colors for the point cloud plot
    markers = ["o", "s", "D", "^", "v", "<", ">", "p", "*", "h"]
    colors = plt.cm.tab10.colors

    # Define text positions for the metrics text annotation
    text_positions = [(0.3, i * 0.05 + 0.01) for i in range(len(output_indices))]

    if unnormalize and y_normalizer is not None and y_data_is_scaled:
        y_train_real = _inverse_y_preserve_shape(y_train, y_normalizer)
        y_val_real = _inverse_y_preserve_shape(y_val, y_normalizer)
        y_test_real = _inverse_y_preserve_shape(y_test, y_normalizer)
    else:
        y_train_real = y_train
        y_val_real = y_val
        y_test_real = y_test

    if evaluation_mode == "deterministic":
        pred_train = prediction_bundle["train"]["mean"]
        pred_val = prediction_bundle["validation"]["mean"]
        pred_test = prediction_bundle["test"]["mean"]

        if unnormalize and y_normalizer is not None:
            pred_train = _inverse_y_preserve_shape(pred_train, y_normalizer)
            pred_val = _inverse_y_preserve_shape(pred_val, y_normalizer)
            pred_test = _inverse_y_preserve_shape(pred_test, y_normalizer)

        # Plot predictions for each output index
        for i, idx in enumerate(output_indices):
            # Compute R-squared and RMSE for each dataset
            r_sq_train = compute_rsquared(y_train_real[:, idx], pred_train[:, idx])
            rmse_train = compute_rmse(y_train_real[:, idx], pred_train[:, idx])
            r_sq_val = compute_rsquared(y_val_real[:, idx], pred_val[:, idx])
            rmse_val = compute_rmse(y_val_real[:, idx], pred_val[:, idx])
            r_sq_test = compute_rsquared(y_test_real[:, idx], pred_test[:, idx])
            rmse_test = compute_rmse(y_test_real[:, idx], pred_test[:, idx])

            # Create point cloud plot for each dataset
            point_cloud_plot(
                axes[0],
                y_train_real[:, idx],
                pred_train[:, idx],
                r_sq_train,
                rmse_train,
                f"Output {idx}",
                markers[i % len(markers)],
                colors[i % len(colors)],
                text_pos=text_positions[i % len(text_positions)],
            )
            point_cloud_plot(
                axes[1],
                y_val_real[:, idx],
                pred_val[:, idx],
                r_sq_val,
                rmse_val,
                f"Output {idx}",
                markers[i % len(markers)],
                colors[i % len(colors)],
                text_pos=text_positions[i % len(text_positions)],
            )
            point_cloud_plot(
                axes[2],
                y_test_real[:, idx],
                pred_test[:, idx],
                r_sq_test,
                rmse_test,
                f"Output {idx}",
                markers[i % len(markers)],
                colors[i % len(colors)],
                text_pos=text_positions[i % len(text_positions)],
            )
    else:
        # Use native point-cloud UQ plotting helper directly on endpoint axes.
        idx = output_indices[0]
        split_plot_specs = [
            (axes[0], prediction_bundle["train"], y_train_real),
            (axes[1], prediction_bundle["validation"], y_val_real),
            (axes[2], prediction_bundle["test"], y_test_real),
        ]

        for ax, split_bundle, y_real in split_plot_specs:
            mean_arr = split_bundle["mean"]
            std_arr = split_bundle["std"]
            if unnormalize and y_normalizer is not None:
                mean_arr = _inverse_y_preserve_shape(mean_arr, y_normalizer)
                std_arr = _inverse_std_preserve_shape(std_arr, y_normalizer)

            mean = _slice_output(mean_arr, idx)
            std = _slice_output(std_arr, idx)
            truth = _slice_output(y_real, idx)
            _validate_uq_shapes(mean, std, truth)
            point_cloud_plot_with_uncertainty(
                ax,
                truth,
                mean,
                std,
                metrics_to_display=None,
            )

    # Set plot titles
    axes[0].set_title("Training Data")
    axes[1].set_title("Validation Data")
    axes[2].set_title("Test Data")

    fig.suptitle("Predictions")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    print(f"Plots prepared, converting to images...")

    # Convert figure to image data URIs and return
    response = convert_fig_to_image(fig, image_format="pdf")
    if evaluation_mode != "deterministic":
        response["uq"] = {
            "requested_mode": evaluation_mode,
            "method_used": uq_method_used,
            "ensemble_size_used": (
                ensemble_size if uq_method_used in {"epistemic", "total"} else None
            ),
        }
    return response


@router.post("/evaluate_temporal_supervised_model")
async def evaluate_temporal_supervised_model(request: Request):
    """
    Make notebook-style temporal evaluation plots for sequence models.
    """
    body = await request.json()

    model_dict = body.get("model")
    x_data = body.get("x_data")
    y_data = body.get("y_data")
    x_normalizer_payload = body.get("x_normalizer", None)
    y_normalizer_payload = body.get("y_normalizer", None)
    unnormalize = bool(body.get("unnormalize", True))
    x_data_is_scaled_input = _parse_bool_or_none(body.get("x_data_is_scaled_input"))
    y_data_is_scaled_input = _parse_bool_or_none(body.get("y_data_is_scaled_input"))
    output_index = int(body.get("output_index", 0))
    show_train = bool(body.get("show_train", True))
    show_val = bool(body.get("show_val", True))
    show_test = bool(body.get("show_test", True))
    show_uncertainty = bool(body.get("show_uncertainty", True))
    figsize = body.get("figsize", [18, 6])

    if not isinstance(figsize, (list, tuple)) or len(figsize) != 2:
        figsize = [18, 6]

    if isinstance(model_dict, dict) and "model_id" in model_dict:
        model_id = model_dict["model_id"]
        if not model_id:
            return JSONResponse(
                status_code=400,
                content={"error": "model.model_id is missing or null."},
            )
        model_store = getattr(request.app.state, "model_store", {})
        model_dict = model_store.get(model_id)
        if inspect.isawaitable(model_dict):
            model_dict = await model_dict
        if model_dict is None:
            return JSONResponse(
                status_code=404,
                content={"error": f"Model with id {model_id} not found."},
            )
    else:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid model format. Expected a dict with 'model_id'."},
        )

    try:
        x_train, x_val, x_test = _parse_split_triplet(x_data, "x_data")
        y_train, y_val, y_test = _parse_split_triplet(y_data, "y_data")
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})

    if x_train.ndim != 3 or x_val.ndim != 3 or x_test.ndim != 3:
        return JSONResponse(
            status_code=400,
            content={
                "error": (
                    "Temporal evaluation expects x_data splits with shape "
                    "[samples, timesteps, features]."
                )
            },
        )

    model = model_dict["model"]
    model_metadata = model_dict.get("model_meta", {}) or {}
    seq_length = int(model_metadata.get("seq_length", x_train.shape[1]))
    seq_to_one = bool(model_metadata.get("seq_to_one", True))
    x_normalizer = _deserialize_normalizer(
        x_normalizer_payload, model_dict.get("x_normalizer")
    )
    y_normalizer = _deserialize_normalizer(
        y_normalizer_payload, model_dict.get("y_normalizer")
    )

    x_data_is_scaled = (
        x_data_is_scaled_input
        if x_data_is_scaled_input is not None
        else bool(model_metadata.get("x_data_is_scaled", False))
    )
    y_data_is_scaled = (
        y_data_is_scaled_input
        if y_data_is_scaled_input is not None
        else bool(model_metadata.get("y_data_is_scaled", False))
    )

    if x_normalizer is not None and not x_data_is_scaled:
        x_train_model = _transform_x_split_for_model(x_train, x_normalizer)
        x_val_model = _transform_x_split_for_model(x_val, x_normalizer)
        x_test_model = _transform_x_split_for_model(x_test, x_normalizer)
    else:
        x_train_model, x_val_model, x_test_model = x_train, x_val, x_test

    if model.num_mixtures > 0:
        pred_train, std_pred_train = make_predictions(
            model,
            x_train_model,
            y_normalizer=None,
            unnormalize=False,
            training=False,
        )
        pred_val, std_pred_val = make_predictions(
            model,
            x_val_model,
            y_normalizer=None,
            unnormalize=False,
            training=False,
        )
        pred_test, std_pred_test = make_predictions(
            model,
            x_test_model,
            y_normalizer=None,
            unnormalize=False,
            training=False,
        )
    else:
        pred_train = make_predictions(
            model,
            x_train_model,
            y_normalizer=None,
            unnormalize=False,
            training=False,
        )
        pred_val = make_predictions(
            model,
            x_val_model,
            y_normalizer=None,
            unnormalize=False,
            training=False,
        )
        pred_test = make_predictions(
            model,
            x_test_model,
            y_normalizer=None,
            unnormalize=False,
            training=False,
        )
        std_pred_train = np.zeros_like(pred_train)
        std_pred_val = np.zeros_like(pred_val)
        std_pred_test = np.zeros_like(pred_test)

    # Only inverse-transform y_data when the provided y_data is scaled.
    if unnormalize and y_normalizer is not None and y_data_is_scaled:
        y_train_real = _inverse_y_preserve_shape(y_train, y_normalizer)
        y_val_real = _inverse_y_preserve_shape(y_val, y_normalizer)
        y_test_real = _inverse_y_preserve_shape(y_test, y_normalizer)
    else:
        y_train_real = y_train
        y_val_real = y_val
        y_test_real = y_test

    if unnormalize and y_normalizer is not None:
        pred_train = _inverse_y_preserve_shape(pred_train, y_normalizer)
        pred_val = _inverse_y_preserve_shape(pred_val, y_normalizer)
        pred_test = _inverse_y_preserve_shape(pred_test, y_normalizer)
        std_pred_train = _inverse_std_preserve_shape(std_pred_train, y_normalizer)
        std_pred_val = _inverse_std_preserve_shape(std_pred_val, y_normalizer)
        std_pred_test = _inverse_std_preserve_shape(std_pred_test, y_normalizer)

    max_targets = pred_train.shape[1] if pred_train.ndim > 1 else 1
    if output_index < 0 or output_index >= max_targets:
        return JSONResponse(
            status_code=400,
            content={
                "error": (
                    f"output_index out of range. Got {output_index}, "
                    f"valid range is [0, {max_targets - 1}]."
                )
            },
        )

    fig, axes = plt.subplots(1, 3, figsize=(float(figsize[0]), float(figsize[1])))
    split_specs = [
        ("Train", axes[0], y_train_real, pred_train, std_pred_train, show_train),
        ("Validation", axes[1], y_val_real, pred_val, std_pred_val, show_val),
        ("Test", axes[2], y_test_real, pred_test, std_pred_test, show_test),
    ]

    for split_name, ax, y_split, pred_split, std_split, enabled in split_specs:
        if not enabled:
            ax.axis("off")
            ax.set_title(f"{split_name} (hidden)")
            continue

        y_series = y_split[:, output_index] if y_split.ndim > 1 else y_split
        pred_series = pred_split[:, output_index] if pred_split.ndim > 1 else pred_split
        std_series = std_split[:, output_index] if std_split.ndim > 1 else std_split

        time_offset = max(0, seq_length - 1) if seq_to_one else 0
        t = np.arange(len(pred_series)) + time_offset

        rmse = compute_rmse(y_series, pred_series)
        r2 = compute_rsquared(y_series, pred_series)

        ax.plot(t, y_series, color="black", linewidth=1.2, label="Truth")
        ax.plot(t, pred_series, color="C1", linewidth=1.2, label="Prediction")

        if show_uncertainty and np.any(std_series > 0):
            ax.fill_between(
                t,
                pred_series - std_series,
                pred_series + std_series,
                color="C1",
                alpha=0.25,
                label="Uncertainty",
            )

        ax.set_title(f"{split_name} (R2={r2:.3f}, RMSE={rmse:.3f})")
        ax.set_xlabel("Temporal Index")
        ax.set_ylabel(f"Target {output_index}")
        ax.legend(loc="best")

    fig.suptitle(
        f"Temporal Evaluation (seq_length={seq_length}, seq_to_one={seq_to_one})"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    image_payload = convert_fig_to_image(fig, image_format="pdf")
    image_payload["alignment"] = {
        "seq_length": seq_length,
        "seq_to_one": seq_to_one,
        "time_offset": max(0, seq_length - 1) if seq_to_one else 0,
        "output_index": output_index,
    }
    return image_payload
