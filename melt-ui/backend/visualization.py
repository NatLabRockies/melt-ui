import base64
import inspect
import json
import math
from io import BytesIO
from typing import List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from .utils import convert_fig_to_image

router = APIRouter()


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


def _coerce_x_for_split(
    x_raw, split: str = "train", min_features: int = 2
) -> np.ndarray:
    x_parsed = _parse_json_if_needed(x_raw)

    if not isinstance(x_parsed, (list, tuple)):
        raise ValueError("x must be an array-like payload.")

    if len(x_parsed) == 0:
        raise ValueError("x cannot be empty.")

    looks_like_splits = (
        len(x_parsed) == 3
        and all(isinstance(item, (list, tuple)) for item in x_parsed)
        and all(
            len(item) > 0 and isinstance(item[0], (list, tuple)) for item in x_parsed
        )
    )

    if looks_like_splits:
        idx = _split_index_from_name(split)
        x_np = np.asarray(x_parsed[idx], dtype=np.float32)
    else:
        x_np = np.asarray(x_parsed, dtype=np.float32)

    if x_np.ndim != 2:
        raise ValueError("x must be a 2D array [samples, features].")
    if x_np.shape[1] < int(min_features):
        raise ValueError(f"x must contain at least {int(min_features)} features.")

    return x_np


def _encode_vae_latent(model, x_np: np.ndarray):
    model.eval()
    with torch.no_grad():
        xt = torch.tensor(x_np, dtype=torch.float32)
        x_encoded = model.layer_dict["encoder_block"](xt)
        mdn_output = model.layer_dict["encoder_output"](x_encoded)
        mix_coeffs, means, log_vars = model.split_mdn_output(mdn_output)
        z = model.layer_dict["reparameterization_layer"](mix_coeffs, means, log_vars)

    return (
        z.detach().cpu().numpy(),
        mix_coeffs.detach().cpu().numpy(),
        means.detach().cpu().numpy(),
        log_vars.detach().cpu().numpy(),
    )


def _decode_vae_latent(model, z_np: np.ndarray):
    model.eval()
    with torch.no_grad():
        zt = torch.tensor(z_np, dtype=torch.float32)
        x_dec = model.layer_dict["decoder_block"](zt)
        x_dec = model.layer_dict["decoder_output"](x_dec)
    return x_dec.detach().cpu().numpy()


def _decode_from_payload(
    model,
    z,
    n_samples: Optional[int],
    latent_scale: Optional[float],
    random_state: Optional[int],
    max_decode_samples: int = 10000,
):
    if z is None:
        latent_dims = int(getattr(model, "latent_dims", 0) or 0)
        if latent_dims < 1:
            raise ValueError("Could not determine model latent_dims for sampling.")

        sample_count = int(n_samples if n_samples else 100)
        if sample_count < 1:
            raise ValueError("n_samples must be >= 1.")
        if sample_count > max_decode_samples:
            raise ValueError(
                f"n_samples too large ({sample_count}). Max allowed is {max_decode_samples}."
            )

        rng = np.random.default_rng(int(random_state or 42))
        z_np = rng.normal(
            loc=0.0,
            scale=float(latent_scale if latent_scale else 1.0),
            size=(sample_count, latent_dims),
        ).astype(np.float32)
    else:
        z_parsed = _parse_json_if_needed(z)
        z_np = np.asarray(z_parsed, dtype=np.float32)
        if z_np.ndim == 1:
            z_np = z_np.reshape(1, -1)
        if z_np.ndim != 2:
            raise ValueError("z must be 1D or 2D array.")
        if z_np.shape[0] > max_decode_samples:
            raise ValueError(
                f"z contains too many samples ({z_np.shape[0]}). Max allowed is {max_decode_samples}."
            )

    x_decoded = _decode_vae_latent(model, z_np)
    return z_np, x_decoded


class VAEEncodeLatentPayload(BaseModel):
    model: dict
    x: Union[List[float], List[List[float]], List[List[List[float]]], str]
    split: Optional[str] = "train"
    include_encoder_stats: Optional[bool] = True


@router.post("/vae_encode_latent")
async def vae_encode_latent(payload: VAEEncodeLatentPayload, request: Request):
    try:
        model = await _resolve_model_from_payload(payload.model, request)
        x_np = _coerce_x_for_split(
            payload.x,
            split=payload.split or "train",
            min_features=1,
        )

        z_np, mix_np, means_np, log_vars_np = _encode_vae_latent(model, x_np)

        include_stats = bool(payload.include_encoder_stats)
        return JSONResponse(
            content={
                "z": z_np.tolist(),
                "mix_coeffs": mix_np.tolist() if include_stats else [],
                "means": means_np.tolist() if include_stats else [],
                "log_vars": log_vars_np.tolist() if include_stats else [],
                "shape": {
                    "z": [int(v) for v in z_np.shape],
                    "mix_coeffs": [int(v) for v in mix_np.shape],
                    "means": [int(v) for v in means_np.shape],
                    "log_vars": [int(v) for v in log_vars_np.shape],
                },
            }
        )
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


class VAEReconstructDataPayload(BaseModel):
    model: dict
    x: Union[List[float], List[List[float]], List[List[List[float]]], str]
    split: Optional[str] = "train"
    error_metric: Optional[str] = "mse"


@router.post("/vae_reconstruct_data")
async def vae_reconstruct_data(payload: VAEReconstructDataPayload, request: Request):
    try:
        model = await _resolve_model_from_payload(payload.model, request)
        x_np = _coerce_x_for_split(
            payload.x,
            split=payload.split or "train",
            min_features=1,
        )

        model.eval()
        with torch.no_grad():
            xt = torch.tensor(x_np, dtype=torch.float32)
            x_reconstructed, _, _, _ = model(xt)
            xr_np = x_reconstructed.detach().cpu().numpy()

        residual = x_np - xr_np
        metric = str(payload.error_metric or "mse").lower()
        if metric == "mae":
            recon_error = np.mean(np.abs(residual), axis=1)
        else:
            recon_error = np.mean(np.square(residual), axis=1)

        return JSONResponse(
            content={
                "x_reconstructed": xr_np.tolist(),
                "residual": residual.tolist(),
                "reconstruction_error": recon_error.tolist(),
                "shape": {
                    "x_reconstructed": [int(v) for v in xr_np.shape],
                    "residual": [int(v) for v in residual.shape],
                    "reconstruction_error": [int(v) for v in recon_error.shape],
                },
                "metadata": {
                    "error_metric": metric,
                },
            }
        )
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


class VAEAnomalyScoresPayload(BaseModel):
    model: dict
    x: Union[List[float], List[List[float]], List[List[List[float]]], str]
    split: Optional[str] = "train"
    error_metric: Optional[str] = "mse"
    threshold_quantile: Optional[float] = 0.99


@router.post("/vae_anomaly_scores")
async def vae_anomaly_scores(payload: VAEAnomalyScoresPayload, request: Request):
    try:
        model = await _resolve_model_from_payload(payload.model, request)
        x_np = _coerce_x_for_split(
            payload.x,
            split=payload.split or "train",
            min_features=1,
        )

        q = float(payload.threshold_quantile if payload.threshold_quantile else 0.99)
        if q <= 0.0 or q >= 1.0:
            raise ValueError("threshold_quantile must be in (0, 1).")

        model.eval()
        with torch.no_grad():
            xt = torch.tensor(x_np, dtype=torch.float32)
            x_reconstructed, _, _, _ = model(xt)
            xr_np = x_reconstructed.detach().cpu().numpy()

        residual = x_np - xr_np
        metric = str(payload.error_metric or "mse").lower()
        if metric == "mae":
            scores = np.mean(np.abs(residual), axis=1)
        else:
            scores = np.mean(np.square(residual), axis=1)

        threshold = float(np.quantile(scores, q))
        anomaly_mask = scores >= threshold
        ranked_indices = np.argsort(-scores)

        return JSONResponse(
            content={
                "scores": scores.tolist(),
                "threshold": threshold,
                "anomaly_mask": anomaly_mask.tolist(),
                "ranked_indices": ranked_indices.tolist(),
                "shape": {
                    "scores": [int(scores.shape[0])],
                    "anomaly_mask": [int(anomaly_mask.shape[0])],
                    "ranked_indices": [int(ranked_indices.shape[0])],
                },
                "metadata": {
                    "error_metric": metric,
                    "threshold_quantile": q,
                    "num_anomalies": int(np.sum(anomaly_mask)),
                },
            }
        )
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


class VAELatentClusterPayload(BaseModel):
    model: dict
    x: Union[List[float], List[List[float]], List[List[List[float]]], str]
    split: Optional[str] = "train"
    n_clusters: Optional[int] = 5
    random_state: Optional[int] = 42
    n_init: Optional[int] = 10


@router.post("/vae_latent_cluster")
async def vae_latent_cluster(payload: VAELatentClusterPayload, request: Request):
    try:
        model = await _resolve_model_from_payload(payload.model, request)
        x_np = _coerce_x_for_split(
            payload.x,
            split=payload.split or "train",
            min_features=1,
        )

        n_clusters = int(payload.n_clusters if payload.n_clusters else 5)
        if n_clusters < 2:
            raise ValueError("n_clusters must be >= 2.")

        z_np, _, _, _ = _encode_vae_latent(model, x_np)
        if z_np.shape[0] < n_clusters:
            raise ValueError("n_clusters cannot exceed number of samples.")

        km = KMeans(
            n_clusters=n_clusters,
            random_state=int(payload.random_state if payload.random_state else 42),
            n_init=int(payload.n_init if payload.n_init else 10),
        )
        labels = km.fit_predict(z_np)
        centroids = km.cluster_centers_

        return JSONResponse(
            content={
                "z": z_np.tolist(),
                "cluster_labels": labels.tolist(),
                "centroids": centroids.tolist(),
                "shape": {
                    "z": [int(v) for v in z_np.shape],
                    "cluster_labels": [int(labels.shape[0])],
                    "centroids": [int(v) for v in centroids.shape],
                },
                "metadata": {
                    "n_clusters": n_clusters,
                    "inertia": float(km.inertia_),
                },
            }
        )
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


class VAEDecodeLatentPayload(BaseModel):
    model: dict
    z: Optional[Union[List[float], List[List[float]], str]] = None
    n_samples: Optional[int] = 100
    latent_scale: Optional[float] = 1.0
    random_state: Optional[int] = 42


@router.post("/vae_decode_latent")
async def vae_decode_latent(payload: VAEDecodeLatentPayload, request: Request):
    try:
        model = await _resolve_model_from_payload(payload.model, request)

        z_np, x_decoded = _decode_from_payload(
            model=model,
            z=payload.z,
            n_samples=payload.n_samples,
            latent_scale=payload.latent_scale,
            random_state=payload.random_state,
        )

        return JSONResponse(
            content={
                "z": z_np.tolist(),
                "x_decoded": x_decoded.tolist(),
                "shape": {
                    "z": [int(v) for v in z_np.shape],
                    "x_decoded": [int(v) for v in x_decoded.shape],
                },
            }
        )
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


class PlotVAEDecodedPayload(BaseModel):
    model: dict
    z: Optional[Union[List[float], List[List[float]], str]] = None
    labels: Optional[Union[List[float], List[List[float]], List[int], str]] = None
    n_samples: Optional[int] = 100
    latent_scale: Optional[float] = 1.0
    random_state: Optional[int] = 42
    feature_x: Optional[int] = 0
    feature_y: Optional[int] = 1
    feature_z: Optional[int] = 2
    mode: Optional[str] = "auto"
    use_pca: Optional[bool] = True
    pairplot: Optional[bool] = False
    plot_3d: Optional[bool] = False
    alpha: Optional[float] = 0.7
    point_size: Optional[float] = 18.0
    max_plot_samples: Optional[int] = 2000
    figsize: Optional[Tuple[int, int]] = (8, 6)
    image_format: Optional[str] = "pdf"


@router.post("/plot_vae_decoded")
async def plot_vae_decoded(payload: PlotVAEDecodedPayload, request: Request):
    try:
        model = await _resolve_model_from_payload(payload.model, request)
        z_np, x_decoded = _decode_from_payload(
            model=model,
            z=payload.z,
            n_samples=payload.n_samples,
            latent_scale=payload.latent_scale,
            random_state=payload.random_state,
        )

        labels_np = None
        if payload.labels is not None:
            labels_parsed = _parse_json_if_needed(payload.labels)
            labels_np = np.asarray(labels_parsed).reshape(-1)
            if labels_np.shape[0] != x_decoded.shape[0]:
                raise ValueError(
                    f"labels length ({labels_np.shape[0]}) must match decoded sample count ({x_decoded.shape[0]})."
                )

        max_plot_samples = int(payload.max_plot_samples or 2000)
        if max_plot_samples < 10:
            raise ValueError("max_plot_samples must be >= 10.")

        if x_decoded.shape[0] > max_plot_samples:
            idx = np.linspace(0, x_decoded.shape[0] - 1, max_plot_samples).astype(int)
            x_plot = x_decoded[idx]
            labels_plot = labels_np[idx] if labels_np is not None else None
        else:
            x_plot = x_decoded
            labels_plot = labels_np

        fx = int(payload.feature_x if payload.feature_x is not None else 0)
        fy = int(payload.feature_y if payload.feature_y is not None else 1)
        fz = int(payload.feature_z if payload.feature_z is not None else 2)

        mode = str(payload.mode or "auto").lower().strip()
        if mode not in {"auto", "scatter2d", "pca2d", "pairplot", "scatter3d"}:
            raise ValueError(
                "mode must be one of: auto, scatter2d, pca2d, pairplot, scatter3d"
            )

        alpha = float(payload.alpha if payload.alpha is not None else 0.7)
        point_size = float(
            payload.point_size if payload.point_size is not None else 18.0
        )

        n_features = int(x_plot.shape[1])

        if mode == "auto":
            if n_features == 1:
                mode = "scatter2d"
            elif n_features == 2:
                mode = "scatter2d"
            elif bool(payload.pairplot) and n_features <= 6 and x_plot.shape[0] <= 500:
                mode = "pairplot"
            elif bool(payload.plot_3d) and n_features >= 3:
                mode = "scatter3d"
            elif bool(payload.use_pca) and n_features > 2:
                mode = "pca2d"
            else:
                mode = "scatter2d"

        if mode == "pairplot":
            if n_features > 6:
                raise ValueError("pairplot mode supports at most 6 features.")
            if x_plot.shape[0] > 500:
                raise ValueError("pairplot mode supports at most 500 samples.")

            col_names = [f"decoded_{i}" for i in range(n_features)]
            df = pd.DataFrame(x_plot, columns=col_names)
            if labels_plot is not None:
                df["label"] = labels_plot
                grid = sns.pairplot(df, vars=col_names, hue="label")
            else:
                grid = sns.pairplot(df, vars=col_names)
            fig = grid.fig
            fig.suptitle("VAE Decoded Samples (Pairplot)", y=1.02)
            return convert_fig_to_image(fig, image_format=payload.image_format or "pdf")

        fig = plt.figure(figsize=payload.figsize or (8, 6))

        if mode == "pca2d":
            if n_features < 2:
                raise ValueError("pca2d mode requires at least 2 decoded features.")
            proj = PCA(n_components=2).fit_transform(x_plot)
            ax = fig.add_subplot(111)
            if labels_plot is None:
                ax.scatter(proj[:, 0], proj[:, 1], alpha=alpha, s=point_size)
            else:
                sc = ax.scatter(
                    proj[:, 0],
                    proj[:, 1],
                    c=labels_plot,
                    cmap="viridis",
                    alpha=alpha,
                    s=point_size,
                )
                fig.colorbar(sc, ax=ax, label="Label")
            ax.set_xlabel("PC 1")
            ax.set_ylabel("PC 2")
            ax.set_title("VAE Decoded Samples (PCA 2D)")

        elif mode == "scatter3d":
            if n_features < 3:
                raise ValueError("scatter3d mode requires at least 3 decoded features.")
            _validate_axis_pair(fx, fy, n_features, "feature")
            if fz < 0 or fz >= n_features:
                raise ValueError(
                    f"feature_z out of range for decoded dimension {n_features}: {fz}."
                )
            ax = fig.add_subplot(111, projection="3d")
            if labels_plot is None:
                ax.scatter(
                    x_plot[:, fx],
                    x_plot[:, fy],
                    x_plot[:, fz],
                    alpha=alpha,
                    s=point_size,
                )
            else:
                sc = ax.scatter(
                    x_plot[:, fx],
                    x_plot[:, fy],
                    x_plot[:, fz],
                    c=labels_plot,
                    cmap="viridis",
                    alpha=alpha,
                    s=point_size,
                )
                fig.colorbar(sc, ax=ax, label="Label")
            ax.set_xlabel(f"Decoded {fx + 1}")
            ax.set_ylabel(f"Decoded {fy + 1}")
            ax.set_zlabel(f"Decoded {fz + 1}")
            ax.set_title("VAE Decoded Samples (3D)")

        else:
            # scatter2d fallback (supports 1D and 2D+)
            ax = fig.add_subplot(111)
            if n_features == 1:
                ax.hist(x_plot[:, 0], bins=30, alpha=0.85)
                ax.set_xlabel("Decoded 1")
                ax.set_ylabel("Count")
                ax.set_title("VAE Decoded Samples (1D Histogram)")
            else:
                _validate_axis_pair(fx, fy, n_features, "feature")
                if labels_plot is None:
                    ax.scatter(
                        x_plot[:, fx],
                        x_plot[:, fy],
                        alpha=alpha,
                        s=point_size,
                    )
                else:
                    sc = ax.scatter(
                        x_plot[:, fx],
                        x_plot[:, fy],
                        c=labels_plot,
                        cmap="viridis",
                        alpha=alpha,
                        s=point_size,
                    )
                    fig.colorbar(sc, ax=ax, label="Label")
                ax.set_xlabel(f"Decoded {fx + 1}")
                ax.set_ylabel(f"Decoded {fy + 1}")
                ax.set_title("VAE Decoded Samples (2D)")

        # ensure tight_layout does not overlap the external colorbar area
        fig.tight_layout(rect=[0.0, 0.0, 0.88, 1.0])
        return convert_fig_to_image(fig, image_format=payload.image_format or "pdf")
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


class VAELatentInterpolatePayload(BaseModel):
    model: dict
    z_start: Union[List[float], List[List[float]], str]
    z_end: Union[List[float], List[List[float]], str]
    num_steps: Optional[int] = 16


@router.post("/vae_latent_interpolate")
async def vae_latent_interpolate(
    payload: VAELatentInterpolatePayload, request: Request
):
    try:
        model = await _resolve_model_from_payload(payload.model, request)

        z_start = np.asarray(_parse_json_if_needed(payload.z_start), dtype=np.float32)
        z_end = np.asarray(_parse_json_if_needed(payload.z_end), dtype=np.float32)

        if z_start.ndim > 1:
            z_start = z_start.reshape(-1)
        if z_end.ndim > 1:
            z_end = z_end.reshape(-1)
        if z_start.ndim != 1 or z_end.ndim != 1:
            raise ValueError("z_start and z_end must be vectors.")
        if z_start.shape[0] != z_end.shape[0]:
            raise ValueError("z_start and z_end must have the same dimensionality.")

        num_steps = int(payload.num_steps if payload.num_steps else 16)
        if num_steps < 2:
            raise ValueError("num_steps must be >= 2.")

        alphas = np.linspace(0.0, 1.0, num_steps, dtype=np.float32)
        z_path = np.stack(
            [(1.0 - a) * z_start + a * z_end for a in alphas],
            axis=0,
        )

        x_path = _decode_vae_latent(model, z_path)

        return JSONResponse(
            content={
                "z_path": z_path.tolist(),
                "x_path": x_path.tolist(),
                "shape": {
                    "z_path": [int(v) for v in z_path.shape],
                    "x_path": [int(v) for v in x_path.shape],
                },
            }
        )
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


def _coerce_labels_for_split(labels_raw, split: str, n_samples: int):
    if labels_raw is None:
        return None

    labels_parsed = _parse_json_if_needed(labels_raw)
    if not isinstance(labels_parsed, (list, tuple)):
        raise ValueError("labels must be array-like when provided.")

    labels_np = None

    looks_like_splits = len(labels_parsed) == 3 and all(
        isinstance(item, (list, tuple)) for item in labels_parsed
    )

    if looks_like_splits:
        idx = _split_index_from_name(split)
        labels_np = np.asarray(labels_parsed[idx])
    else:
        labels_np = np.asarray(labels_parsed)

    labels_np = labels_np.reshape(-1)
    if labels_np.shape[0] != n_samples:
        raise ValueError(
            f"labels length ({labels_np.shape[0]}) must match sample count ({n_samples})."
        )

    return labels_np


async def _resolve_model_from_payload(request_model, request):
    if not isinstance(request_model, dict) or "model_id" not in request_model:
        raise ValueError("Invalid model payload. Expected object with 'model_id'.")

    model_id = request_model.get("model_id")
    if not model_id:
        raise ValueError("model.model_id is missing or null.")

    model_store = getattr(request.app.state, "model_store", {})
    model_dict = model_store.get(model_id)
    if inspect.isawaitable(model_dict):
        model_dict = await model_dict
    if model_dict is None:
        raise ValueError(f"Model with id {model_id} not found.")

    if str(model_dict.get("model_meta", {}).get("architecture", "")).lower() != "vae":
        raise ValueError("Provided model is not a VAE model.")

    model = model_dict.get("model")
    if model is None:
        raise ValueError("Model payload missing trained model object.")

    return model


def _validate_axis_pair(axis_x: int, axis_y: int, max_dims: int, name: str):
    if axis_x < 0 or axis_y < 0:
        raise ValueError(f"{name} axis indices must be >= 0.")
    if axis_x >= max_dims or axis_y >= max_dims:
        raise ValueError(
            f"{name} axis index out of range for dimension {max_dims}: "
            f"received ({axis_x}, {axis_y})."
        )


class PlotVAEInitialClustersPayload(BaseModel):
    x: Union[List[float], List[List[float]], List[List[List[float]]], str]
    labels: Optional[Union[List[float], List[List[float]], List[int], str]] = None
    split: Optional[str] = "train"
    feature_x: Optional[int] = 0
    feature_y: Optional[int] = 1
    alpha: Optional[float] = 0.7
    point_size: Optional[float] = 18.0
    figsize: Optional[Tuple[int, int]] = (8, 6)
    image_format: Optional[str] = "pdf"


@router.post("/plot_vae_initial_clusters")
async def plot_vae_initial_clusters(payload: PlotVAEInitialClustersPayload):
    try:
        x_np = _coerce_x_for_split(payload.x, split=payload.split or "train")
        labels_np = _coerce_labels_for_split(
            payload.labels, split=payload.split or "train", n_samples=x_np.shape[0]
        )

        fx = int(payload.feature_x if payload.feature_x is not None else 0)
        fy = int(payload.feature_y if payload.feature_y is not None else 1)
        _validate_axis_pair(fx, fy, x_np.shape[1], "feature")

        fig, ax = plt.subplots(figsize=payload.figsize or (8, 6))

        if labels_np is None:
            ax.scatter(
                x_np[:, fx],
                x_np[:, fy],
                alpha=float(payload.alpha if payload.alpha is not None else 0.7),
                s=float(payload.point_size if payload.point_size is not None else 18.0),
            )
        else:
            sc = ax.scatter(
                x_np[:, fx],
                x_np[:, fy],
                c=labels_np,
                cmap="viridis",
                alpha=float(payload.alpha if payload.alpha is not None else 0.7),
                s=float(payload.point_size if payload.point_size is not None else 18.0),
            )
            fig.colorbar(sc, ax=ax, label="Cluster")

        ax.set_title("Initial Feature-Space Clusters")
        ax.set_xlabel(f"Feature {fx + 1}")
        ax.set_ylabel(f"Feature {fy + 1}")
        fig.tight_layout()

        return convert_fig_to_image(fig, image_format=payload.image_format or "pdf")
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


class PlotVAELatentClustersPayload(BaseModel):
    model: dict
    x: Union[List[float], List[List[float]], List[List[List[float]]], str]
    labels: Optional[Union[List[float], List[List[float]], List[int], str]] = None
    split: Optional[str] = "train"
    latent_x: Optional[int] = 0
    latent_y: Optional[int] = 1
    alpha: Optional[float] = 0.7
    point_size: Optional[float] = 18.0
    figsize: Optional[Tuple[int, int]] = (8, 6)
    image_format: Optional[str] = "pdf"


@router.post("/plot_vae_latent_clusters")
async def plot_vae_latent_clusters(
    payload: PlotVAELatentClustersPayload, request: Request
):
    try:
        model = await _resolve_model_from_payload(payload.model, request)
        x_np = _coerce_x_for_split(payload.x, split=payload.split or "train")
        labels_np = _coerce_labels_for_split(
            payload.labels, split=payload.split or "train", n_samples=x_np.shape[0]
        )

        model.eval()
        with torch.no_grad():
            xt = torch.tensor(x_np, dtype=torch.float32)
            x_encoded = model.layer_dict["encoder_block"](xt)
            mdn_output = model.layer_dict["encoder_output"](x_encoded)
            mix_coeffs, means, log_vars = model.split_mdn_output(mdn_output)
            z = model.layer_dict["reparameterization_layer"](
                mix_coeffs, means, log_vars
            )
            z_np = z.detach().cpu().numpy()

        lx = int(payload.latent_x if payload.latent_x is not None else 0)
        ly = int(payload.latent_y if payload.latent_y is not None else 1)
        _validate_axis_pair(lx, ly, z_np.shape[1], "latent")

        fig, ax = plt.subplots(figsize=payload.figsize or (8, 6))
        if labels_np is None:
            ax.scatter(
                z_np[:, lx],
                z_np[:, ly],
                alpha=float(payload.alpha if payload.alpha is not None else 0.7),
                s=float(payload.point_size if payload.point_size is not None else 18.0),
            )
        else:
            sc = ax.scatter(
                z_np[:, lx],
                z_np[:, ly],
                c=labels_np,
                cmap="viridis",
                alpha=float(payload.alpha if payload.alpha is not None else 0.7),
                s=float(payload.point_size if payload.point_size is not None else 18.0),
            )
            fig.colorbar(sc, ax=ax, label="Cluster")

        ax.set_title("Latent-Space Representation")
        ax.set_xlabel(f"Latent Dim {lx + 1}")
        ax.set_ylabel(f"Latent Dim {ly + 1}")
        fig.tight_layout()

        return convert_fig_to_image(fig, image_format=payload.image_format or "pdf")
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


class PlotVAEReconstructionPayload(BaseModel):
    model: dict
    x: Union[List[float], List[List[float]], List[List[List[float]]], str]
    labels: Optional[Union[List[float], List[List[float]], List[int], str]] = None
    split: Optional[str] = "train"
    feature_x: Optional[int] = 0
    feature_y: Optional[int] = 1
    alpha: Optional[float] = 0.7
    point_size: Optional[float] = 18.0
    figsize: Optional[Tuple[int, int]] = (10, 4)
    image_format: Optional[str] = "pdf"


@router.post("/plot_vae_reconstruction")
async def plot_vae_reconstruction(
    payload: PlotVAEReconstructionPayload, request: Request
):
    try:
        model = await _resolve_model_from_payload(payload.model, request)
        x_np = _coerce_x_for_split(payload.x, split=payload.split or "train")
        labels_np = _coerce_labels_for_split(
            payload.labels, split=payload.split or "train", n_samples=x_np.shape[0]
        )

        fx = int(payload.feature_x if payload.feature_x is not None else 0)
        fy = int(payload.feature_y if payload.feature_y is not None else 1)
        _validate_axis_pair(fx, fy, x_np.shape[1], "feature")

        model.eval()
        with torch.no_grad():
            xt = torch.tensor(x_np, dtype=torch.float32)
            x_reconstructed, _, _, _ = model(xt)
            xr_np = x_reconstructed.detach().cpu().numpy()

        fig, axes = plt.subplots(1, 2, figsize=payload.figsize or (10, 4))

        scatter_kwargs = {
            "alpha": float(payload.alpha if payload.alpha is not None else 0.7),
            "s": float(payload.point_size if payload.point_size is not None else 18.0),
        }

        if labels_np is None:
            axes[0].scatter(x_np[:, fx], x_np[:, fy], **scatter_kwargs)
            axes[1].scatter(xr_np[:, fx], xr_np[:, fy], **scatter_kwargs)
        else:
            sc0 = axes[0].scatter(
                x_np[:, fx], x_np[:, fy], c=labels_np, cmap="viridis", **scatter_kwargs
            )
            axes[1].scatter(
                xr_np[:, fx],
                xr_np[:, fy],
                c=labels_np,
                cmap="viridis",
                **scatter_kwargs,
            )
            # reserve space on the right for an external colorbar and add it there
            fig.subplots_adjust(right=0.88)
            cax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
            fig.colorbar(sc0, cax=cax, label="Cluster")

        axes[0].set_title("Original Data")
        axes[1].set_title("Reconstructed Data")
        for ax in axes:
            ax.set_xlabel(f"Feature {fx + 1}")
            ax.set_ylabel(f"Feature {fy + 1}")

        # fig.tight_layout()
        return convert_fig_to_image(fig, image_format=payload.image_format or "pdf")
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


def _to_numpy(array_like):
    """Convert numpy/pandas/torch to numpy.ndarray."""
    if pd is not None and isinstance(array_like, (pd.Series, pd.DataFrame)):
        return array_like.values
    if torch is not None and isinstance(array_like, torch.Tensor):
        return array_like.detach().cpu().numpy()
    return np.asarray(array_like)


def plot_xy(
    x,
    y,
    figsize: Tuple[int, int] = (10, 6),
    n_cols: int = 3,
    return_fig: bool = True,
):
    """
    Plot x against y  using seaborn whitegrid.
    """
    sns.set_style("whitegrid")

    x_np = _to_numpy(x)
    y_np = _to_numpy(y)

    n_samples, n_features = x_np.shape
    if n_samples != y_np.shape[0]:
        raise ValueError(
            f"Number of samples mismatch: x has {n_samples}, y has {y_np.shape[0]}"
        )

    _, n_targets = y_np.shape if y_np.ndim == 2 else (y_np.shape[0], 1)

    # Feature names
    feature_names = [f"x{i}" for i in range(n_features)]

    # Target names
    target_names = [f"y{i}" for i in range(n_targets)]

    # If single feature, single plot
    if n_features == 1:
        fig, ax = plt.subplots(figsize=figsize)
        if y_np.ndim == 1:
            sns.scatterplot(
                x=x_np.ravel(),
                y=y_np,
                ax=ax,
                color="tab:blue",
                s=30,
                label=target_names[0],
            )
        else:
            palette = sns.color_palette("tab10")
            for j in range(y_np.shape[1]):
                sns.scatterplot(
                    x=x_np.ravel(),
                    y=y_np[:, j],
                    ax=ax,
                    color=palette[j % len(palette)],
                    s=30,
                    label=target_names[j],
                )
        ax.legend()

        ax.set_xlabel(feature_names[0])
        ax.set_ylabel("target")

        plt.tight_layout()
        return (fig, ax) if return_fig else None

    # If multiple features, grid of plots
    cols = min(n_cols, n_features)
    rows = math.ceil(n_features / cols)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(figsize[0], max(figsize[1], 3 * rows)),
        squeeze=False,
    )
    axes_flat = axes.flatten()
    palette = sns.color_palette("tab10")

    for i in range(n_features):
        ax = axes_flat[i]
        xi = x_np[:, i]
        # sns.scatterplot(x=xi, y=y_np, ax=ax, color=palette[i % len(palette)], s=30)
        if y_np.ndim == 1:
            sns.scatterplot(
                x=xi,
                y=y_np,
                ax=ax,
                color="tab:blue",
                s=30,
                label=target_names[0],
            )
        else:
            palette = sns.color_palette("tab10")
            for j in range(y_np.shape[1]):
                sns.scatterplot(
                    x=xi,
                    y=y_np[:, j],
                    ax=ax,
                    color=palette[j % len(palette)],
                    s=30,
                    label=target_names[j],
                )
        ax.legend()

        ax.set_xlabel(feature_names[i])
        ax.set_ylabel("target")

    for j in range(n_features, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    return (fig, axes) if return_fig else None


class PlotXYPayload(BaseModel):
    x: Union[List[float], List[List[float]]]
    y: Union[List[float], List[List[float]]]
    figsize: Optional[Tuple[int, int]] = (10, 6)
    n_cols: int = 3
    image_format: Optional[str] = "pdf"


async def plot_xy_node(payload: PlotXYPayload) -> dict:
    """
    Async backend node that takes x/y (and options),
    builds the plot, and returns a base64 PNG string.
    """
    # Convert payload x,y to numpy-compatible
    x = _to_numpy(payload.x)
    y = _to_numpy(payload.y)

    fig, _ = plot_xy(
        x,
        y,
        figsize=payload.figsize or (10, 6),
        n_cols=payload.n_cols,
        return_fig=True,
    )

    # Serialize to chosen format in-memory
    fmt = (payload.image_format or "png").lower()
    buf_main = BytesIO()
    fig.savefig(buf_main, format=fmt, bbox_inches="tight")
    buf_main.seek(0)
    main_data = buf_main.getvalue()

    preview_buf = BytesIO()
    fig.savefig(preview_buf, format="png", bbox_inches="tight", dpi=150)
    preview_buf.seek(0)
    preview_data = preview_buf.getvalue()

    plt.close(fig)

    b64_main = base64.b64encode(main_data).decode("ascii")
    b64_preview = base64.b64encode(preview_data).decode("ascii")

    if fmt == "svg":
        image_uri = f"data:image/svg+xml;base64,{b64_main}"
    elif fmt == "pdf":
        image_uri = f"data:application/pdf;base64,{b64_main}"
    else:
        image_uri = f"data:image/png;base64,{b64_main}"

    # preview always PNG
    preview_uri = f"data:image/png;base64,{b64_preview}"

    return {"image": image_uri, "preview": preview_uri}


@router.post("/plot_xy")
async def plot_xy_endpoint(payload: PlotXYPayload):
    """
    FastAPI endpoint that wraps the async node.
    Returns JSON: { "image": "data:image/png;base64,..." }
    or { "error": "..." } on failure.
    """
    try:
        result = await plot_xy_node(payload)
        return JSONResponse(content=result)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


class PlotHistoryPayload(BaseModel):
    history: dict
    plot_log: Optional[bool] = True
    figsize: Optional[Tuple[int, int]] = (10, 6)
    image_format: Optional[str] = "pdf"


async def plot_history_node(payload: PlotHistoryPayload) -> dict:
    """
    Async backend node that takes training history (and options),
    builds the plot, and returns a base64 PNG string.
    """

    history = payload.history
    if not isinstance(history, dict) or len(history) == 0:
        raise ValueError("history must be a non-empty dictionary.")

    metrics = [
        k
        for k, v in history.items()
        if not k.startswith("val_")
        and k not in {"lr", "epoch", "kl_per_dim"}
        and isinstance(v, (list, tuple))
        and len(v) > 0
    ]
    if "loss" in metrics:
        metrics = ["loss"] + [m for m in metrics if m != "loss"]
    if not metrics:
        raise ValueError(
            "history does not contain plottable metric arrays (e.g. 'loss')."
        )
    plot_log = payload.plot_log if payload.plot_log is not None else True
    # figsize = payload.figsize if payload.figsize is not None else (10, 6)

    # Create layout: metrics on top row, LR on bottom row
    if plot_log:
        # 2 rows, 2 columns; bottom row spans both columns
        fig = plt.figure(figsize=(12, 6))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])

        ax1 = fig.add_subplot(gs[0, 0])  # linear metrics
        ax2 = fig.add_subplot(gs[0, 1])  # log metrics
        ax_lr = fig.add_subplot(gs[1, 0])  # LR, linear axes
    else:
        # 2 rows, 1 column: metrics over LR
        fig = plt.figure(figsize=(6, 6))
        gs = fig.add_gridspec(2, 1, height_ratios=[2, 1])

        ax1 = fig.add_subplot(gs[0, 0])  # linear metrics
        ax2 = None
        ax_lr = fig.add_subplot(gs[1, 0])  # LR, linear axes

    # Plot metrics for both training and validation sets
    for metric in metrics:
        ax1.plot(history[metric], label=f"Train {metric}")
        if f"val_{metric}" in history:
            ax1.plot(history[f"val_{metric}"], label=f"Validation {metric}")

        if plot_log and ax2 is not None:
            ax2.plot(history[metric], label=f"Train {metric}")
            if f"val_{metric}" in history:
                ax2.plot(history[f"val_{metric}"], label=f"Validation {metric}")

    # Set plot labels and legend for metrics
    ax1.legend()
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Metrics")
    # ax1.set_title("Training / Validation metrics (linear)")

    if plot_log and ax2 is not None:
        ax2.legend()
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Metrics")
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        # ax2.set_title("Training / Validation metrics (log-log)")

    # New: plot learning rate on its own row, linear axes
    if "lr" in history:
        ax_lr.plot(history["lr"])
        ax_lr.set_xlabel("Epoch")
        ax_lr.set_ylabel("Learning rate")
        # ax_lr.set_title("Learning rate schedule")

    fig.tight_layout()

    # Serialize to chosen format in-memory
    fmt = (payload.image_format or "png").lower()
    buf_main = BytesIO()
    fig.savefig(buf_main, format=fmt, bbox_inches="tight")
    buf_main.seek(0)
    main_data = buf_main.getvalue()

    preview_buf = BytesIO()
    fig.savefig(preview_buf, format="png", bbox_inches="tight", dpi=150)
    preview_buf.seek(0)
    preview_data = preview_buf.getvalue()

    plt.close(fig)

    b64_main = base64.b64encode(main_data).decode("ascii")
    b64_preview = base64.b64encode(preview_data).decode("ascii")

    if fmt == "svg":
        image_uri = f"data:image/svg+xml;base64,{b64_main}"
    elif fmt == "pdf":
        image_uri = f"data:application/pdf;base64,{b64_main}"
    else:
        image_uri = f"data:image/png;base64,{b64_main}"

    # preview always PNG
    preview_uri = f"data:image/png;base64,{b64_preview}"

    return {"image": image_uri, "preview": preview_uri}


@router.post("/plot_history")
async def plot_history_endpoint(payload: PlotHistoryPayload):
    """
    FastAPI endpoint that plots training history.
    """
    try:
        result = await plot_history_node(payload)
        return JSONResponse(content=result)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
