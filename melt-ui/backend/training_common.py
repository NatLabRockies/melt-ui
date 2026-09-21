import json
import threading
import uuid
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from ptmelt.utils.preprocessing import IdentityScaler, get_normalizers
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from .utils import scaler_to_dict

SCHEDULER_KWARGS: Dict[str, Any] = {
    "factor": 0.5,
    "patience": 50,
    "min_lr": 1e-6,
    "mode": "min",
    "threshold": 1e-4,
    "threshold_mode": "rel",
    "cooldown": 10,
}

HYPERPARAMETER_ALIASES: Dict[str, str] = {
    "act_fun": "activation_function",
    "arch_type": "model_architecture",
    "dropout": "dropout_rate",
    "epochs": "num_epochs",
    "loss_fn": "loss_function",
}

COMMON_HYPERPARAMETER_KEYS = {
    "activation_function",
    "batch_norm",
    "batch_size",
    "dropout_rate",
    "learning_rate",
    "loss_function",
    "l1_reg",
    "l2_reg",
    "lr_scheduler",
    "num_epochs",
    "num_mixtures",
    "optimizer",
    "output_activation",
    "random_state",
    "shuffle",
}

HYPERPARAMETER_ALLOWLISTS = {
    "static": COMMON_HYPERPARAMETER_KEYS
    | {
        "architecture",
        "layers_per_block",
        "model_architecture",
        "node_list",
        "width",
        "depth",
    },
    "temporal_rnn": COMMON_HYPERPARAMETER_KEYS
    | {
        "architecture",
        "depth",
        "head_type",
        "node_list",
        "rnn_type",
        "seq_length",
        "seq_to_one",
        "suffix_crop",
        "suffix_crop_min_length",
        "width",
    },
    "temporal_transformer": COMMON_HYPERPARAMETER_KEYS
    | {
        "depth",
        "ff_dim",
        "head_type",
        "max_seq_len",
        "num_heads",
        "seq_length",
        "seq_to_one",
        "use_causal_mask",
        "width",
    },
    "vae": COMMON_HYPERPARAMETER_KEYS
    | {
        "decoder_node_list",
        "encoder_node_list",
        "latent_dims",
    },
}


def _extract_hyperparameter_mapping(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {}

    for key in (
        "trainer_hyperparameters",
        "best_hyperparameters",
        "hyperparameters",
        "hyperparams",
    ):
        nested = value.get(key)
        if isinstance(nested, dict):
            return dict(nested)

    payload = value.get("payload")
    if isinstance(payload, dict):
        extracted = _extract_hyperparameter_mapping(payload)
        if extracted:
            return extracted

    return dict(value)


def _sampled_layers_to_node_list(overrides: Dict[str, Any]) -> None:
    if "node_list" in overrides:
        return

    max_depth = overrides.get("max_depth")
    if max_depth is None:
        return

    node_list = []
    for layer_index in range(int(max_depth)):
        width = int(overrides.get(f"layer_{layer_index}_width", 0) or 0)
        if width > 0:
            node_list.append(width)
    if node_list:
        overrides["node_list"] = node_list


def apply_hyperparameter_overrides(
    body: Dict[str, Any], trainer_family: str
) -> Dict[str, Any]:
    """Merge validated HPO overrides into a request body.

    The original body wins for data and execution-only fields; only allowlisted
    trainer initialization fields can be overridden.
    """

    raw = (
        body.get("hyperparameters")
        or body.get("hyperparams")
        or body.get("best_hyperparameters")
    )
    overrides = _extract_hyperparameter_mapping(raw)
    if not overrides:
        return body

    normalized: Dict[str, Any] = {}
    _sampled_layers_to_node_list(overrides)
    for key, value in overrides.items():
        normalized_key = HYPERPARAMETER_ALIASES.get(str(key), str(key))
        normalized[normalized_key] = value

    allowed = HYPERPARAMETER_ALLOWLISTS.get(trainer_family, set())
    merged = dict(body)
    applied = {}
    ignored = []
    for key, value in normalized.items():
        if key in allowed:
            merged[key] = value
            applied[key] = value
        else:
            ignored.append(key)

    merged["applied_hyperparameters"] = applied
    if ignored:
        merged["ignored_hyperparameters"] = sorted(set(ignored))
    if isinstance(raw, dict) and raw.get("tuning_id"):
        merged["tuning_id"] = raw.get("tuning_id")
    return merged


class TrainingCancelledError(RuntimeError):
    pass


def _ensure_training_cancel_registry(app):
    registry = getattr(app.state, "training_cancel_registry", None)
    if registry is None:
        registry = {}
        app.state.training_cancel_registry = registry
    lock = getattr(app.state, "training_cancel_registry_lock", None)
    if lock is None:
        lock = threading.Lock()
        app.state.training_cancel_registry_lock = lock
    return registry, lock


def register_training_run(app, run_id: str) -> str:
    resolved = str(run_id or uuid.uuid4().hex)
    registry, lock = _ensure_training_cancel_registry(app)
    with lock:
        registry[resolved] = False
    return resolved


def cancel_training_run(app, run_id: str) -> bool:
    if not run_id:
        return False
    registry, lock = _ensure_training_cancel_registry(app)
    with lock:
        if run_id not in registry:
            return False
        registry[run_id] = True
    return True


def is_training_cancelled(app, run_id: str) -> bool:
    if not run_id:
        return False
    registry, lock = _ensure_training_cancel_registry(app)
    with lock:
        return bool(registry.get(run_id, False))


def clear_training_run(app, run_id: str) -> None:
    if not run_id:
        return
    registry, lock = _ensure_training_cancel_registry(app)
    with lock:
        registry.pop(run_id, None)


class CancellableDataLoader:
    def __init__(self, dataloader, should_cancel, run_id: str):
        self._dataloader = dataloader
        self._should_cancel = should_cancel
        self._run_id = run_id

    def __iter__(self):
        for batch in self._dataloader:
            if self._should_cancel():
                raise TrainingCancelledError(
                    f"Training cancelled for run_id={self._run_id}"
                )
            yield batch

    def __len__(self):
        return len(self._dataloader)

    def __getattr__(self, name):
        return getattr(self._dataloader, name)


def make_cancellable_dataloader(dataloader, should_cancel, run_id: str):
    return CancellableDataLoader(
        dataloader=dataloader, should_cancel=should_cancel, run_id=run_id
    )


def split_train_val_test(
    x: np.ndarray,
    y: np.ndarray,
    val_size: float,
    test_size: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    test_size_combined = val_size + test_size
    if test_size_combined <= 0.0 or test_size_combined >= 1.0:
        raise ValueError("val_size + test_size must be in the range (0, 1).")

    x_train, x_tmp, y_train, y_tmp = train_test_split(
        x, y, test_size=test_size_combined, random_state=random_state
    )
    relative_test_size = test_size / test_size_combined
    x_val, x_test, y_val, y_test = train_test_split(
        x_tmp, y_tmp, test_size=relative_test_size, random_state=random_state
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def split_train_val_test_temporal(
    x: np.ndarray,
    y: np.ndarray,
    val_size: float,
    test_size: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Chronological split for temporal modeling.

    Keeps order intact and trims test data from the end of the series to avoid
    temporal leakage.
    """
    test_size_combined = val_size + test_size
    if test_size_combined <= 0.0 or test_size_combined >= 1.0:
        raise ValueError("val_size + test_size must be in the range (0, 1).")

    n_samples = int(x.shape[0])
    if n_samples != int(y.shape[0]):
        raise ValueError("x and y must have the same number of samples.")
    if n_samples < 3:
        raise ValueError("Need at least 3 samples to create train/val/test splits.")

    n_test = max(1, int(round(n_samples * test_size)))
    n_val = max(1, int(round(n_samples * val_size)))

    # Ensure at least one training sample remains.
    if n_test + n_val >= n_samples:
        n_val = max(1, n_val)
        n_test = max(1, n_samples - n_val - 1)
        if n_test < 1:
            raise ValueError("Not enough samples for requested temporal split sizes.")

    train_end = n_samples - n_val - n_test
    val_end = n_samples - n_test

    x_train = x[:train_end]
    y_train = y[:train_end]
    x_val = x[train_end:val_end]
    y_val = y[train_end:val_end]
    x_test = x[val_end:]
    y_test = y[val_end:]

    return x_train, x_val, x_test, y_train, y_val, y_test


def scale_splits(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    normalizer_type: str,
):
    x_normalizer, y_normalizer = get_normalizers(
        norm_type=normalizer_type, n_normalizers=2
    )

    # For temporal tensors [N, T, F], fit normalizer over flattened [N*T, F]
    if x_train.ndim == 3:
        x_train_fit = x_train.reshape(-1, x_train.shape[-1])
    else:
        x_train_fit = x_train

    x_normalizer.fit(x_train_fit)
    y_normalizer.fit(y_train)

    if x_train.ndim == 3:

        def _scale_x_3d(x_in: np.ndarray) -> np.ndarray:
            x_2d = x_in.reshape(-1, x_in.shape[-1])
            x_2d_scaled = x_normalizer.transform(x_2d)
            return x_2d_scaled.reshape(x_in.shape)

        x_train_scaled = _scale_x_3d(x_train)
        x_val_scaled = _scale_x_3d(x_val)
        x_test_scaled = _scale_x_3d(x_test)
    else:
        x_train_scaled = x_normalizer.transform(x_train)
        x_val_scaled = x_normalizer.transform(x_val)
        x_test_scaled = x_normalizer.transform(x_test)

    y_train_scaled = y_normalizer.transform(y_train)
    y_val_scaled = y_normalizer.transform(y_val)
    y_test_scaled = y_normalizer.transform(y_test)

    return (
        x_train_scaled,
        x_val_scaled,
        x_test_scaled,
        y_train_scaled,
        y_val_scaled,
        y_test_scaled,
        x_normalizer,
        y_normalizer,
    )


def make_dataloaders(
    x_train_scaled: np.ndarray,
    y_train_scaled: np.ndarray,
    x_val_scaled: np.ndarray,
    y_val_scaled: np.ndarray,
    batch_size: int,
    shuffle: bool,
):
    train_dataset = TensorDataset(
        torch.from_numpy(x_train_scaled).float(),
        torch.from_numpy(y_train_scaled).float(),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(x_val_scaled).float(),
        torch.from_numpy(y_val_scaled).float(),
    )

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader


def build_training_components(
    model,
    loss_function: str,
    optimizer_name: str,
    learning_rate: float,
    lr_scheduler: str,
):
    criterion = model.get_loss_fn(loss=loss_function, reduction="mean")
    optimizer = model.get_optimizer(optimizer_name, lr=learning_rate)

    scheduler = None
    if str(lr_scheduler).lower() != "none":
        scheduler = model.get_scheduler(
            lr_scheduler,
            optimizer,
            **SCHEDULER_KWARGS,
        )

    return criterion, optimizer, scheduler


def serialize_normalizer(normalizer):
    if isinstance(normalizer, IdentityScaler):
        return None
    return scaler_to_dict(normalizer)


async def store_trained_model(
    request,
    model,
    model_meta: Dict[str, Any],
    history: Dict[str, Any],
    x_normalizer=None,
    y_normalizer=None,
):
    from .trainers import attach_model_store

    if getattr(request.app.state, "model_store", None) is None:
        attach_model_store(request.app, max_items=10, ttl_seconds=3600)

    model_id = uuid.uuid4().hex
    await request.app.state.model_store.set(
        model_id,
        {
            "model": model,
            "model_meta": model_meta,
            "history": history,
            "x_normalizer": x_normalizer,
            "y_normalizer": y_normalizer,
        },
    )

    return model_id


def build_training_response(
    model_id: str,
    model_meta: Dict[str, Any],
    x_data,
    y_data,
    x_data_scaled,
    y_data_scaled,
    x_normalizer,
    y_normalizer,
    history: Dict[str, Any],
):
    return {
        "model_id": model_id,
        "model_meta": model_meta,
        "x_data": json.dumps([arr.tolist() for arr in x_data]),
        "y_data": json.dumps([arr.tolist() for arr in y_data]),
        "x_data_scaled": json.dumps([arr.tolist() for arr in x_data_scaled]),
        "y_data_scaled": json.dumps([arr.tolist() for arr in y_data_scaled]),
        "x_normalizer": serialize_normalizer(x_normalizer),
        "y_normalizer": serialize_normalizer(y_normalizer),
        "history": history,
    }
