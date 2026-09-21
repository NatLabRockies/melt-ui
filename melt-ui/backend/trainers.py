import asyncio
import base64
import inspect
import io
import time
from collections import OrderedDict

import numpy as np
import torch
from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse
from ptmelt.models import (
    ArtificialNeuralNetwork,
    BayesianNeuralNetwork,
    ResidualNeuralNetwork,
)

from .training_common import (
    TrainingCancelledError,
    apply_hyperparameter_overrides,
    build_training_components,
    build_training_response,
    cancel_training_run,
    clear_training_run,
    is_training_cancelled,
    make_cancellable_dataloader,
    make_dataloaders,
    register_training_run,
    scale_splits,
    split_train_val_test,
    store_trained_model,
)

router = APIRouter()


class ModelStore:
    def __init__(self, max_items: int = 5, ttl_seconds: int = 3600):
        self.max_items = int(max_items)
        self.ttl = int(ttl_seconds)
        self.store = OrderedDict()  # model_id -> entry dict
        self._lock = asyncio.Lock()

    async def set(self, model_id: str, entry: dict):
        async with self._lock:
            now = time.time()
            entry.setdefault("created_at", now)
            entry["last_access"] = now
            self.store[model_id] = entry
            self.store.move_to_end(model_id)
            await self._evict_if_needed()

    async def get(self, model_id: str):
        async with self._lock:
            entry = self.store.get(model_id)
            if not entry:
                return None
            entry["last_access"] = time.time()
            self.store.move_to_end(model_id)
            return entry

    async def delete(self, model_id: str):
        async with self._lock:
            self.store.pop(model_id, None)

    async def _evict_if_needed(self):
        now = time.time()
        # remove expired by TTL
        expired = [
            k
            for k, v in self.store.items()
            if (now - v.get("last_access", v.get("created_at", now))) > self.ttl
        ]
        for k in expired:
            self.store.pop(k, None)
        # evict oldest until within max_items
        while len(self.store) > self.max_items:
            self.store.popitem(last=False)

    async def cleanup_loop(self, interval_seconds: int = 60):
        while True:
            await asyncio.sleep(interval_seconds)
            async with self._lock:
                await self._evict_if_needed()


def attach_model_store(
    app: FastAPI,
    *,
    max_items: int = 10,
    ttl_seconds: int = 3600,
    cleanup_interval: int = 60,
):
    """
    Call this once from your main FastAPI app setup (before serving) to attach a model store and
    start a background cleanup task.

    Example in your main app:
      from backend.trainers import attach_model_store
      attach_model_store(app, max_items=20, ttl_seconds=3600)
    """
    if getattr(app.state, "model_store", None) is None:
        ms = ModelStore(max_items=max_items, ttl_seconds=ttl_seconds)
        app.state.model_store = ms

        # start background cleanup on startup
        @app.on_event("startup")
        async def _start_model_store_cleanup():
            # fire-and-forget cleanup loop
            asyncio.create_task(ms.cleanup_loop(interval_seconds=cleanup_interval))


@router.post("/melt_supervised_trainer")
async def melt_supervised_trainer(request: Request):
    body = await request.json()
    body = apply_hyperparameter_overrides(body, "static")
    run_id = register_training_run(
        request.app,
        str(body.get("training_run_id") or "").strip() or None,
    )

    try:
        # unpack settings from node
        x_raw = body.get("x")
        if x_raw is None:
            return JSONResponse(
                status_code=400,
                content={"error": "Missing x input for supervised training."},
            )

        x = np.asarray(x_raw)
        model_architecture = str(body.get("model_architecture", "ann")).lower()
        if model_architecture == "vae":
            return JSONResponse(
                status_code=400,
                content={
                    "error": "VAE training moved to /melt_vae_trainer. Use the dedicated VAE Trainer node."
                },
            )
        if model_architecture == "rnn":
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Temporal RNN training moved to /melt_temporal_supervised_trainer. Use the dedicated temporal trainer node."
                },
            )

        y_raw = body.get("y")
        y = np.asarray(y_raw) if y_raw is not None else None

        if y is None:
            return JSONResponse(
                status_code=400,
                content={"error": "Missing y input for supervised architectures."},
            )

        if x.ndim != 2:
            return JSONResponse(
                status_code=400,
                content={
                    "error": (
                        "Supervised x input must have shape [samples, features]. "
                        f"Received shape={list(x.shape)}."
                    )
                },
            )

        if y.ndim == 1:
            y = y.reshape(-1, 1)
        elif y.ndim != 2:
            return JSONResponse(
                status_code=400,
                content={"error": "Supervised y input must be 1D or 2D."},
            )

        if x.shape[0] != y.shape[0]:
            return JSONResponse(
                status_code=400,
                content={"error": "x and y must have matching sample counts."},
            )

        val_size = float(body.get("val_size", 0.1))
        test_size = float(body.get("test_size", 0.1))
        normalizer_type = body.get("norm_type", "none")
        random_state = int(body.get("random_state", 42))
        shuffle = bool(body.get("shuffle", True))
        batch_size = int(body.get("batch_size", 32))
        num_epochs = int(body.get("num_epochs", 10))
        loss_function = body.get("loss_function", "mse")
        optimizer = body.get("optimizer", "Adam")
        learning_rate = float(body.get("learning_rate", 1e-3))
        dropout_rate = float(body.get("dropout_rate", 0.0))
        batch_norm = bool(body.get("batch_norm", False))
        activation_function = body.get("activation_function", "relu")
        output_activation = body.get("output_activation", "linear")
        node_list = body.get("node_list", [])
        l1_reg = float(body.get("l1_reg", 0.0))
        l2_reg = float(body.get("l2_reg", 0.0))
        lr_scheduler = body.get("lr_scheduler", "ReduceLROnPlateau")
        num_mixtures = int(body.get("num_mixtures", 0))

        model_meta = {
            "architecture": model_architecture,
            "node_list": node_list,
            "activation_function": activation_function,
            "output_activation": output_activation,
            "dropout_rate": dropout_rate,
            "batch_norm": batch_norm,
            "l1_reg": l1_reg,
            "l2_reg": l2_reg,
            "num_mixtures": num_mixtures,
            "random_state": random_state,
            "x_data_is_scaled": False,
            "y_data_is_scaled": False,
            "x_data_scaled_available": True,
            "y_data_scaled_available": True,
        }
        if body.get("applied_hyperparameters"):
            model_meta["applied_hyperparameters"] = body["applied_hyperparameters"]
        if body.get("tuning_id"):
            model_meta["tuning_id"] = body["tuning_id"]

        x_train, x_val, x_test, y_train, y_val, y_test = split_train_val_test(
            x=x,
            y=y,
            val_size=val_size,
            test_size=test_size,
            random_state=random_state,
        )
        (
            x_train_scaled,
            x_val_scaled,
            x_test_scaled,
            y_train_scaled,
            y_val_scaled,
            y_test_scaled,
            x_normalizer,
            y_normalizer,
        ) = scale_splits(
            x_train=x_train,
            x_val=x_val,
            x_test=x_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            normalizer_type=normalizer_type,
        )

        train_dataloader, val_dataloader = make_dataloaders(
            x_train_scaled=x_train_scaled,
            y_train_scaled=y_train_scaled,
            x_val_scaled=x_val_scaled,
            y_val_scaled=y_val_scaled,
            batch_size=batch_size,
            shuffle=shuffle,
        )
        should_cancel = lambda: is_training_cancelled(request.app, run_id)
        train_dataloader = make_cancellable_dataloader(
            train_dataloader,
            should_cancel,
            run_id,
        )
        val_dataloader = make_cancellable_dataloader(
            val_dataloader,
            should_cancel,
            run_id,
        )

        # Create the MELT model based on architecture
        if model_architecture == "ann":
            model = ArtificialNeuralNetwork(
                num_features=x.shape[1],
                num_outputs=y.shape[1] if y.ndim > 1 else 1,
                node_list=node_list,
                act_fun=activation_function,
                output_activation=output_activation,
                dropout=dropout_rate,
                batch_norm=batch_norm,
                l1_reg=l1_reg,
                l2_reg=l2_reg,
                num_mixtures=num_mixtures,
                seed=random_state,
            )
        elif model_architecture == "resnet":
            model = ResidualNeuralNetwork(
                num_features=x.shape[1],
                num_outputs=y.shape[1] if y.ndim > 1 else 1,
                node_list=node_list,
                act_fun=activation_function,
                output_activation=output_activation,
                dropout=dropout_rate,
                batch_norm=batch_norm,
                l1_reg=l1_reg,
                l2_reg=l2_reg,
                num_mixtures=num_mixtures,
                seed=random_state,
            )
        elif model_architecture == "bnn":
            model = BayesianNeuralNetwork(
                num_features=x.shape[1],
                num_outputs=y.shape[1] if y.ndim > 1 else 1,
                node_list=node_list,
                act_fun=activation_function,
                output_activation=output_activation,
                dropout=dropout_rate,
                batch_norm=batch_norm,
                l1_reg=l1_reg,
                l2_reg=l2_reg,
                num_mixtures=num_mixtures,
                seed=random_state,
            )
        else:
            return JSONResponse(
                status_code=400,
                content={
                    "error": f"Unsupported model architecture: {model_architecture}"
                },
            )

        # Build the model and set up scheduler
        model.build()

        criterion, optimizer, scheduler = build_training_components(
            model=model,
            loss_function=loss_function,
            optimizer_name=optimizer,
            learning_rate=learning_rate,
            lr_scheduler=lr_scheduler,
        )

        # Train in a worker thread so the event loop can still handle cancel requests.
        await asyncio.to_thread(
            model.fit,
            train_dl=train_dataloader,
            val_dl=val_dataloader,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            stopping=True,
            num_epochs=num_epochs,
            verbose=True,
        )

        # Prepare response
        history = model.history
        x_data = (x_train, x_val, x_test)
        y_data = (y_train, y_val, y_test)

        x_data_scaled = (x_train_scaled, x_val_scaled, x_test_scaled)
        y_data_scaled = (y_train_scaled, y_val_scaled, y_test_scaled)

        model_id = await store_trained_model(
            request=request,
            model=model,
            model_meta=model_meta,
            history=history,
            x_normalizer=x_normalizer,
            y_normalizer=y_normalizer,
        )

        return JSONResponse(
            content=build_training_response(
                model_id=model_id,
                model_meta=model_meta,
                x_data=x_data,
                y_data=y_data,
                x_data_scaled=x_data_scaled,
                y_data_scaled=y_data_scaled,
                x_normalizer=x_normalizer,
                y_normalizer=y_normalizer,
                history=history,
            )
        )
    except TrainingCancelledError:
        return JSONResponse(
            status_code=200,
            content={
                "cancelled": True,
                "run_id": run_id,
                "error": "Training cancelled by user.",
            },
        )
    finally:
        clear_training_run(request.app, run_id)


@router.post("/melt_cancel_training")
async def melt_cancel_training(request: Request):
    body = await request.json()
    run_id = str(body.get("run_id") or "").strip()
    if not run_id:
        return JSONResponse(status_code=400, content={"error": "Missing run_id"})
    cancelled = cancel_training_run(request.app, run_id)
    return JSONResponse(content={"ok": True, "run_id": run_id, "cancelled": cancelled})


@router.get("/models/{model_id}/state_dict")
async def get_model_state_dict(request: Request, model_id: str):
    model_store = getattr(request.app.state, "model_store", None)
    if model_store is None:
        return JSONResponse(
            status_code=404,
            content={"error": "Model store is not initialized."},
        )

    model_entry = model_store.get(model_id)
    if inspect.isawaitable(model_entry):
        model_entry = await model_entry

    if not model_entry:
        return JSONResponse(
            status_code=404,
            content={"error": f"Model ID {model_id} not found."},
        )

    model = model_entry.get("model")
    if model is None or not hasattr(model, "state_dict"):
        return JSONResponse(
            status_code=500,
            content={"error": f"Model ID {model_id} has no serializable model object."},
        )

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    state_dict_b64 = base64.b64encode(buffer.read()).decode("utf-8")

    return JSONResponse(
        content={
            "model_id": model_id,
            "state_b64": state_dict_b64,
            "meta": model_entry.get("model_meta", {}),
        }
    )
