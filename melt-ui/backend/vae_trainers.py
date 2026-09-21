import asyncio

import numpy as np
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from .training_common import (
    TrainingCancelledError,
    apply_hyperparameter_overrides,
    build_training_components,
    build_training_response,
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


@router.post("/melt_vae_trainer")
async def melt_vae_trainer(request: Request):
    try:
        import importlib

        VariationalAutoencoder = importlib.import_module(
            "ptmelt.models"
        ).VariationalAutoencoder
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Could not import ptmelt VariationalAutoencoder: {e}"},
        )

    body = await request.json()
    body = apply_hyperparameter_overrides(body, "vae")
    run_id = register_training_run(
        request.app,
        str(body.get("training_run_id") or "").strip() or None,
    )

    try:

        x_raw = body.get("x")
        if x_raw is None:
            return JSONResponse(status_code=400, content={"error": "Missing x input."})

        x = np.asarray(x_raw)
        if x.ndim < 2:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "VAE input x must be at least 2D [samples, features]."
                },
            )

        val_size = float(body.get("val_size", 0.1))
        test_size = float(body.get("test_size", 0.1))
        normalizer_type = body.get("norm_type", "none")
        random_state = int(body.get("random_state", 42))
        shuffle = bool(body.get("shuffle", True))
        batch_size = int(body.get("batch_size", 32))
        num_epochs = int(body.get("num_epochs", 100))
        loss_function = body.get("loss_function", "mse")
        optimizer_name = body.get("optimizer", "Adam")
        learning_rate = float(body.get("learning_rate", 1e-3))
        dropout_rate = float(body.get("dropout_rate", 0.0))
        batch_norm = bool(body.get("batch_norm", False))
        activation_function = body.get("activation_function", "relu")
        output_activation = body.get("output_activation", "linear")
        latent_dims = int(body.get("latent_dims", 8))
        encoder_node_list = body.get("encoder_node_list", [32, 16])
        decoder_node_list = body.get("decoder_node_list", [16, 32])
        l1_reg = float(body.get("l1_reg", 0.0))
        l2_reg = float(body.get("l2_reg", 0.0))
        lr_scheduler = body.get("lr_scheduler", "ReduceLROnPlateau")
        num_mixtures = max(int(body.get("num_mixtures", 1)), 1)

        if latent_dims < 1:
            return JSONResponse(
                status_code=400,
                content={"error": "latent_dims must be >= 1."},
            )

        y = x.copy()

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

        model = VariationalAutoencoder(
            num_features=x.shape[1],
            num_outputs=x.shape[1],
            latent_dims=latent_dims,
            encoder_node_list=encoder_node_list,
            decoder_node_list=decoder_node_list,
            act_fun=activation_function,
            output_activation=output_activation,
            dropout=dropout_rate,
            batch_norm=batch_norm,
            l1_reg=l1_reg,
            l2_reg=l2_reg,
            num_mixtures=num_mixtures,
            seed=random_state,
        )
        model.build()

        criterion, optimizer, scheduler = build_training_components(
            model=model,
            loss_function=loss_function,
            optimizer_name=optimizer_name,
            learning_rate=learning_rate,
            lr_scheduler=lr_scheduler,
        )

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

        history = model.history
        model_meta = {
            "architecture": "vae",
            "activation_function": activation_function,
            "output_activation": output_activation,
            "dropout_rate": dropout_rate,
            "batch_norm": batch_norm,
            "l1_reg": l1_reg,
            "l2_reg": l2_reg,
            "num_mixtures": num_mixtures,
            "random_state": random_state,
            "latent_dims": latent_dims,
            "encoder_node_list": encoder_node_list,
            "decoder_node_list": decoder_node_list,
            "is_self_supervised": True,
        }
        if body.get("applied_hyperparameters"):
            model_meta["applied_hyperparameters"] = body["applied_hyperparameters"]
        if body.get("tuning_id"):
            model_meta["tuning_id"] = body["tuning_id"]

        model_id = await store_trained_model(
            request=request,
            model=model,
            model_meta=model_meta,
            history=history,
            x_normalizer=x_normalizer,
            y_normalizer=y_normalizer,
        )

        x_data = (x_train, x_val, x_test)
        y_data = (y_train, y_val, y_test)
        x_data_scaled = (x_train_scaled, x_val_scaled, x_test_scaled)
        y_data_scaled = (y_train_scaled, y_val_scaled, y_test_scaled)

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
