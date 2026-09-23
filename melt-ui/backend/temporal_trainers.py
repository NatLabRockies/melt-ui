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
    split_train_val_test_temporal,
    store_trained_model,
)

router = APIRouter()


def _validate_temporal_lengths(lengths_raw, sample_count, sequence_length):
    if lengths_raw is None:
        return np.full(sample_count, sequence_length, dtype=np.int64)

    raw = np.asarray(lengths_raw)
    if raw.ndim != 1 or raw.shape[0] != sample_count:
        raise ValueError("lengths must contain one value per temporal sample.")
    if raw.dtype == np.bool_:
        raise ValueError("lengths values must be integers.")

    try:
        numeric = raw.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("lengths values must be numeric integers.") from exc

    if not np.all(np.isfinite(numeric)):
        raise ValueError("lengths values must be finite integers.")
    if not np.all(numeric == np.floor(numeric)):
        raise ValueError("lengths values must be integers.")

    lengths = numeric.astype(np.int64)
    if np.any(lengths < 1) or np.any(lengths > sequence_length):
        raise ValueError("lengths values must be between 1 and seq_length.")

    return lengths


def _split_temporal_lengths(lengths, x_train, x_val, x_test):
    n_train = int(x_train.shape[0])
    n_val = int(x_val.shape[0])
    n_test = int(x_test.shape[0])

    if n_train + n_val + n_test != int(lengths.shape[0]):
        raise ValueError("Temporal length split does not match data split.")

    train_lengths = lengths[:n_train]
    val_lengths = lengths[n_train : n_train + n_val]
    test_lengths = lengths[n_train + n_val :]

    return train_lengths, val_lengths, test_lengths


def prepare_sequences_from_raw(x_raw, y_raw, seq_length, seq_to_one):
    """
    Prepare sequences from raw 2D data using sliding window approach.
    Similar to the notebook implementation in lines 677-710.

    Args:
        x_raw: 2D array of shape [samples, features]
        y_raw: 2D array of shape [samples, targets]
        seq_length: Length of input sequences
        seq_to_one: Whether to use seq-to-one prediction (target at end of sequence)

    Returns:
        x_sequences: 3D array of shape [num_sequences, seq_length, features]
        y_targets: 2D array of shape [num_sequences, targets]
    """
    x_raw = np.asarray(x_raw)
    y_raw = np.asarray(y_raw)

    if x_raw.ndim != 2:
        raise ValueError(f"x_raw should be 2D, got shape {x_raw.shape}")
    if y_raw.ndim == 1:
        y_raw = y_raw.reshape(-1, 1)

    x_sequences = []
    y_targets = []

    # Create sequences using sliding window
    for i in range(len(x_raw) - seq_length + 1):
        x_seq = x_raw[i : i + seq_length]
        if seq_to_one:
            # For seq-to-one prediction, target is at the end of the sequence
            y_target = y_raw[i + seq_length - 1]
        else:
            # For seq-to-seq prediction, target is the next sequence
            y_target = y_raw[i : i + seq_length]

        x_sequences.append(x_seq)
        y_targets.append(y_target)

    x_sequences = np.stack(x_sequences)
    y_targets = np.stack(y_targets)

    return x_sequences, y_targets


@router.post("/melt_temporal_supervised_trainer")
async def melt_temporal_supervised_trainer(request: Request):
    try:
        import importlib

        RecurrentNeuralNetwork = importlib.import_module(
            "ptmelt.models"
        ).RecurrentNeuralNetwork
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Could not import ptmelt RecurrentNeuralNetwork: {e}"},
        )

    body = await request.json()
    body = apply_hyperparameter_overrides(body, "temporal_rnn")
    requested_run_id = str(body.get("training_run_id") or "").strip() or None

    x_raw = body.get("x")
    y_raw = body.get("y")
    if x_raw is None:
        x_raw = body.get("x_seq")
    if y_raw is None:
        y_raw = body.get("y_seq")

    if x_raw is None or y_raw is None:
        return JSONResponse(
            status_code=400,
            content={
                "error": "Missing temporal inputs. Expected x/y (or x_seq/y_seq) for temporal supervised training."
            },
        )

    x = np.asarray(x_raw)
    y = np.asarray(y_raw)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

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
    activation_function = body.get("activation_function", "relu")
    output_activation = body.get("output_activation", "linear")
    width = int(body.get("width", 32))
    depth = int(body.get("depth", 2))
    rnn_type = str(body.get("rnn_type", "lstm")).lower()
    head_type = str(body.get("head_type", "last")).lower()
    l1_reg = float(body.get("l1_reg", 0.0))
    l2_reg = float(body.get("l2_reg", 0.0))
    lr_scheduler = body.get("lr_scheduler", "ReduceLROnPlateau")
    num_mixtures = int(body.get("num_mixtures", 0))
    seq_length = int(body.get("seq_length", 60))
    seq_to_one = bool(body.get("seq_to_one", True))

    if not seq_to_one:
        return JSONResponse(
            status_code=400,
            content={
                "error": (
                    "seq_to_one=false is not supported by the current "
                    "PT-MELT temporal models."
                )
            },
        )
    suffix_crop = bool(body.get("suffix_crop", False))
    suffix_crop_min_length = int(body.get("suffix_crop_min_length", 32))

    if seq_length < 1:
        return JSONResponse(
            status_code=400,
            content={"error": "seq_length must be >= 1."},
        )

    if suffix_crop_min_length < 1:
        return JSONResponse(
            status_code=400,
            content={"error": "suffix_crop_min_length must be >= 1."},
        )

    if suffix_crop and suffix_crop_min_length > seq_length:
        return JSONResponse(
            status_code=400,
            content={
                "error": (
                    "suffix_crop_min_length must be <= seq_length when suffix_crop "
                    "is enabled. "
                    f"Got suffix_crop_min_length={suffix_crop_min_length}, "
                    f"seq_length={seq_length}."
                )
            },
        )

    if x.shape[0] != y.shape[0]:
        return JSONResponse(
            status_code=400,
            content={
                "error": (
                    "x and y must have the same number of samples. "
                    f"Got x={x.shape[0]} and y={y.shape[0]}."
                )
            },
        )

    # Check if we need to prepare sequences from raw data
    prepare_sequences = x.ndim == 2

    # These are assigned in either path below.
    x_train = x_val = x_test = None
    y_train = y_val = y_test = None
    x_train_scaled = x_val_scaled = x_test_scaled = None
    y_train_scaled = y_val_scaled = y_test_scaled = None
    train_lengths = val_lengths = test_lengths = None

    if prepare_sequences:
        # Input is raw 2D data [samples, features] - split first, then scale, then window each split.
        # Notebook-aligned temporal split on raw arrays.
        x_train_raw, x_val_raw, x_test_raw, y_train_raw, y_val_raw, y_test_raw = (
            split_train_val_test_temporal(
                x=x,
                y=y,
                val_size=val_size,
                test_size=test_size,
            )
        )

        # Ensure each split has enough samples to build at least one sequence.
        split_lengths = {
            "train": int(x_train_raw.shape[0]),
            "val": int(x_val_raw.shape[0]),
            "test": int(x_test_raw.shape[0]),
        }
        too_short = [k for k, v in split_lengths.items() if v < seq_length]
        if too_short:
            return JSONResponse(
                status_code=400,
                content={
                    "error": (
                        "Not enough samples in split(s) for sequence construction with "
                        f"seq_length={seq_length}. "
                        f"Split lengths: {split_lengths}."
                    )
                },
            )

        # Scale raw splits using train statistics only.
        (
            x_train_raw_scaled,
            x_val_raw_scaled,
            x_test_raw_scaled,
            y_train_raw_scaled,
            y_val_raw_scaled,
            y_test_raw_scaled,
            x_normalizer,
            y_normalizer,
        ) = scale_splits(
            x_train=x_train_raw,
            x_val=x_val_raw,
            x_test=x_test_raw,
            y_train=y_train_raw,
            y_val=y_val_raw,
            y_test=y_test_raw,
            normalizer_type=normalizer_type,
            x_train_lengths=train_lengths,
            x_val_lengths=val_lengths,
            x_test_lengths=test_lengths,
        )

        # Build notebook-style sequences independently per split.
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

        train_lengths = np.full(
            x_train_scaled.shape[0],
            seq_length,
            dtype=np.int64,
        )
        val_lengths = np.full(
            x_val_scaled.shape[0],
            seq_length,
            dtype=np.int64,
        )
        test_lengths = np.full(
            x_test_scaled.shape[0],
            seq_length,
            dtype=np.int64,
        )

    elif x.ndim != 3:
        return JSONResponse(
            status_code=400,
            content={
                "error": (
                    "Temporal trainer expects x with shape "
                    "[samples, timesteps, features]. "
                    f"Received shape={list(x.shape)}"
                )
            },
        )

    else:
        # Pre-windowed 3D sequences path.
        if x.shape[1] != seq_length:
            return JSONResponse(
                status_code=400,
                content={
                    "error": (
                        "Input sequence length mismatch for pre-windowed data. "
                        f"Received timesteps={x.shape[1]}, expected seq_length={seq_length}."
                    )
                },
            )

        try:
            lengths = _validate_temporal_lengths(
                body.get("lengths"),
                sample_count=x.shape[0],
                sequence_length=seq_length,
            )
        except ValueError as exc:
            return JSONResponse(
                status_code=400,
                content={"error": str(exc)},
            )

        x_train, x_val, x_test, y_train, y_val, y_test = split_train_val_test_temporal(
            x=x,
            y=y,
            val_size=val_size,
            test_size=test_size,
        )

        train_lengths, val_lengths, test_lengths = _split_temporal_lengths(
            lengths,
            x_train,
            x_val,
            x_test,
        )

        # Pre-windowed sequences can overlap across split boundaries.
        # Drop a boundary buffer to better emulate split-then-window semantics.
        boundary_gap = max(0, seq_length - 1)
        if boundary_gap > 0:
            if x_val.shape[0] <= boundary_gap or x_test.shape[0] <= boundary_gap:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": (
                            "Not enough pre-windowed sequences after applying boundary "
                            f"gap of {boundary_gap}. Reduce seq_length or provide raw 2D input."
                        )
                    },
                )
            x_val = x_val[boundary_gap:]
            y_val = y_val[boundary_gap:]
            val_lengths = val_lengths[boundary_gap:]

            x_test = x_test[boundary_gap:]
            y_test = y_test[boundary_gap:]
            test_lengths = test_lengths[boundary_gap:]

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
            x_train_lengths=train_lengths,
            x_val_lengths=val_lengths,
            x_test_lengths=test_lengths,
        )

    if suffix_crop and np.any(train_lengths < suffix_crop_min_length):
        return JSONResponse(
            status_code=400,
            content={
                "error": (
                    "suffix_crop_min_length cannot exceed any training "
                    "sequence length."
                )
            },
        )

    run_id = register_training_run(request.app, requested_run_id)

    train_dataloader, val_dataloader = make_dataloaders(
        x_train_scaled=x_train_scaled,
        y_train_scaled=y_train_scaled,
        x_val_scaled=x_val_scaled,
        y_val_scaled=y_val_scaled,
        batch_size=batch_size,
        shuffle=shuffle,
        train_lengths=train_lengths,
        val_lengths=val_lengths,
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

    model = RecurrentNeuralNetwork(
        num_features=x_train_scaled.shape[2],
        num_outputs=y_train_scaled.shape[1] if y_train_scaled.ndim > 1 else 1,
        width=width,
        depth=depth,
        rnn_type=rnn_type,
        head_type=head_type,
        act_fun=activation_function,
        output_activation=output_activation,
        dropout=dropout_rate,
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

    try:
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
            suffix_crop=suffix_crop,
            min_length=suffix_crop_min_length,
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

    history = model.history
    model_meta = {
        "architecture": "rnn",
        "width": width,
        "depth": depth,
        "rnn_type": rnn_type,
        "head_type": head_type,
        "activation_function": activation_function,
        "output_activation": output_activation,
        "dropout_rate": dropout_rate,
        "l1_reg": l1_reg,
        "l2_reg": l2_reg,
        "num_mixtures": num_mixtures,
        "random_state": random_state,
        "split_mode": "temporal_ordered",
        "seq_length": seq_length,
        "seq_to_one": seq_to_one,
        "suffix_crop": suffix_crop,
        "suffix_crop_min_length": suffix_crop_min_length,
        "sequence_preparation": prepare_sequences,
        "x_data_is_scaled": False,
        "y_data_is_scaled": False,
        "x_data_scaled_available": True,
        "y_data_scaled_available": True,
    }
    if body.get("applied_hyperparameters"):
        model_meta["applied_hyperparameters"] = body["applied_hyperparameters"]
    if body.get("tuning_id"):
        model_meta["tuning_id"] = body["tuning_id"]

    model_meta["window_order"] = (
        "split_then_scale_then_window"
        if prepare_sequences
        else "prewindowed_split_then_scale"
    )

    model_meta["has_lengths"] = True
    model_meta["variable_lengths"] = bool(
        np.any(train_lengths != seq_length)
        or np.any(val_lengths != seq_length)
        or np.any(test_lengths != seq_length)
    )
    model_meta["lengths_note"] = (
        "Sequence lengths are carried through scaling and batching; "
        "padded timesteps are excluded from feature normalization."
    )

    if not prepare_sequences:
        model_meta["prewindow_overlap_trim"] = max(0, seq_length - 1)

    model_id = await store_trained_model(
        request=request,
        model=model,
        model_meta=model_meta,
        history=history,
        x_normalizer=x_normalizer,
        y_normalizer=y_normalizer,
        lengths_data=(train_lengths, val_lengths, test_lengths),
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


@router.post("/melt_temporal_transformer_trainer")
async def melt_temporal_transformer_trainer(request: Request):
    try:
        import importlib

        TemporalTransformerNetwork = importlib.import_module(
            "ptmelt.models"
        ).TemporalTransformerNetwork
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Could not import ptmelt TemporalTransformerNetwork: {e}"
            },
        )

    body = await request.json()
    body = apply_hyperparameter_overrides(body, "temporal_transformer")
    requested_run_id = str(body.get("training_run_id") or "").strip() or None

    x_raw = body.get("x")
    y_raw = body.get("y")
    if x_raw is None:
        x_raw = body.get("x_seq")
    if y_raw is None:
        y_raw = body.get("y_seq")

    if x_raw is None or y_raw is None:
        return JSONResponse(
            status_code=400,
            content={
                "error": "Missing temporal inputs. Expected x/y (or x_seq/y_seq) for temporal transformer training."
            },
        )

    x = np.asarray(x_raw)
    y = np.asarray(y_raw)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

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
    activation_function = body.get("activation_function", "relu")
    output_activation = body.get("output_activation", "linear")
    width = int(body.get("width", 64))
    depth = int(body.get("depth", 2))
    head_type = str(body.get("head_type", "last")).lower()
    num_heads = int(body.get("num_heads", 4))
    ff_dim_raw = body.get("ff_dim", 0)
    ff_dim = int(ff_dim_raw) if ff_dim_raw not in (None, "") else 0
    max_seq_len = int(body.get("max_seq_len", 2048))
    use_causal_mask = bool(body.get("use_causal_mask", False))
    l1_reg = float(body.get("l1_reg", 0.0))
    l2_reg = float(body.get("l2_reg", 0.0))
    lr_scheduler = body.get("lr_scheduler", "ReduceLROnPlateau")
    num_mixtures = int(body.get("num_mixtures", 0))
    seq_length = int(body.get("seq_length", 60))
    seq_to_one = bool(body.get("seq_to_one", True))

    if not seq_to_one:
        return JSONResponse(
            status_code=400,
            content={
                "error": (
                    "seq_to_one=false is not supported by the current "
                    "PT-MELT temporal models."
                )
            },
        )

    if seq_length < 1:
        return JSONResponse(
            status_code=400,
            content={"error": "seq_length must be >= 1."},
        )

    if num_heads < 1:
        return JSONResponse(
            status_code=400,
            content={"error": "num_heads must be >= 1."},
        )

    if width < 1 or depth < 1:
        return JSONResponse(
            status_code=400,
            content={"error": "width and depth must be >= 1."},
        )

    if width % num_heads != 0:
        return JSONResponse(
            status_code=400,
            content={
                "error": (
                    "Transformer constraint violation: width must be divisible by "
                    f"num_heads. Got width={width}, num_heads={num_heads}."
                )
            },
        )

    if ff_dim < 0:
        return JSONResponse(
            status_code=400,
            content={"error": "ff_dim must be >= 0 (0 means auto/default)."},
        )

    if max_seq_len < seq_length:
        return JSONResponse(
            status_code=400,
            content={
                "error": (
                    "max_seq_len must be >= seq_length. "
                    f"Got max_seq_len={max_seq_len}, seq_length={seq_length}."
                )
            },
        )

    if x.shape[0] != y.shape[0]:
        return JSONResponse(
            status_code=400,
            content={
                "error": (
                    "x and y must have the same number of samples. "
                    f"Got x={x.shape[0]} and y={y.shape[0]}."
                )
            },
        )

    prepare_sequences = x.ndim == 2

    x_train = x_val = x_test = None
    y_train = y_val = y_test = None
    x_train_scaled = x_val_scaled = x_test_scaled = None
    y_train_scaled = y_val_scaled = y_test_scaled = None
    train_lengths = val_lengths = test_lengths = None

    if prepare_sequences:
        x_train_raw, x_val_raw, x_test_raw, y_train_raw, y_val_raw, y_test_raw = (
            split_train_val_test_temporal(
                x=x,
                y=y,
                val_size=val_size,
                test_size=test_size,
            )
        )

        split_lengths = {
            "train": int(x_train_raw.shape[0]),
            "val": int(x_val_raw.shape[0]),
            "test": int(x_test_raw.shape[0]),
        }
        too_short = [k for k, v in split_lengths.items() if v < seq_length]
        if too_short:
            return JSONResponse(
                status_code=400,
                content={
                    "error": (
                        "Not enough samples in split(s) for sequence construction with "
                        f"seq_length={seq_length}. "
                        f"Split lengths: {split_lengths}."
                    )
                },
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
        ) = scale_splits(
            x_train=x_train_raw,
            x_val=x_val_raw,
            x_test=x_test_raw,
            y_train=y_train_raw,
            y_val=y_val_raw,
            y_test=y_test_raw,
            normalizer_type=normalizer_type,
            x_train_lengths=train_lengths,
            x_val_lengths=val_lengths,
            x_test_lengths=test_lengths,
        )

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

        train_lengths = np.full(
            x_train_scaled.shape[0],
            seq_length,
            dtype=np.int64,
        )
        val_lengths = np.full(
            x_val_scaled.shape[0],
            seq_length,
            dtype=np.int64,
        )
        test_lengths = np.full(
            x_test_scaled.shape[0],
            seq_length,
            dtype=np.int64,
        )

    elif x.ndim != 3:
        return JSONResponse(
            status_code=400,
            content={
                "error": (
                    "Temporal transformer trainer expects x with shape "
                    "[samples, timesteps, features]. "
                    f"Received shape={list(x.shape)}"
                )
            },
        )

    else:
        try:
            lengths = _validate_temporal_lengths(
                body.get("lengths"),
                sample_count=x.shape[0],
                sequence_length=seq_length,
            )
        except ValueError as exc:
            return JSONResponse(
                status_code=400,
                content={"error": str(exc)},
            )

        if x.shape[1] != seq_length:
            return JSONResponse(
                status_code=400,
                content={
                    "error": (
                        "Input sequence length mismatch for pre-windowed data. "
                        f"Received timesteps={x.shape[1]}, expected seq_length={seq_length}."
                    )
                },
            )

        x_train, x_val, x_test, y_train, y_val, y_test = split_train_val_test_temporal(
            x=x,
            y=y,
            val_size=val_size,
            test_size=test_size,
        )

        train_lengths, val_lengths, test_lengths = _split_temporal_lengths(
            lengths,
            x_train,
            x_val,
            x_test,
        )

        boundary_gap = max(0, seq_length - 1)
        if boundary_gap > 0:
            if x_val.shape[0] <= boundary_gap or x_test.shape[0] <= boundary_gap:
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": (
                            "Not enough pre-windowed sequences after applying boundary "
                            f"gap of {boundary_gap}. Reduce seq_length or provide raw 2D input."
                        )
                    },
                )
            x_val = x_val[boundary_gap:]
            y_val = y_val[boundary_gap:]
            val_lengths = val_lengths[boundary_gap:]

            x_test = x_test[boundary_gap:]
            y_test = y_test[boundary_gap:]
            test_lengths = test_lengths[boundary_gap:]

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
            x_train_lengths=train_lengths,
            x_val_lengths=val_lengths,
            x_test_lengths=test_lengths,
        )

    run_id = register_training_run(request.app, requested_run_id)

    train_dataloader, val_dataloader = make_dataloaders(
        x_train_scaled=x_train_scaled,
        y_train_scaled=y_train_scaled,
        x_val_scaled=x_val_scaled,
        y_val_scaled=y_val_scaled,
        batch_size=batch_size,
        shuffle=shuffle,
        train_lengths=train_lengths,
        val_lengths=val_lengths,
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

    model = TemporalTransformerNetwork(
        num_features=x_train_scaled.shape[2],
        num_outputs=y_train_scaled.shape[1] if y_train_scaled.ndim > 1 else 1,
        width=width,
        depth=depth,
        num_heads=num_heads,
        ff_dim=None if ff_dim == 0 else ff_dim,
        max_seq_len=max_seq_len,
        head_type=head_type,
        use_causal_mask=use_causal_mask,
        act_fun=activation_function,
        output_activation=output_activation,
        dropout=dropout_rate,
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

    try:
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

    history = model.history
    model_meta = {
        "architecture": "temporal_transformer",
        "width": width,
        "depth": depth,
        "head_type": head_type,
        "num_heads": num_heads,
        "ff_dim": ff_dim,
        "max_seq_len": max_seq_len,
        "use_causal_mask": use_causal_mask,
        "activation_function": activation_function,
        "output_activation": output_activation,
        "dropout_rate": dropout_rate,
        "l1_reg": l1_reg,
        "l2_reg": l2_reg,
        "num_mixtures": num_mixtures,
        "random_state": random_state,
        "split_mode": "temporal_ordered",
        "seq_length": seq_length,
        "seq_to_one": seq_to_one,
        "sequence_preparation": prepare_sequences,
        "x_data_is_scaled": False,
        "y_data_is_scaled": False,
        "x_data_scaled_available": True,
        "y_data_scaled_available": True,
    }
    if body.get("applied_hyperparameters"):
        model_meta["applied_hyperparameters"] = body["applied_hyperparameters"]
    if body.get("tuning_id"):
        model_meta["tuning_id"] = body["tuning_id"]

    model_meta["window_order"] = (
        "split_then_scale_then_window"
        if prepare_sequences
        else "prewindowed_split_then_scale"
    )

    model_meta["has_lengths"] = True
    model_meta["variable_lengths"] = bool(
        np.any(train_lengths != seq_length)
        or np.any(val_lengths != seq_length)
        or np.any(test_lengths != seq_length)
    )
    model_meta["lengths_note"] = (
        "Sequence lengths are carried through scaling and batching; "
        "padded timesteps are excluded from feature normalization."
    )

    if not prepare_sequences:
        model_meta["prewindow_overlap_trim"] = max(0, seq_length - 1)

    model_id = await store_trained_model(
        request=request,
        model=model,
        model_meta=model_meta,
        history=history,
        x_normalizer=x_normalizer,
        y_normalizer=y_normalizer,
        lengths_data=(train_lengths, val_lengths, test_lengths),
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
