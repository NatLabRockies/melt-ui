from __future__ import annotations

import base64
import io
import os
import uuid
from typing import Any

import numpy as np
import torch
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

router = APIRouter()

DEFAULT_SAM_CHECKPOINT = "facebook/sam-vit-base"
SAM_MODEL_PRESETS: dict[str, dict[str, Any]] = {
    "facebook/sam-vit-base": {
        "label": "SAM ViT Base",
        "family": "sam1",
        "model_class": "SamModel",
        "processor_class": "SamProcessor",
    },
    "facebook/sam-vit-large": {
        "label": "SAM ViT Large",
        "family": "sam1",
        "model_class": "SamModel",
        "processor_class": "SamProcessor",
    },
    "facebook/sam-vit-huge": {
        "label": "SAM ViT Huge",
        "family": "sam1",
        "model_class": "SamModel",
        "processor_class": "SamProcessor",
    },
    "facebook/sam2-hiera-tiny": {
        "label": "SAM 2 Hiera Tiny",
        "family": "sam2",
        "model_class": "Sam2Model",
        "processor_class": "Sam2Processor",
    },
    "facebook/sam2-hiera-small": {
        "label": "SAM 2 Hiera Small",
        "family": "sam2",
        "model_class": "Sam2Model",
        "processor_class": "Sam2Processor",
    },
    "facebook/sam2-hiera-base-plus": {
        "label": "SAM 2 Hiera Base Plus",
        "family": "sam2",
        "model_class": "Sam2Model",
        "processor_class": "Sam2Processor",
    },
    "facebook/sam2-hiera-large": {
        "label": "SAM 2 Hiera Large",
        "family": "sam2",
        "model_class": "Sam2Model",
        "processor_class": "Sam2Processor",
    },
    "facebook/sam2.1-hiera-tiny": {
        "label": "SAM 2.1 Hiera Tiny",
        "family": "sam2",
        "model_class": "Sam2Model",
        "processor_class": "Sam2Processor",
    },
    "facebook/sam2.1-hiera-small": {
        "label": "SAM 2.1 Hiera Small",
        "family": "sam2",
        "model_class": "Sam2Model",
        "processor_class": "Sam2Processor",
    },
    "facebook/sam2.1-hiera-base-plus": {
        "label": "SAM 2.1 Hiera Base Plus",
        "family": "sam2",
        "model_class": "Sam2Model",
        "processor_class": "Sam2Processor",
    },
    "facebook/sam2.1-hiera-large": {
        "label": "SAM 2.1 Hiera Large",
        "family": "sam2",
        "model_class": "Sam2Model",
        "processor_class": "Sam2Processor",
    },
    "syscv-community/sam-hq-vit-base": {
        "label": "SAM-HQ ViT Base",
        "family": "sam_hq",
        "model_class": "SamHQModel",
        "processor_class": "SamHQProcessor",
    },
}


def _get_sam_model_preset(checkpoint: str) -> dict[str, Any]:
    preset = SAM_MODEL_PRESETS.get(checkpoint)
    if preset is None:
        supported = ", ".join(SAM_MODEL_PRESETS.keys())
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported SAM checkpoint: {checkpoint}. Supported: {supported}",
        )
    return preset


def _get_transformers_class(class_name: str):
    try:
        import transformers
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Transformers SAM dependencies are missing. "
                "Install with `pip install transformers pillow`."
            ),
        ) from exc

    cls = getattr(transformers, class_name, None)
    if cls is None:
        raise HTTPException(
            status_code=500,
            detail=f"Installed transformers does not provide {class_name}",
        )
    return cls


def _ensure_sam_store(app) -> dict[str, Any]:
    store = getattr(app.state, "sam_store", None)
    if store is None:
        store = {}
        app.state.sam_store = store
    return store


def _decode_data_uri_image(data_uri: str):
    try:
        from PIL import Image
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Pillow is required for image processing. "
                "Install it with `pip install pillow`."
            ),
        ) from exc

    if not isinstance(data_uri, str) or not data_uri.startswith("data:image"):
        raise HTTPException(
            status_code=400, detail="image_data must be a data:image URI"
        )

    try:
        _, b64_part = data_uri.split(",", 1)
        raw = base64.b64decode(b64_part)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        return img
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail=f"Failed to decode image_data: {exc}"
        ) from exc


def _encode_rgb_to_data_uri(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _encode_jpeg_to_data_uri(img, quality: int = 90) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def _encode_mask_to_data_uri(mask: np.ndarray) -> str:
    try:
        from PIL import Image
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Pillow is required for image processing. "
                "Install it with `pip install pillow`."
            ),
        ) from exc

    m = mask.astype(np.uint8) * 255
    img = Image.fromarray(m, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _mask_color(index: int) -> np.ndarray:
    colors = [
        [0.0, 180.0, 255.0],
        [255.0, 99.0, 132.0],
        [75.0, 192.0, 192.0],
        [255.0, 206.0, 86.0],
        [153.0, 102.0, 255.0],
        [255.0, 159.0, 64.0],
        [46.0, 204.0, 113.0],
        [231.0, 76.0, 60.0],
    ]
    return np.array(colors[index % len(colors)], dtype=np.float32)


def _encode_colored_masks_to_data_uri(masks: np.ndarray) -> str:
    try:
        from PIL import Image
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Pillow is required for image processing. "
                "Install it with `pip install pillow`."
            ),
        ) from exc

    if masks.ndim == 2:
        masks = masks.reshape(1, masks.shape[0], masks.shape[1])

    colored = np.zeros((masks.shape[-2], masks.shape[-1], 3), dtype=np.float32)
    for i, mask in enumerate(masks):
        idx = mask.astype(bool)
        colored[idx] = _mask_color(i)

    img = Image.fromarray(np.clip(colored, 0, 255).astype(np.uint8), mode="RGB")
    return _encode_rgb_to_data_uri(img)


def _make_overlay(image_rgb: np.ndarray, mask: np.ndarray) -> str:
    try:
        from PIL import Image
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Pillow is required for image processing. "
                "Install it with `pip install pillow`."
            ),
        ) from exc

    overlay = image_rgb.astype(np.float32).copy()
    alpha = 0.45
    color = np.array([0.0, 180.0, 255.0], dtype=np.float32)
    idx = mask.astype(bool)
    overlay[idx] = (1.0 - alpha) * overlay[idx] + alpha * color
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    img = Image.fromarray(overlay, mode="RGB")
    return _encode_rgb_to_data_uri(img)


def _make_multi_mask_overlay(image_rgb: np.ndarray, masks: np.ndarray) -> str:
    try:
        from PIL import Image
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Pillow is required for image processing. "
                "Install it with `pip install pillow`."
            ),
        ) from exc

    if masks.ndim == 2:
        masks = masks.reshape(1, masks.shape[0], masks.shape[1])

    overlay = image_rgb.astype(np.float32).copy()
    alpha = 0.45
    for i, mask in enumerate(masks):
        idx = mask.astype(bool)
        color = _mask_color(i)
        overlay[idx] = (1.0 - alpha) * overlay[idx] + alpha * color

    img = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB")
    return _encode_rgb_to_data_uri(img)


def _resolve_device(device_pref: str | None) -> str:
    pref = (device_pref or "auto").strip().lower()
    if pref == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if pref in {"cpu", "cuda", "mps"}:
        return pref
    raise HTTPException(status_code=400, detail=f"Unsupported device: {device_pref}")


@router.get("/sam_health")
async def sam_health():
    """Check if SAM dependencies are installed and available."""
    diagnostics = {
        "transformers_available": False,
        "pillow_available": False,
        "torch_available": False,
        "cuda_available": False,
        "errors": [],
    }

    try:
        import transformers

        diagnostics["transformers_available"] = True
        diagnostics["transformers_version"] = transformers.__version__
        diagnostics["sam_classes"] = {
            name: hasattr(transformers, name)
            for name in [
                "SamModel",
                "SamProcessor",
                "Sam2Model",
                "Sam2Processor",
                "SamHQModel",
                "SamHQProcessor",
            ]
        }
    except Exception as e:
        diagnostics["errors"].append(f"transformers: {e}")

    try:
        import PIL

        diagnostics["pillow_available"] = True
        diagnostics["pillow_version"] = PIL.__version__
    except Exception as e:
        diagnostics["errors"].append(f"pillow: {e}")

    try:
        import torch

        diagnostics["torch_available"] = True
        diagnostics["torch_version"] = torch.__version__
        diagnostics["cuda_available"] = torch.cuda.is_available()
        if diagnostics["cuda_available"]:
            diagnostics["cuda_device"] = torch.cuda.get_device_name(0)
    except Exception as e:
        diagnostics["errors"].append(f"torch: {e}")

    all_ok = (
        diagnostics["transformers_available"]
        and diagnostics["pillow_available"]
        and diagnostics["torch_available"]
    )

    return {
        "ok": all_ok,
        "diagnostics": diagnostics,
    }


@router.post("/sam_load_model")
async def sam_load_model(request: Request):
    body = await request.json()

    checkpoint = str(body.get("checkpoint") or DEFAULT_SAM_CHECKPOINT).strip()
    if not checkpoint:
        raise HTTPException(status_code=400, detail="checkpoint is required")
    preset = _get_sam_model_preset(checkpoint)
    family = str(preset.get("family") or "sam1")

    device = _resolve_device(body.get("device"))

    print(
        f"[SAM] Loading model: checkpoint={checkpoint}, family={family}, device={device}"
    )

    store = _ensure_sam_store(request.app)

    existing_id = None
    for sid, entry in store.items():
        if (
            entry.get("checkpoint") == checkpoint
            and entry.get("device") == device
            and entry.get("family", "sam1") == family
            and entry.get("kind") == "sam_model"
        ):
            existing_id = sid
            break

    if existing_id is not None:
        print(f"[SAM] Model cache hit: {existing_id}")
        return {
            "model": {
                "model_id": existing_id,
                "checkpoint": checkpoint,
                "device": device,
                "family": family,
                "label": preset.get("label") or checkpoint,
                "experimental": bool(preset.get("experimental", False)),
                "kind": "sam_model",
            },
            "status": "cached",
        }

    try:
        processor_class = _get_transformers_class(str(preset["processor_class"]))
        model_class = _get_transformers_class(str(preset["model_class"]))
        print(f"[SAM] Loading processor from {checkpoint}...")
        processor = processor_class.from_pretrained(checkpoint)
        print(f"[SAM] Processor loaded. Loading model from {checkpoint}...")
        model = model_class.from_pretrained(checkpoint)
        print(f"[SAM] Model loaded. Moving to device={device}...")
        model.to(device)
        model.eval()
        print("[SAM] Model on device and in eval mode.")
    except Exception as exc:
        print(f"[SAM] Load error: {exc}")
        raise HTTPException(
            status_code=500, detail=f"Failed to load SAM model: {exc}"
        ) from exc

    model_id = str(uuid.uuid4())
    store[model_id] = {
        "kind": "sam_model",
        "model": model,
        "processor": processor,
        "checkpoint": checkpoint,
        "device": device,
        "family": family,
        "label": preset.get("label") or checkpoint,
        "experimental": bool(preset.get("experimental", False)),
    }
    print(f"[SAM] Model stored with id={model_id}")

    return {
        "model": {
            "model_id": model_id,
            "checkpoint": checkpoint,
            "device": device,
            "family": family,
            "label": preset.get("label") or checkpoint,
            "experimental": bool(preset.get("experimental", False)),
            "kind": "sam_model",
        },
        "status": "loaded",
    }


@router.post("/sam_load_image")
async def sam_load_image(request: Request):
    body = await request.json()
    path = str(body.get("path") or "").strip()
    image_data = body.get("image_data")

    if not path and not image_data:
        raise HTTPException(status_code=400, detail="Provide either path or image_data")

    try:
        from PIL import Image
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Pillow is required for image processing. "
                "Install it with `pip install pillow`."
            ),
        ) from exc

    if path:
        if not os.path.exists(path):
            raise HTTPException(
                status_code=400, detail=f"Image path does not exist: {path}"
            )
        try:
            img = Image.open(path).convert("RGB")
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Failed to open image: {exc}"
            ) from exc
        uri = _encode_jpeg_to_data_uri(img)
        source_path = path
    else:
        img = _decode_data_uri_image(str(image_data))
        uri = _encode_jpeg_to_data_uri(img)
        source_path = ""

    width, height = img.size
    return {
        "image": {
            "image_data": uri,
            "width": int(width),
            "height": int(height),
            "source_path": source_path,
        }
    }


def _normalize_points(points_raw: Any, width: int, height: int):
    out_coords: list[list[float]] = []
    out_labels: list[int] = []
    if points_raw is None:
        return out_coords, out_labels

    if not isinstance(points_raw, list):
        raise HTTPException(status_code=400, detail="points must be a list")

    for p in points_raw:
        if not isinstance(p, dict):
            raise HTTPException(status_code=400, detail="Each point must be an object")
        x = float(p.get("x"))
        y = float(p.get("y"))
        label = int(p.get("label", 1))
        if label not in (0, 1):
            raise HTTPException(status_code=400, detail="Point label must be 0 or 1")
        if x < 0 or y < 0 or x >= width or y >= height:
            raise HTTPException(
                status_code=400,
                detail=f"Point ({x}, {y}) is outside image bounds ({width}, {height})",
            )
        out_coords.append([x, y])
        out_labels.append(label)

    return out_coords, out_labels


def _normalize_boxes(boxes_raw: Any, width: int, height: int):
    out: list[list[float]] = []
    if boxes_raw is None:
        return out
    if not isinstance(boxes_raw, list):
        raise HTTPException(status_code=400, detail="boxes must be a list")

    for b in boxes_raw:
        if not isinstance(b, dict):
            raise HTTPException(status_code=400, detail="Each box must be an object")

        x1 = float(b.get("x1"))
        y1 = float(b.get("y1"))
        x2 = float(b.get("x2"))
        y2 = float(b.get("y2"))

        xa, xb = sorted([x1, x2])
        ya, yb = sorted([y1, y2])
        xa = max(0.0, min(float(width - 1), xa))
        xb = max(0.0, min(float(width - 1), xb))
        ya = max(0.0, min(float(height - 1), ya))
        yb = max(0.0, min(float(height - 1), yb))

        if xb <= xa or yb <= ya:
            raise HTTPException(
                status_code=400, detail="Box has zero area after normalization"
            )

        out.append([xa, ya, xb, yb])

    return out


@router.post("/sam_segment")
async def sam_segment(request: Request):
    body = await request.json()
    model_ref = body.get("model") or {}
    image_ref = body.get("image") or {}
    prompts = body.get("prompts") or {}
    multimask_output = bool(body.get("multimask_output", True))

    model_id = None
    if isinstance(model_ref, dict):
        model_id = model_ref.get("model_id")
    elif isinstance(model_ref, str):
        model_id = model_ref

    if not model_id:
        raise HTTPException(status_code=400, detail="Missing model.model_id")

    image_data = None
    if isinstance(image_ref, dict):
        image_data = image_ref.get("image_data")
    elif isinstance(image_ref, str):
        image_data = image_ref

    if not image_data:
        raise HTTPException(status_code=400, detail="Missing image.image_data")

    store = _ensure_sam_store(request.app)
    entry = store.get(model_id)
    if not entry or entry.get("kind") != "sam_model":
        raise HTTPException(status_code=404, detail=f"SAM model not found: {model_id}")

    model = entry.get("model")
    processor = entry.get("processor")
    device = entry.get("device")
    family = str(entry.get("family") or "sam1")

    pil_img = _decode_data_uri_image(str(image_data))
    width, height = pil_img.size
    image_np = np.array(pil_img)

    points_raw = prompts.get("points") if isinstance(prompts, dict) else None
    boxes_raw = prompts.get("boxes") if isinstance(prompts, dict) else None

    points, labels = _normalize_points(points_raw, width, height)
    boxes = _normalize_boxes(boxes_raw, width, height)

    if len(points) == 0 and len(boxes) == 0:
        raise HTTPException(
            status_code=400,
            detail="At least one prompt is required (point or box)",
        )

    processor_kwargs = {
        "images": pil_img,
        "return_tensors": "pt",
    }
    if family == "sam2":
        if points:
            processor_kwargs["input_points"] = [[points]]
            processor_kwargs["input_labels"] = [[labels]]
        if boxes:
            processor_kwargs["input_boxes"] = [boxes]
    else:
        if points:
            processor_kwargs["input_points"] = [points]
            processor_kwargs["input_labels"] = [labels]
        if boxes:
            processor_kwargs["input_boxes"] = [boxes]

    try:
        inputs = processor(**processor_kwargs)

        def _run_inference(inference_inputs):
            kwargs = {"multimask_output": multimask_output}
            with torch.no_grad():
                return model(**inference_inputs, **kwargs)

        # SAM1 via HuggingFace has known MPS compatibility gaps; fall back to CPU.
        if device == "mps":

            def _to_device(tensor_dict, dev):
                """Move tensors to device, casting floats to float32 (MPS doesn't support float64)."""
                out = {}
                for k, v in tensor_dict.items():
                    if isinstance(v, torch.Tensor):
                        if v.is_floating_point():
                            v = v.to(torch.float32)
                        v = v.to(dev)
                    out[k] = v
                return out

            if device == "mps":
                try:
                    inputs_mps = _to_device(inputs, "mps")
                    model.to("mps")
                    outputs = _run_inference(inputs_mps)
                    inputs = inputs_mps
                except Exception as mps_exc:
                    print(f"[SAM] MPS inference failed ({mps_exc}), retrying on CPU...")
                    model.to("cpu")
                    inputs = _to_device(inputs, "cpu")
                    device = "cpu"
                    outputs = _run_inference(inputs)
            else:
                inputs = _to_device(inputs, device)
                outputs = _run_inference(inputs)
    except Exception as exc:
        print(f"[SAM] Inference error: {exc}")
        raise HTTPException(
            status_code=500, detail=f"SAM inference failed: {exc}"
        ) from exc

    try:
        if family == "sam2":
            masks_list = processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                binarize=True,
            )
        else:
            masks_list = processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu(),
                binarize=True,
            )
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Mask post-processing failed: {exc}"
        ) from exc

    masks_tensor = masks_list[0]
    iou_scores = outputs.iou_scores.detach().cpu().numpy()
    # SAM scores and masks may include prompt and multimask dimensions; flatten them together.
    scores = iou_scores.flatten().tolist()

    if masks_tensor.ndim == 2:
        masks_tensor = masks_tensor.unsqueeze(0)
    elif masks_tensor.ndim >= 3:
        masks_tensor = masks_tensor.reshape(
            -1,
            masks_tensor.shape[-2],
            masks_tensor.shape[-1],
        )
    else:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected mask tensor shape: {tuple(masks_tensor.shape)}",
        )

    masks_np = masks_tensor.detach().cpu().numpy().astype(np.uint8)
    mask_count = int(masks_np.shape[0])

    if mask_count == 0:
        return JSONResponse(
            {
                "masks": [],
                "scores": [],
                "best_index": -1,
                "best_mask_image": "",
                "overlay_image": "",
                "shape": {
                    "num_masks": 0,
                    "height": int(height),
                    "width": int(width),
                },
            }
        )

    if not scores:
        scores = [0.0 for _ in range(mask_count)]
    elif len(scores) > mask_count:
        scores = scores[:mask_count]
    elif len(scores) < mask_count:
        scores.extend([0.0 for _ in range(mask_count - len(scores))])

    best_idx = int(np.argmax(np.asarray(scores, dtype=np.float32)))
    best_mask = masks_np[best_idx]
    if multimask_output and mask_count > 1:
        mask_image_uri = _encode_colored_masks_to_data_uri(masks_np)
        overlay_uri = _make_multi_mask_overlay(image_np, masks_np)
    else:
        mask_image_uri = _encode_mask_to_data_uri(best_mask)
        overlay_uri = _make_overlay(image_np, best_mask)

    # JSON payload contains masks for downstream numeric processing.
    masks_serialized = [m.astype(int).tolist() for m in masks_np]

    return JSONResponse(
        {
            "masks": masks_serialized,
            "scores": [float(s) for s in scores],
            "best_index": best_idx,
            "best_mask_image": mask_image_uri,
            "overlay_image": overlay_uri,
            "shape": {
                "num_masks": int(len(masks_serialized)),
                "height": int(height),
                "width": int(width),
            },
        }
    )


def _ensure_dino_store(app) -> dict[str, Any]:
    store = getattr(app.state, "dino_store", None)
    if store is None:
        store = {}
        app.state.dino_store = store
    return store


def _resolve_dino_device(device_pref: str | None = None) -> str:
    pref = (device_pref or "auto").strip().lower()
    if pref == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if pref in {"cpu", "cuda", "mps"}:
        return pref
    raise HTTPException(status_code=400, detail=f"Unsupported device: {device_pref}")


def _get_dino_model(request: Request, checkpoint: str, device: str):
    store = _ensure_dino_store(request.app)
    cache_key = f"{checkpoint}:{device}"
    entry = store.get(cache_key)
    if entry and entry.get("kind") == "grounded_dino":
        return entry["processor"], entry["model"], cache_key, True

    try:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Transformers Grounded DINO dependencies are missing. "
                "Install with `pip install transformers pillow`."
            ),
        ) from exc

    try:
        processor = AutoProcessor.from_pretrained(checkpoint)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(checkpoint)
        model.to(device)
        model.eval()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load Grounded DINO model '{checkpoint}': {exc}",
        ) from exc

    store[cache_key] = {
        "kind": "grounded_dino",
        "processor": processor,
        "model": model,
        "checkpoint": checkpoint,
        "device": device,
    }
    return processor, model, cache_key, False


@router.post("/grounded_dino_detect")
async def grounded_dino_detect(request: Request):
    """Detect objects in an image using text prompts via Grounded DINO."""
    body = await request.json()
    image_ref = body.get("image") or {}
    text_prompt = str(body.get("text_prompt") or "").strip()
    box_threshold = float(body.get("box_threshold", 0.3))
    text_threshold = float(body.get("text_threshold", 0.25))
    checkpoint = str(
        body.get("checkpoint") or "IDEA-Research/grounding-dino-tiny"
    ).strip()
    device = _resolve_dino_device(body.get("device"))

    image_data = None
    if isinstance(image_ref, dict):
        image_data = image_ref.get("image_data")
    elif isinstance(image_ref, str):
        image_data = image_ref

    if not image_data:
        raise HTTPException(status_code=400, detail="Missing image.image_data")

    if not text_prompt:
        raise HTTPException(status_code=400, detail="text_prompt is required")

    pil_img = _decode_data_uri_image(str(image_data))
    width, height = pil_img.size

    try:
        import supervision as sv
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Grounded DINO dependencies are missing. "
                "Install with `pip install transformers supervision pillow`."
            ),
        ) from exc

    try:
        processor, model, cache_key, cache_hit = _get_dino_model(
            request, checkpoint, device
        )
        if cache_hit:
            print(f"[Grounded DINO] Model cache hit: {cache_key}")

        inputs = processor(images=pil_img, text=text_prompt, return_tensors="pt")
        if device == "mps":
            inputs = inputs.to("mps")
        elif device == "cuda":
            inputs = inputs.to("cuda")
        else:
            inputs = inputs.to("cpu")

        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs.input_ids,
            threshold=box_threshold,
            text_threshold=text_threshold,
            target_sizes=[(height, width)],
        )

        result = results[0] if results else {}
        boxes = result.get("boxes")
        scores_tensor = result.get("scores")
        text_labels = result.get("text_labels") or result.get("labels") or []

        detections = sv.Detections(
            xyxy=(
                boxes.detach().cpu().numpy() if boxes is not None else np.zeros((0, 4))
            ),
            confidence=(
                scores_tensor.detach().cpu().numpy()
                if scores_tensor is not None
                else np.zeros((0,), dtype=np.float32)
            ),
            class_id=np.zeros(len(boxes) if boxes is not None else 0, dtype=int),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Grounded DINO detection failed: {exc}",
        ) from exc

    normalized_boxes = []
    scores = []
    labels = []

    if hasattr(detections, "xyxy") and detections.xyxy is not None:
        for i, box in enumerate(detections.xyxy):
            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            normalized_boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2})
            if hasattr(detections, "confidence") and detections.confidence is not None:
                scores.append(float(detections.confidence[i]))
            else:
                scores.append(1.0)
            if i < len(text_labels):
                labels.append(str(text_labels[i]))
            else:
                labels.append(text_prompt)

    return JSONResponse(
        {
            "boxes": normalized_boxes,
            "scores": scores,
            "labels": labels,
            "text_prompt": text_prompt,
            "checkpoint": checkpoint,
            "device": device,
            "num_detections": len(normalized_boxes),
            "shape": {"height": int(height), "width": int(width)},
        }
    )
