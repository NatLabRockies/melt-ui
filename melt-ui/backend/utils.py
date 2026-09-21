import base64
import importlib
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np


class _WireIdentityScaler:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

    def inverse_transform(self, X):
        return X

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self


def _to_jsonable(val):
    """Convert different Python types to JSONable versions."""
    # numpy scalar
    if isinstance(val, np.generic):
        return val.item()

    # numpy array
    if isinstance(val, np.ndarray):
        return val.tolist()

    # lists/tuples
    if isinstance(val, (list, tuple)):
        return [_to_jsonable(v) for v in val]

    # dicts
    if isinstance(val, dict):
        return {str(k): _to_jsonable(v) for k, v in val.items()}

    # sets
    if isinstance(val, set):
        return [_to_jsonable(v) for v in val]

    # fallback: try numpy coercion
    try:
        arr = np.asarray(val)
        # object arrays can still fail .tolist() sometimes, but usually ok
        return arr.tolist()
    except Exception:
        # last resort: try primitive or stringify
        if isinstance(val, (str, int, float, bool)) or val is None:
            return val
        return repr(val)


def _from_jsonable(val):
    """
    Convert JSON-friendly values back to numpy where appropriate.
    """
    if isinstance(val, list):
        return np.asarray(val)
    if isinstance(val, dict):
        # Could be nested scaler dict or plain dict; caller decides.
        return {k: _from_jsonable(v) for k, v in val.items()}
    return val


# Private attributes that some sklearn transformers need at runtime.
_PRIVATE_ATTR_ALLOWLIST = {
    "PowerTransformer": ["_scaler"],
}


def scaler_to_dict(scaler):
    """Serialize a sklearn scaler/transformer to a JSON-serializable dict."""
    if scaler is None:
        return None
    data = {
        "class": scaler.__class__.__name__,
        "module": scaler.__module__,
        "params": scaler.get_params(deep=True) if hasattr(scaler, "get_params") else {},
        "attributes": {},
    }
    # for attr in ("mean_", "scale_", "var_", "n_samples_seen_", "feature_names_in_"):
    #     if hasattr(scaler, attr):
    #         val = getattr(scaler, attr)
    #         # convert numpy scalars/arrays to Python lists / scalars
    #         try:
    #             arr = np.asarray(val)
    #             data["attributes"][attr] = arr.tolist()
    #         except Exception:
    #             data["attributes"][attr] = val

    # automatically get attributes for each scaler type class (diff methods, diff attr)
    # for name, val in vars(scaler).items():
    #     if not name.endswith("_") or not name.startswith("_"):
    #         continue
    #     if callable(val):
    #         continue
    #     data["attributes"][name] = _to_jsonable(val)
    for name, val in vars(scaler).items():
        if not name.endswith("_"):
            continue
        if name.startswith("__"):
            continue
        if callable(val):
            continue
        data["attributes"][name] = _to_jsonable(val)

    # Save required private internals (recursively)
    for name in _PRIVATE_ATTR_ALLOWLIST.get(data["class"], []):
        if hasattr(scaler, name):
            inner = getattr(scaler, name)
            # If it's an sklearn-like object, recurse; else jsonable
            if hasattr(inner, "__class__") and hasattr(inner, "__module__"):
                data["attributes"][name] = scaler_to_dict(inner)
            else:
                data["attributes"][name] = _to_jsonable(inner)

    return data


def scaler_from_dict(d):
    """Reconstruct a sklearn scaler/transformer from dict produced by scaler_to_dict."""
    if d is None or d == {}:
        return None

    if isinstance(d, dict) and d.get("class") == "IdentityScaler":
        return _WireIdentityScaler()

    module = importlib.import_module(d["module"])
    cls = getattr(module, d["class"])
    obj = cls(**d.get("params", {}))
    attrs = d.get("attributes", {}) or {}

    for attr, val in attrs.items():
        # Nested scaler dict (e.g., PowerTransformer._scaler)
        if isinstance(val, dict) and "class" in val and "module" in val:
            setattr(obj, attr, scaler_from_dict(val))
        else:
            setattr(obj, attr, _from_jsonable(val))
    return obj


def convert_fig_to_image(fig, image_format="pdf"):

    # Serialize to chosen format in-memory
    fmt = (image_format or "png").lower()
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
