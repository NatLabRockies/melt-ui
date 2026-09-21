// Base class for plot nodes
class PlotNodeBase extends LiteGraph.LGraphNode {
  constructor(title, endpointDefault, widgetAreaHeight = 100) {
    super();
    this.title = title;

    // Single image output
    this.addOutput("image", "string");

    // Shared properties
    this.addProperty("fig_width", 10, "number");
    this.addProperty("fig_height", 6, "number");
    this.addProperty("auto_update", true, "boolean");
    this.addProperty("endpoint", endpointDefault, "string");

    // Internal visual state
    this._image = null; // HTMLImageElement
    this._imageReady = false;
    this._error = "";
    this._fetching = false;
    this._lastRequestId = 0;
    this._widgetAreaHeight = widgetAreaHeight;
    this._currentUri = null; // data URI we expose on output
    this._objectURL = null;

    // Async coordination
    this._runner = new NodeRunner(this, 0);

    // Auto-update timer
    this._autoIntervalMs = 2000;
    this._autoTimer = null;

    // Widgets: auto_update toggle, Refresh, Open
    this._initWidgets();

    // Start auto-update if enabled
    if (this.properties.auto_update) this.startAutoUpdate();
  }

  _initWidgets() {
    const node = this;

    this.addWidget(
      "toggle",
      "auto_update",
      this.properties.auto_update,
      function (v) {
        node.properties.auto_update = v;
        if (v) node.startAutoUpdate();
        else node.stopAutoUpdate();
      },
    );

    this.addWidget("button", "Refresh", null, function () {
      if (node._runner) node._runner.invalidate();
      node.onExecute();
    });

    this.addWidget("button", "Open in new tab PDF", null, function () {
      // Prefer the PDF/full-image blob URL if present, fall back to other URLs
      const url =
        node._pdfObjectURL ||
        node._objectURL ||
        node._currentUri ||
        (node._image && node._image.src);
      if (url) window.open(url, "_blank");
    });
  }

  startAutoUpdate() {
    if (this._autoTimer) return;
    this._autoTimer = setInterval(() => {
      if (!this.properties.auto_update) return;
      if (this._fetching || this._currentPromise) return;

      // Let onExecute decide whether we actually need to fetch:
      this.onExecute();
    }, this._autoIntervalMs);
  }

  stopAutoUpdate() {
    if (this._autoTimer) {
      clearInterval(this._autoTimer);
      this._autoTimer = null;
    }
  }

  /**
   * Subclasses must implement:
   *   - resolveInput(): returns either a value or a Promise, OR null/undefined if no work.
   *   - buildPayload(resolvedInput): returns { endpoint, payload }.
   */

  resolveInput() {
    // to be overridden
    return null;
  }

  buildPayload(_resolvedInput) {
    throw new Error(
      "buildPayload(resolvedInput) must be implemented in subclass",
    );
  }

  onPropertyChanged(name, value, prevValue) {
    if (value !== prevValue && this._runner) {
      this._runner.invalidate();
    }

    if (name === "auto_update") {
      if (value) this.startAutoUpdate();
      else this.stopAutoUpdate();
    }
  }

  onExecute() {
    const maybeInput = this.resolveInput();

    // No data → just expose current image (if any) and bail
    if (maybeInput === null || maybeInput === undefined) {
      this.setOutputData(0, this._currentUri || null);
      return;
    }

    const runner = this._runner || (this._runner = new NodeRunner(this, 0));

    const fetcher = (resolvedInput) => {
      if (resolvedInput === null || resolvedInput === undefined) {
        this._error = "Missing input data";
        this._imageReady = false;
        this.setDirtyCanvas(true, true);
        return Promise.resolve(null);
      }

      const cfg = this.buildPayload(resolvedInput) || {};
      const endpoint = cfg.endpoint || this.properties.endpoint;
      const payload = cfg.payload;

      if (!endpoint) {
        this._error = "No endpoint configured";
        this._imageReady = false;
        this.setDirtyCanvas(true, true);
        return Promise.resolve(null);
      }

      if (!payload) {
        // buildPayload already set error if needed
        return Promise.resolve(null);
      }

      return this._requestPlot(endpoint, payload);
    };

    // const extractImage = () => this._currentUri || null;
    // extractor: return the image data URI (or objectURL) once the plot request finishes
    const extractImage = (res) =>
      res
        ? this._currentUri ||
          this._objectURL ||
          (this._image && this._image.src) ||
          null
        : null;

    runner.run(maybeInput, fetcher, extractImage, null);
  }

  // Generic HTTP + image loader using preview/image from backend
  _requestPlot(endpoint, payload) {
    this._fetching = true;
    this._error = "";
    this._imageReady = false;

    const requestId = ++this._lastRequestId;
    const node = this;

    return new Promise(async (resolve, reject) => {
      try {
        const response = await fetch(endpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });

        const text = await response.text();

        if (requestId !== node._lastRequestId) {
          resolve(null);
          return;
        }

        if (!response.ok) {
          node._error = `HTTP ${response.status}: ${text}`;
          node._image = null;
          node._imageReady = false;
          node._fetching = false;
          node.setDirtyCanvas(true, true);
          reject(new Error(node._error));
          return;
        }

        const json = text ? JSON.parse(text) : null;
        if (requestId !== node._lastRequestId) {
          resolve(null);
          return;
        }

        // const uri = (json && (json.preview || json.image)) || null;
        const previewUri = json && json.preview ? json.preview : null;
        const mainUri = json && json.image ? json.image : null;
        const displayUri = previewUri || mainUri || null;
        if (!displayUri) {
          node._error = "No image data in response";
          node._image = null;
          node._imageReady = false;
          node._fetching = false;
          node.setDirtyCanvas(true, true);
          reject(new Error(node._error));
          return;
        }

        // Clean up old object URLs if they exist
        if (node._objectURL) {
          try {
            URL.revokeObjectURL(node._objectURL);
          } catch (e) {}
          node._objectURL = null;
          node._objectURLMime = null;
        }
        if (node._pdfObjectURL) {
          try {
            URL.revokeObjectURL(node._pdfObjectURL);
          } catch (e) {}
          node._pdfObjectURL = null;
          node._pdfObjectURLMime = null;
        }

        // If server provided a data: URI for the main image (e.g. a PDF), create a blob URL we can open.
        try {
          if (typeof mainUri === "string" && mainUri.startsWith("data:")) {
            const m = mainUri.match(/^data:([^;]+);base64,(.*)$/);
            if (m) {
              const mime = m[1];
              const b64 = m[2];
              const bytes = atob(b64);
              const arr = new Uint8Array(bytes.length);
              for (let i = 0; i < bytes.length; i++)
                arr[i] = bytes.charCodeAt(i);
              const blob = new Blob([arr], { type: mime });
              node._pdfObjectURL = URL.createObjectURL(blob);
              node._pdfObjectURLMime = mime;
            } else {
              node._pdfObjectURL = mainUri;
            }
          } else if (mainUri) {
            // If it's an http(s) URL, store it directly
            node._pdfObjectURL = mainUri;
          }
        } catch (e) {
          node._pdfObjectURL = mainUri;
        }

        // For display, prefer the preview; if it's a data: URI turn it into a blob: URL for reliability.
        const uri = displayUri;
        const img = new Image();

        img.onload = function () {
          if (requestId !== node._lastRequestId) {
            resolve(null);
            return;
          }

          // Clean up old object URL if you ever change to Blob-based later
          if (node._objectURL) {
            try {
              URL.revokeObjectURL(node._objectURL);
            } catch (e) {}
          }

          node._image = img;
          node._imageReady = true;
          node._fetching = false;
          node._error = "";
          node._currentUri = uri;
          // node._objectURL = uri; // for now just mirror; easy swap later if you go back to Blob
          // If server returned a data: base64 URI, create a Blob URL so opening in a new tab works reliably
          try {
            if (typeof uri === "string" && uri.startsWith("data:")) {
              const m = uri.match(/^data:([^;]+);base64,(.*)$/);
              if (m) {
                const mime = m[1];
                const b64 = m[2];
                const bytes = atob(b64);
                const arr = new Uint8Array(bytes.length);
                for (let i = 0; i < bytes.length; i++)
                  arr[i] = bytes.charCodeAt(i);
                const blob = new Blob([arr], { type: mime });
                node._objectURL = URL.createObjectURL(blob);
                node._objectURLMime = mime;
              } else {
                node._objectURL = uri;
              }
            } else {
              node._objectURL = uri;
            }
          } catch (e) {
            // fallback to the original URI if anything goes wrong
            node._objectURL = uri;
          }

          node.setDirtyCanvas(true, true);
          resolve({ uri });
        };

        img.onerror = function () {
          if (requestId !== node._lastRequestId) {
            resolve(null);
            return;
          }
          node._error = "Failed to load image";
          node._image = null;
          node._imageReady = false;
          node._fetching = false;
          node.setDirtyCanvas(true, true);
          reject(new Error(node._error));
        };

        img.src = uri;
      } catch (err) {
        if (requestId !== node._lastRequestId) {
          resolve(null);
          return;
        }
        node._error = err.message || String(err);
        node._image = null;
        node._imageReady = false;
        node._fetching = false;
        node.setDirtyCanvas(true, true);
        reject(err);
      }
    });
  }

  // Shared drawing logic
  onDrawForeground(ctx) {
    if (this.flags && this.flags.collapsed) return;

    ctx.save();

    const margin = 6;
    const titleHeight = this._widgetAreaHeight;
    const x = margin;
    const y = titleHeight + margin;
    const w = this.size[0] - margin * 2;
    const h = this.size[1] - y - margin;

    ctx.fillStyle = "rgba(0,0,0,0.35)";
    ctx.fillRect(x, y, w, h);

    ctx.strokeStyle = "rgba(255,255,255,0.15)";
    ctx.strokeRect(x, y, w, h);

    if (this._imageReady && this._image) {
      const img = this._image;
      const imgRatio = img.width / img.height;
      const boxRatio = w / h;

      let drawW, drawH, offsetX, offsetY;
      if (imgRatio > boxRatio) {
        drawW = w;
        drawH = w / imgRatio;
        offsetX = x;
        offsetY = y + (h - drawH) / 2;
      } else {
        drawH = h;
        drawW = h * imgRatio;
        offsetX = x + (w - drawW) / 2;
        offsetY = y;
      }

      ctx.drawImage(img, offsetX, offsetY, drawW, drawH);
    } else {
      ctx.font = "12px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillStyle = "#DDD";

      let msg = "No image";
      if (this._fetching) msg = "Loading...";
      else if (this._error) msg = this._error;

      ctx.fillText(msg, x + w / 2, y + h / 2);
    }

    ctx.restore();
  }

  onAdded() {
    this.size = this.properties.size || [500, 500];
  }

  onRemoved() {
    this.stopAutoUpdate();
    if (this._objectURL) {
      try {
        URL.revokeObjectURL(this._objectURL);
      } catch (e) {}
      this._objectURL = null;
      this._objectURLMime = null;
    }
    if (this._pdfObjectURL) {
      try {
        URL.revokeObjectURL(this._pdfObjectURL);
      } catch (e) {}
      this._pdfObjectURL = null;
      this._pdfObjectURLMime = null;
    }
  }
}

class PlotXYNode extends PlotNodeBase {
  constructor() {
    super("Make XY Plot", "/plot_xy", 150); // slightly larger widget area

    // Inputs: x (features), y (target)
    this.addInput("x", "array");
    this.addInput("y", "array");

    // Extra property specific to XY plot
    this.addProperty("n_cols", 3, "number");

    this.addIntPropertyWidget("N Columns", "n_cols", {
      min: 0,
      default: 3,
    });

    // Add size property
    this.addProperty("size", [500, 500], "array");
  }

  // Turn node inputs into maybeInput for NodeRunner
  resolveInput() {
    const x = this.getInputData(0);
    const y = this.getInputData(1);

    const xIsPromise = x && typeof x.then === "function";
    const yIsPromise = y && typeof y.then === "function";

    // If both are missing and nothing connected, do nothing
    if ((x === undefined || x === null) && (y === undefined || y === null)) {
      return null;
    }

    // If either is still pending, let NodeRunner deal with the promise
    if (xIsPromise || yIsPromise) {
      return Promise.all([x, y]);
    }

    // Both must be present to proceed
    if (x === undefined || x === null || y === undefined || y === null) {
      return null;
    }

    return [x, y];
  }

  // Build payload for /plot_xy
  buildPayload(resolvedInput) {
    if (!Array.isArray(resolvedInput) || resolvedInput.length < 2) {
      this._error = "Invalid input pair for plot (need both x and y)";
      this._imageReady = false;
      this.setDirtyCanvas(true, true);
      return { endpoint: null, payload: null };
    }

    const rx = resolvedInput[0];
    const ry = resolvedInput[1];

    // Basic validation
    if (!Array.isArray(rx) || !Array.isArray(ry) || !rx.length || !ry.length) {
      this._error = "Missing or invalid x / y data";
      this._imageReady = false;
      this.setDirtyCanvas(true, true);
      return { endpoint: null, payload: null };
    }

    const endpoint = this.properties.endpoint || "/plot_xy";

    const payload = {
      x: rx,
      y: ry,
      figsize: [
        this.properties.fig_width || 10,
        this.properties.fig_height || 6,
      ],
      n_cols: this.properties.n_cols || 3,
      image_format: "pdf", // backend still returns preview PNG
    };

    return { endpoint, payload };
  }
}

PlotXYNode.title = "Make XY Plot";
PlotXYNode.desc = "Make XY plot and display image";

LiteGraph.registerNodeType("MELT/Plot/XYPlot", PlotXYNode);

class PlotHistoryNode extends PlotNodeBase {
  constructor() {
    super("Plot Training History", "/plot_history", 100);

    // Single input: history object
    this.addInput("history", "object");

    // Add size property
    this.addProperty("size", [500, 500], "array");
  }

  resolveInput() {
    const history = this.getInputData(0);
    const historyIsPromise = history && typeof history.then === "function";

    if (historyIsPromise) return history;

    if (history === undefined || history === null) {
      return null;
    }

    return history;
  }

  buildPayload(resolvedHistory) {
    if (!resolvedHistory) {
      this._error = "Missing history input";
      this._imageReady = false;
      this.setDirtyCanvas(true, true);
      return { endpoint: null, payload: null };
    }

    const endpoint = this.properties.endpoint || "/plot_history";

    const payload = {
      history: resolvedHistory,
      figsize: [
        this.properties.fig_width || 10,
        this.properties.fig_height || 6,
      ],
      image_format: "pdf", // backend returns preview PNG as well
    };

    return { endpoint, payload };
  }
}

PlotHistoryNode.title = "Plot Training History";
PlotHistoryNode.desc = "Plot training history metrics";

LiteGraph.registerNodeType("MELT/Plot/HistoryPlot", PlotHistoryNode);

class PlotEvaluationNode extends PlotNodeBase {
  constructor() {
    super("Plot Evaluation Metrics", "/evaluate_supervised_model", 150);

    // Inputs: model, x_data, y_data, y_normalizer
    this.addInput("model", "object");
    this.addInput("x_data", "object");
    this.addInput("y_data", "object");
    this.addInput("y_normalizer", "object");

    // Properties
    this.addProperty("unnormalize", true, "boolean");
    this.addProperty("evaluation_mode", "deterministic", "string");
    this.addProperty("ensemble_size", 100, "number");
    this.addProperty("output_index", -1, "number");

    // Add size property
    this.addProperty("size", [1000, 500], "array");

    // add endpoint property
    this.addProperty("endpoint", "/evaluate_supervised_model", "string");

    this.addBooleanPropertyWidget("Unnormalize Data", "unnormalize", {
      default: true,
    });
    this._evaluationModeWidget = this.addWidget(
      "combo",
      "Evaluation Mode",
      "Deterministic",
      (displayLabel) => {
        const label = String(displayLabel || "Deterministic");
        const modeMap = {
          Deterministic: "deterministic",
          "Aleatoric UQ": "aleatoric",
          "Epistemic UQ": "epistemic",
          "Total UQ": "total",
        };
        this.properties.evaluation_mode = modeMap[label] || "deterministic";
        this._syncEvaluationModeWidgets();
        if (this._runner) this._runner.invalidate();
      },
      {
        values: ["Deterministic", "Aleatoric UQ", "Epistemic UQ", "Total UQ"],
      },
    );
    this._ensembleSizeWidget = this.addIntPropertyWidget(
      "Ensemble Samples",
      "ensemble_size",
      {
        min: 2,
        default: 100,
      },
    );
    this.addIntPropertyWidget("Output Index", "output_index", {
      min: -1,
      default: -1,
    });
    this._syncEvaluationModeWidgets();
  }

  _syncEvaluationModeWidgets() {
    const mode = String(this.properties.evaluation_mode || "deterministic");
    const showEnsemble = mode === "epistemic" || mode === "total";
    if (this._ensembleSizeWidget) {
      this._ensembleSizeWidget.hidden = !showEnsemble;
    }
  }

  onPropertyChanged(name, value, prevValue) {
    super.onPropertyChanged(name, value, prevValue);
    if (name === "evaluation_mode") {
      this._syncEvaluationModeWidgets();
    }
  }

  _inferInputScaledFlag(inputIndex, fallback = false) {
    try {
      const input = this.inputs && this.inputs[inputIndex];
      if (!input || input.link == null || !this.graph || !this.graph.links) {
        return !!fallback;
      }
      const link = this.graph.links[input.link];
      if (!link) return !!fallback;

      const originId =
        link.origin_id !== undefined ? link.origin_id : link.origin;
      const originSlot =
        link.origin_slot !== undefined ? link.origin_slot : link.originSlot;
      const originNode = this.graph.getNodeById(originId);
      if (!originNode || !originNode.outputs || originSlot == null) {
        return !!fallback;
      }

      const outputInfo = originNode.outputs[originSlot];
      const outputName = outputInfo && outputInfo.name ? outputInfo.name : "";
      if (typeof outputName === "string") {
        const lowered = outputName.toLowerCase();
        if (lowered.includes("_scaled")) return true;
        if (lowered === "x_data" || lowered === "y_data") return false;
      }
    } catch (_err) {
      // fall through to fallback
    }

    return !!fallback;
  }

  // Turn node inputs into maybeInput for NodeRunner
  resolveInput() {
    const model = this.getInputData(0);
    const xData = this.getInputData(1);
    const yData = this.getInputData(2);
    const yNormalizer = this.getInputData(3);

    const modelIsPromise = model && typeof model.then === "function";
    const xIsPromise = xData && typeof xData.then === "function";
    const yIsPromise = yData && typeof yData.then === "function";
    const normIsPromise = yNormalizer && typeof yNormalizer.then === "function";

    // If all are missing and nothing connected, do nothing
    if (
      (model === undefined || model === null) &&
      (xData === undefined || xData === null) &&
      (yData === undefined || yData === null)
    ) {
      return null;
    }

    // If any is still pending, let NodeRunner deal with the promise
    if (modelIsPromise || xIsPromise || yIsPromise || normIsPromise) {
      return Promise.all([model, xData, yData, yNormalizer]);
    }

    // All must be present to proceed
    if (
      model === undefined ||
      model === null ||
      xData === undefined ||
      xData === null ||
      yData === undefined ||
      yData === null
    ) {
      return null;
    }

    return [model, xData, yData, yNormalizer];
  }

  // Build payload for /evaluate_supervised_model
  buildPayload(resolvedInput) {
    if (!Array.isArray(resolvedInput) || resolvedInput.length < 3) {
      this._error = "Invalid input for evaluation (need model, x, y)";
      this._imageReady = false;
      this.setDirtyCanvas(true, true);
      return { endpoint: null, payload: null };
    }

    const model = resolvedInput[0];
    const xData = resolvedInput[1];
    const yData = resolvedInput[2];
    const yNormalizer = resolvedInput[3];

    // Basic validation
    if (!model || !xData || !yData) {
      this._error = "Missing or invalid model / x / y data";
      this._imageReady = false;
      this.setDirtyCanvas(true, true);
      return { endpoint: null, payload: null };
    }

    const endpoint = this.properties.endpoint || "/evaluate_supervised_model";

    const normalizerPayload =
      yNormalizer && typeof yNormalizer === "object" ? yNormalizer : {};
    const hasNormalizer = Object.keys(normalizerPayload).length > 0;
    const shouldUnnormalize = !!this.properties.unnormalize && hasNormalizer;

    const payload = {
      model: model,
      x_data: xData,
      y_data: yData,
      y_normalizer: normalizerPayload,
      x_data_is_scaled_input: this._inferInputScaledFlag(1, false),
      y_data_is_scaled_input: this._inferInputScaledFlag(2, false),
      unnormalize: shouldUnnormalize,
      evaluation_mode: String(
        this.properties.evaluation_mode || "deterministic",
      ),
      ensemble_size: Math.max(
        2,
        parseInt(this.properties.ensemble_size || 100),
      ),
      figsize: [
        this.properties.fig_width || 10,
        this.properties.fig_height || 6,
      ],
      image_format: "pdf", // backend still returns preview PNG
    };

    const outputIndex = parseInt(this.properties.output_index, 10);
    if (Number.isFinite(outputIndex) && outputIndex >= 0) {
      payload.output_index = outputIndex;
    }

    return { endpoint, payload };
  }
}

PlotEvaluationNode.title = "Plot Model Evaluation";
PlotEvaluationNode.desc = "Plot model evaluation metrics";

LiteGraph.registerNodeType("MELT/Plot/EvaluationPlot", PlotEvaluationNode);

class PlotTemporalEvaluationNode extends PlotNodeBase {
  constructor() {
    super(
      "Plot Temporal Evaluation Metrics",
      "/evaluate_temporal_supervised_model",
      260,
    );

    // Inputs: model, x_data, y_data, y_normalizer
    this.addInput("model", "object");
    this.addInput("x_data", "object");
    this.addInput("y_data", "object");
    this.addInput("y_normalizer", "object");

    // Properties
    this.addProperty("unnormalize", true, "boolean");
    this.addProperty("output_index", 0, "number");
    this.addProperty("show_train", true, "boolean");
    this.addProperty("show_val", true, "boolean");
    this.addProperty("show_test", true, "boolean");
    this.addProperty("show_uncertainty", true, "boolean");
    this.addProperty("size", [1100, 550], "array");
    this.addProperty(
      "endpoint",
      "/evaluate_temporal_supervised_model",
      "string",
    );

    this.addBooleanPropertyWidget("Unnormalize Data", "unnormalize", {
      default: true,
    });
    this.addIntPropertyWidget("Output Index", "output_index", {
      min: 0,
      default: 0,
    });
    this.addBooleanPropertyWidget("Show Train", "show_train", {
      default: true,
    });
    this.addBooleanPropertyWidget("Show Validation", "show_val", {
      default: true,
    });
    this.addBooleanPropertyWidget("Show Test", "show_test", {
      default: true,
    });
    this.addBooleanPropertyWidget("Show Uncertainty", "show_uncertainty", {
      default: true,
    });
  }

  resolveInput() {
    const model = this.getInputData(0);
    const xData = this.getInputData(1);
    const yData = this.getInputData(2);
    const yNormalizer = this.getInputData(3);

    const modelIsPromise = model && typeof model.then === "function";
    const xIsPromise = xData && typeof xData.then === "function";
    const yIsPromise = yData && typeof yData.then === "function";
    const normIsPromise = yNormalizer && typeof yNormalizer.then === "function";

    if (
      (model === undefined || model === null) &&
      (xData === undefined || xData === null) &&
      (yData === undefined || yData === null)
    ) {
      return null;
    }

    if (modelIsPromise || xIsPromise || yIsPromise || normIsPromise) {
      return Promise.all([model, xData, yData, yNormalizer]);
    }

    if (
      model === undefined ||
      model === null ||
      xData === undefined ||
      xData === null ||
      yData === undefined ||
      yData === null
    ) {
      return null;
    }

    return [model, xData, yData, yNormalizer];
  }

  _inferInputScaledFlag(inputIndex, fallback = false) {
    try {
      const input = this.inputs && this.inputs[inputIndex];
      if (!input || input.link == null || !this.graph || !this.graph.links) {
        return !!fallback;
      }
      const link = this.graph.links[input.link];
      if (!link) return !!fallback;

      const originId =
        link.origin_id !== undefined ? link.origin_id : link.origin;
      const originSlot =
        link.origin_slot !== undefined ? link.origin_slot : link.originSlot;
      const originNode = this.graph.getNodeById(originId);
      if (!originNode || !originNode.outputs || originSlot == null) {
        return !!fallback;
      }

      const outputInfo = originNode.outputs[originSlot];
      const outputName = outputInfo && outputInfo.name ? outputInfo.name : "";
      if (typeof outputName === "string") {
        const lowered = outputName.toLowerCase();
        if (lowered.includes("_scaled")) return true;
        if (lowered === "x_data" || lowered === "y_data") return false;
      }
    } catch (_err) {
      // fall through to fallback
    }

    return !!fallback;
  }

  buildPayload(resolvedInput) {
    if (!Array.isArray(resolvedInput) || resolvedInput.length < 3) {
      this._error = "Invalid input for temporal evaluation (need model, x, y)";
      this._imageReady = false;
      this.setDirtyCanvas(true, true);
      return { endpoint: null, payload: null };
    }

    const model = resolvedInput[0];
    const xData = resolvedInput[1];
    const yData = resolvedInput[2];
    const yNormalizer = resolvedInput[3];

    if (!model || !xData || !yData) {
      this._error = "Missing or invalid model / x / y data";
      this._imageReady = false;
      this.setDirtyCanvas(true, true);
      return { endpoint: null, payload: null };
    }

    const endpoint =
      this.properties.endpoint || "/evaluate_temporal_supervised_model";

    const normalizerPayload =
      yNormalizer && typeof yNormalizer === "object" ? yNormalizer : {};
    const hasNormalizer = Object.keys(normalizerPayload).length > 0;
    const shouldUnnormalize = !!this.properties.unnormalize && hasNormalizer;

    const payload = {
      model: model,
      x_data: xData,
      y_data: yData,
      y_normalizer: normalizerPayload,
      x_data_is_scaled_input: this._inferInputScaledFlag(1, false),
      y_data_is_scaled_input: this._inferInputScaledFlag(2, false),
      unnormalize: shouldUnnormalize,
      output_index: parseInt(this.properties.output_index || 0),
      show_train: !!this.properties.show_train,
      show_val: !!this.properties.show_val,
      show_test: !!this.properties.show_test,
      show_uncertainty: !!this.properties.show_uncertainty,
      figsize: [
        this.properties.fig_width || 10,
        this.properties.fig_height || 6,
      ],
      image_format: "pdf",
    };

    return { endpoint, payload };
  }
}

PlotTemporalEvaluationNode.title = "Plot Temporal Model Evaluation";
PlotTemporalEvaluationNode.desc =
  "Plot notebook-style aligned temporal predictions for train, validation, and test splits.";

LiteGraph.registerNodeType(
  "MELT/Plot/TemporalEvaluationPlot",
  PlotTemporalEvaluationNode,
);
