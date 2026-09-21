class SAMLoadImageNode extends AsyncMultiOutputNodeBase {
  constructor() {
    super("SAM Load Image");

    this.addOutput("image", "object");
    this.addOutput("image_uri", "string");
    this.addOutput("shape", "object");

    this.addProperty("path", "", "string");
    this.addProperty("endpoint", "/sam_load_image", "string");

    this._pathWidget = this.addTextPropertyWidget("Image Path", "path");

    this.addWidget("button", "Browse Image...", null, () => {
      this._browseFile(
        "Select Image",
        [
          ["Image files", "*.png *.jpg *.jpeg *.webp *.bmp"],
          ["All files", "*.*"],
        ],
        (p) => {
          const prev = this.properties.path;
          this.properties.path = p;
          if (this._pathWidget) this._pathWidget.value = p;
          if (p !== prev && typeof this.onPropertyChanged === "function") {
            this.onPropertyChanged("path", p, prev);
          }
          if (typeof this.invalidateDownstreamNodeRunners === "function") {
            this.invalidateDownstreamNodeRunners();
          }
          if (this._runner) this._runner.invalidate();
        },
      );
    });

    this.size = [340, 130];
  }

  _browseFile(title, filetypes, callback) {
    fetch("/browse_file", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title, filetypes }),
    })
      .then((r) => r.json())
      .then((j) => {
        if (j && j.path) callback(j.path);
      })
      .catch((e) => console.error("Browse image error:", e));
  }

  async fetch() {
    if (!this.properties.path || !this.properties.path.trim()) {
      return { image: null, image_uri: "", shape: { width: 0, height: 0 } };
    }

    const response = await fetch(
      this.properties.endpoint || "/sam_load_image",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ path: this.properties.path }),
      },
    );

    const text = await response.text();
    let json = null;
    try {
      json = text ? JSON.parse(text) : null;
    } catch (err) {
      json = { __raw_text: text };
    }

    if (!response.ok) {
      throw new Error(
        `HTTP ${response.status} ${response.statusText} - ${JSON.stringify(json)}`,
      );
    }

    const image = (json && json.image) || null;
    const shape = image
      ? { width: Number(image.width) || 0, height: Number(image.height) || 0 }
      : { width: 0, height: 0 };

    return {
      image: image,
      image_uri: image && image.image_data ? image.image_data : "",
      shape: shape,
    };
  }

  get extractors() {
    return [
      (j) => (j && j.image) || null,
      (j) => (j && j.image && j.image.image_data) || "",
      (j) =>
        (j &&
          j.image && {
            width: Number(j.image.width) || 0,
            height: Number(j.image.height) || 0,
          }) || { width: 0, height: 0 },
    ];
  }

  get defaults() {
    return [null, "", { width: 0, height: 0 }];
  }

  onExecute() {
    super.onExecute();
    this._pImage = this._p0;
    this._pImageUri = this._p1;
    this._pShape = this._p2;
  }
}

SAMLoadImageNode.title = "SAM Load Image";
SAMLoadImageNode.desc =
  "Loads an image from local path and emits data URI payload.";
LiteGraph.registerNodeType("MELT/SAM/LoadImage", SAMLoadImageNode);

class SAMPromptBuilderNode extends AsyncMultiOutputNodeBase {
  constructor() {
    super("SAM Prompt Builder");

    this.addInput("image", "object");

    this.addOutput("prompts", "object");
    this.addOutput("points", "array");
    this.addOutput("boxes", "array");

    this.addProperty("points_json", "[]", "string");
    this.addProperty("boxes_json", "[]", "string");

    this.addTextPropertyWidget("Points JSON", "points_json", {
      emptyValue: "[]",
    });
    this.addTextPropertyWidget("Boxes JSON", "boxes_json", {
      emptyValue: "[]",
    });

    this.size = [360, 140];
  }

  _parseArray(jsonText, label) {
    let parsed = [];
    try {
      parsed = JSON.parse(jsonText || "[]");
    } catch (err) {
      throw new Error(`${label} must be valid JSON array`);
    }
    if (!Array.isArray(parsed)) {
      throw new Error(`${label} must be a JSON array`);
    }
    return parsed;
  }

  async fetch() {
    const points = this._parseArray(this.properties.points_json, "Points JSON");
    const boxes = this._parseArray(this.properties.boxes_json, "Boxes JSON");
    return {
      prompts: { points, boxes },
      points,
      boxes,
    };
  }

  get extractors() {
    return [
      (j) => (j && j.prompts) || { points: [], boxes: [] },
      (j) => (j && j.points) || [],
      (j) => (j && j.boxes) || [],
    ];
  }

  get defaults() {
    return [{ points: [], boxes: [] }, [], []];
  }

  onExecute() {
    super.onExecute();
    this._pPrompts = this._p0;
    this._pPoints = this._p1;
    this._pBoxes = this._p2;
  }
}

SAMPromptBuilderNode.title = "SAM Prompt Builder";
SAMPromptBuilderNode.desc =
  "Builds SAM points/boxes prompts from JSON text fields.";
LiteGraph.registerNodeType("MELT/SAM/PromptBuilder", SAMPromptBuilderNode);

class SAMInteractivePromptNode extends LiteGraph.LGraphNode {
  constructor() {
    super();
    this.title = "SAM Interactive Prompt";

    this.addInput("image", "object");
    this.addOutput("prompts", "object");
    this.addOutput("points", "array");
    this.addOutput("boxes", "array");

    this.addProperty("mode", "point", "string");
    this.addProperty("point_polarity", "foreground", "string");
    this.addProperty("negative_with_shift", true, "boolean");

    this._points = [];
    this._boxes = [];
    this._image = null;
    this._imageReady = false;
    this._imageUri = "";
    this._imageWidth = 0;
    this._imageHeight = 0;
    this._drawRect = null;

    this._isDraggingBox = false;
    this._boxStart = null;
    this._boxCurrent = null;

    this.addWidget(
      "combo",
      "Mode",
      this.properties.mode,
      (v) => {
        this.properties.mode = String(v || "point");
        this.setDirtyCanvas(true, true);
      },
      {
        values: ["point", "box"],
      },
    );

    this.addWidget(
      "combo",
      "Point Type",
      this.properties.point_polarity,
      (v) => {
        this.properties.point_polarity =
          String(v || "foreground") === "background"
            ? "background"
            : "foreground";
        this.setDirtyCanvas(true, true);
      },
      {
        values: ["foreground", "background"],
      },
    );

    this.addWidget(
      "toggle",
      "Shift=Negative",
      this.properties.negative_with_shift,
      (v) => {
        this.properties.negative_with_shift = !!v;
      },
    );

    this.addWidget("button", "Clear Points", null, () => {
      this._points = [];
      this.setDirtyCanvas(true, true);
    });

    this.addWidget("button", "Clear Boxes", null, () => {
      this._boxes = [];
      this.setDirtyCanvas(true, true);
    });

    this.size = [360, 320];
  }

  _tryLoadImageFromInput(imageObj) {
    if (!imageObj || !imageObj.image_data) return;
    const uri = String(imageObj.image_data || "");
    if (!uri || uri === this._imageUri) return;

    this._imageUri = uri;
    this._imageWidth = Number(imageObj.width) || 0;
    this._imageHeight = Number(imageObj.height) || 0;

    const img = new Image();
    img.onload = () => {
      this._image = img;
      this._imageReady = true;
      if (!this._imageWidth) this._imageWidth = img.width;
      if (!this._imageHeight) this._imageHeight = img.height;
      this.setDirtyCanvas(true, true);
    };
    img.onerror = () => {
      this._image = null;
      this._imageReady = false;
      this.setDirtyCanvas(true, true);
    };
    img.src = uri;
  }

  _nodeToImageXY(nx, ny) {
    const r = this._drawRect;
    if (!r || !this._imageWidth || !this._imageHeight) return null;
    if (nx < r.x || ny < r.y || nx > r.x + r.w || ny > r.y + r.h) return null;

    const u = (nx - r.x) / r.w;
    const v = (ny - r.y) / r.h;
    const x = Math.max(0, Math.min(this._imageWidth - 1, u * this._imageWidth));
    const y = Math.max(
      0,
      Math.min(this._imageHeight - 1, v * this._imageHeight),
    );
    return { x, y };
  }

  _imageToNodeXY(ix, iy) {
    const r = this._drawRect;
    if (!r || !this._imageWidth || !this._imageHeight) return null;
    return {
      x: r.x + (ix / this._imageWidth) * r.w,
      y: r.y + (iy / this._imageHeight) * r.h,
    };
  }

  onExecute() {
    const imageInput = this.getInputData(0);

    if (imageInput && typeof imageInput.then === "function") {
      imageInput
        .then((resolved) => {
          this._tryLoadImageFromInput(resolved);
          this.setDirtyCanvas(true, true);
        })
        .catch(() => {});
    } else {
      this._tryLoadImageFromInput(imageInput);
    }

    const prompts = {
      points: this._points.slice(),
      boxes: this._boxes.slice(),
    };
    this.setOutputData(0, prompts);
    this.setOutputData(1, prompts.points);
    this.setOutputData(2, prompts.boxes);
  }

  onMouseDown(event, pos) {
    if (!this._imageReady) return false;

    const p = this._nodeToImageXY(pos[0], pos[1]);
    if (!p) return false;

    if ((this.properties.mode || "point") === "box") {
      this._isDraggingBox = true;
      this._boxStart = p;
      this._boxCurrent = p;
      this.setDirtyCanvas(true, true);
      return true;
    }

    const baseIsNegative =
      (this.properties.point_polarity || "foreground") === "background";
    const useShiftOverride =
      !!this.properties.negative_with_shift && !!event.shiftKey;
    const isNegative = useShiftOverride ? !baseIsNegative : baseIsNegative;
    this._points.push({ x: p.x, y: p.y, label: isNegative ? 0 : 1 });
    this.setDirtyCanvas(true, true);
    return true;
  }

  onMouseMove(event, pos) {
    if (!this._isDraggingBox) return false;
    const p = this._nodeToImageXY(pos[0], pos[1]);
    if (!p) return false;
    this._boxCurrent = p;
    this.setDirtyCanvas(true, true);
    return true;
  }

  onMouseUp() {
    if (!this._isDraggingBox) return false;
    this._isDraggingBox = false;

    if (this._boxStart && this._boxCurrent) {
      const x1 = Math.min(this._boxStart.x, this._boxCurrent.x);
      const y1 = Math.min(this._boxStart.y, this._boxCurrent.y);
      const x2 = Math.max(this._boxStart.x, this._boxCurrent.x);
      const y2 = Math.max(this._boxStart.y, this._boxCurrent.y);
      if (Math.abs(x2 - x1) > 2 && Math.abs(y2 - y1) > 2) {
        this._boxes.push({ x1, y1, x2, y2 });
      }
    }

    this._boxStart = null;
    this._boxCurrent = null;
    this.setDirtyCanvas(true, true);
    return true;
  }

  _drawOverlay(ctx) {
    for (let i = 0; i < this._boxes.length; i++) {
      const b = this._boxes[i];
      const p1 = this._imageToNodeXY(b.x1, b.y1);
      const p2 = this._imageToNodeXY(b.x2, b.y2);
      if (!p1 || !p2) continue;
      const x = Math.min(p1.x, p2.x);
      const y = Math.min(p1.y, p2.y);
      const w = Math.abs(p2.x - p1.x);
      const h = Math.abs(p2.y - p1.y);
      ctx.strokeStyle = "#ffc857";
      ctx.lineWidth = 2;
      ctx.strokeRect(x, y, w, h);
    }

    if (this._isDraggingBox && this._boxStart && this._boxCurrent) {
      const p1 = this._imageToNodeXY(this._boxStart.x, this._boxStart.y);
      const p2 = this._imageToNodeXY(this._boxCurrent.x, this._boxCurrent.y);
      if (p1 && p2) {
        const x = Math.min(p1.x, p2.x);
        const y = Math.min(p1.y, p2.y);
        const w = Math.abs(p2.x - p1.x);
        const h = Math.abs(p2.y - p1.y);
        ctx.setLineDash([6, 4]);
        ctx.strokeStyle = "#ffe9a8";
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, w, h);
        ctx.setLineDash([]);
      }
    }

    for (let i = 0; i < this._points.length; i++) {
      const pt = this._points[i];
      const np = this._imageToNodeXY(pt.x, pt.y);
      if (!np) continue;
      ctx.beginPath();
      ctx.arc(np.x, np.y, 4, 0, Math.PI * 2);
      ctx.fillStyle = pt.label === 1 ? "#00d26a" : "#ff5e5e";
      ctx.fill();
      ctx.strokeStyle = "#111";
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  }

  onDrawBackground(ctx) {
    const w = this.size[0];
    const h = this.size[1];
    const top = 106;
    const areaH = h - top - 8;

    ctx.save();
    const mode = this.properties.mode || "point";
    const polarity = this.properties.point_polarity || "foreground";
    const activeLabel = polarity === "background" ? "Background" : "Foreground";
    const activeColor = polarity === "background" ? "#ff5e5e" : "#00d26a";

    ctx.font = "12px sans-serif";
    ctx.textAlign = "left";
    ctx.fillStyle = "#ddd";
    ctx.fillText("Point label:", 8, 86);
    ctx.fillStyle = activeColor;
    ctx.fillText(activeLabel, 78, 86);

    if (mode === "point" && this.properties.negative_with_shift) {
      ctx.fillStyle = "#9aa0a6";
      ctx.fillText("(Hold Shift to invert)", 154, 86);
    }

    ctx.fillStyle = "#111";
    ctx.fillRect(0, top, w, areaH);

    this._drawRect = null;

    if (this._imageReady && this._image) {
      const img = this._image;
      const sx = (w - 12) / img.width;
      const sy = (areaH - 12) / img.height;
      const scale = Math.min(sx, sy);
      const drawW = Math.max(1, Math.floor(img.width * scale));
      const drawH = Math.max(1, Math.floor(img.height * scale));
      const ox = Math.floor((w - drawW) / 2);
      const oy = top + Math.floor((areaH - drawH) / 2);

      ctx.drawImage(img, ox, oy, drawW, drawH);
      this._drawRect = { x: ox, y: oy, w: drawW, h: drawH };
      this._drawOverlay(ctx);
    } else {
      ctx.fillStyle = "#888";
      ctx.font = "12px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("Connect image input", w / 2, top + areaH / 2);
      ctx.textAlign = "left";
    }
    ctx.restore();
  }
}

SAMInteractivePromptNode.title = "SAM Interactive Prompt";
SAMInteractivePromptNode.desc =
  "Select foreground/background and click points, or draw boxes on the image.";
LiteGraph.registerNodeType(
  "MELT/SAM/InteractivePrompt",
  SAMInteractivePromptNode,
);

const SAM_MODEL_CHECKPOINTS = [
  "facebook/sam-vit-base",
  "facebook/sam-vit-large",
  "facebook/sam-vit-huge",
  "facebook/sam2-hiera-tiny",
  "facebook/sam2-hiera-small",
  "facebook/sam2-hiera-base-plus",
  "facebook/sam2-hiera-large",
  "facebook/sam2.1-hiera-tiny",
  "facebook/sam2.1-hiera-small",
  "facebook/sam2.1-hiera-base-plus",
  "facebook/sam2.1-hiera-large",
  "syscv-community/sam-hq-vit-base",
];

class SAMLoadModelNode extends AsyncMultiOutputNodeBase {
  constructor() {
    super("SAM Load Model");

    this.addOutput("model", "object");
    this.addOutput("status", "string");

    this.addProperty("checkpoint", "facebook/sam-vit-base", "string");
    this.addProperty("device", "auto", "string");
    this.addProperty("endpoint", "/sam_load_model", "string");

    this._error = "";
    this._isLoading = false;
    this._loadingStartTime = null;

    this.addDropdownPropertyWidget("Checkpoint", "checkpoint", {
      values: SAM_MODEL_CHECKPOINTS,
      default: "facebook/sam-vit-base",
    });

    this.addDropdownPropertyWidget("Device", "device", {
      values: ["auto", "cpu", "cuda", "mps"],
      default: "auto",
    });

    this.size = [340, 180];
  }

  async fetch() {
    this._isLoading = true;
    this._loadingStartTime = Date.now();
    this._error = "";
    this.setDirtyCanvas(true, true);

    try {
      console.log(
        `[SAMLoadModelNode] Fetching checkpoint: ${this.properties.checkpoint} on device: ${this.properties.device}`,
      );

      const response = await fetch(
        this.properties.endpoint || "/sam_load_model",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            checkpoint: this.properties.checkpoint,
            device: this.properties.device,
          }),
        },
      );

      const text = await response.text();
      let json = null;
      try {
        json = text ? JSON.parse(text) : null;
      } catch (err) {
        json = { __raw_text: text };
      }

      if (!response.ok) {
        const errorMsg = `HTTP ${response.status} ${response.statusText} - ${JSON.stringify(json)}`;
        console.error(`[SAMLoadModelNode] Backend error: ${errorMsg}`);
        this._error = errorMsg;
        throw new Error(errorMsg);
      }

      const elapsed = ((Date.now() - this._loadingStartTime) / 1000).toFixed(1);
      console.log(
        `[SAMLoadModelNode] Success (${elapsed}s): ${json.status}, model_id=${json.model?.model_id}`,
      );
      return json;
    } catch (err) {
      this._error = err && err.message ? err.message : String(err);
      console.error(`[SAMLoadModelNode] Exception: ${this._error}`);
      throw err;
    } finally {
      this._isLoading = false;
      this.setDirtyCanvas(true, true);
    }
  }

  get extractors() {
    return [
      (j) => (j && j.model) || null,
      (j) => {
        if (this._error) return `ERROR: ${this._error}`;
        if (this._isLoading) return "Loading...";
        return (j && j.status) || "";
      },
    ];
  }

  get defaults() {
    return [null, ""];
  }

  onExecute() {
    super.onExecute();
    this._pModel = this._p0;
    this._pStatus = this._p1;
  }

  onDrawBackground(ctx) {
    const w = this.size[0];
    const h = this.size[1];
    const titleH = 26;

    ctx.fillStyle = "#1a1a1a";
    ctx.fillRect(0, titleH, w, h - titleH);

    if (this._error) {
      ctx.fillStyle = "#ff3333";
      ctx.fillRect(0, titleH, w, 24);
      ctx.fillStyle = "#fff";
      ctx.font = "bold 12px monospace";
      ctx.textAlign = "center";
      ctx.fillText("ERROR", w / 2, titleH + 16);

      ctx.fillStyle = "#ffaaaa";
      ctx.font = "10px monospace";
      const errLines = this._error.split(" - ");
      for (let i = 0; i < Math.min(errLines.length, 3); i++) {
        ctx.fillText(errLines[i].substring(0, 45), w / 2, titleH + 40 + i * 16);
      }
    } else if (this._isLoading) {
      ctx.fillStyle = "#ffaa00";
      ctx.fillRect(0, titleH, w, 24);
      ctx.fillStyle = "#000";
      ctx.font = "bold 12px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("⏳ LOADING MODEL", w / 2, titleH + 16);

      ctx.fillStyle = "#ffddaa";
      ctx.font = "11px sans-serif";
      const elapsed = this._loadingStartTime
        ? ((Date.now() - this._loadingStartTime) / 1000).toFixed(1)
        : "0";
      ctx.fillText(
        `Elapsed: ${elapsed}s (download size varies)`,
        w / 2,
        titleH + 55,
      );
      ctx.fillText("Check backend logs for progress", w / 2, titleH + 75);
    } else {
      ctx.fillStyle = "#00aa00";
      ctx.fillRect(0, titleH, w, 24);
      ctx.fillStyle = "#000";
      ctx.font = "bold 12px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("✓ READY", w / 2, titleH + 16);
    }
  }
}

SAMLoadModelNode.title = "SAM Load Model";
SAMLoadModelNode.desc =
  "Loads a SAM model from a Hugging Face checkpoint. First load downloads ~375MB.";
LiteGraph.registerNodeType("MELT/SAM/LoadModel", SAMLoadModelNode);

class SAMSegmentNode extends AsyncMultiOutputNodeBase {
  constructor() {
    super("SAM Segment");

    this.addInput("model", "object");
    this.addInput("image", "object");
    this.addInput("prompts", "object");

    this.addOutput("masks", "array");
    this.addOutput("scores", "array");
    this.addOutput("best_mask_image", "string");
    this.addOutput("overlay_image", "string");
    this.addOutput("result", "object");

    this.addProperty("multimask_output", true, "boolean");
    this.addProperty("endpoint", "/sam_segment", "string");

    this.addBooleanPropertyWidget("Multimask", "multimask_output", {
      default: true,
    });

    this.size = [300, 140];
  }

  async fetch(resolvedInput) {
    const model = Array.isArray(resolvedInput) ? resolvedInput[0] : null;
    const image = Array.isArray(resolvedInput) ? resolvedInput[1] : null;
    const prompts = Array.isArray(resolvedInput) ? resolvedInput[2] : null;

    if (!model || !model.model_id) {
      throw new Error("Missing model input with model_id");
    }
    if (!image || !image.image_data) {
      throw new Error("Missing image input with image_data");
    }

    const points = (prompts && prompts.points) || [];
    const boxes = (prompts && prompts.boxes) || [];
    if (points.length === 0 && boxes.length === 0) {
      throw new Error(
        "No prompts yet — add points or boxes in the Interactive Prompt node first",
      );
    }

    const payload = {
      model,
      image,
      prompts: { points, boxes },
      multimask_output: !!this.properties.multimask_output,
    };

    const response = await fetch(this.properties.endpoint || "/sam_segment", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const text = await response.text();
    let json = null;
    try {
      json = text ? JSON.parse(text) : null;
    } catch (err) {
      json = { __raw_text: text };
    }

    if (!response.ok) {
      throw new Error(
        `HTTP ${response.status} ${response.statusText} - ${JSON.stringify(json)}`,
      );
    }

    return json;
  }

  get extractors() {
    return [
      (j) => (j && j.masks) || [],
      (j) => (j && j.scores) || [],
      (j) => (j && j.best_mask_image) || "",
      (j) => (j && j.overlay_image) || "",
      (j) => j || {},
    ];
  }

  get defaults() {
    return [[], [], "", "", {}];
  }

  _invalidateOnPromptsChange(promptsInput) {
    if (!promptsInput || typeof promptsInput.then === "function") return;

    try {
      const sig = JSON.stringify(promptsInput);
      if (sig !== this._lastPromptsSig) {
        this._lastPromptsSig = sig;
        if (this._runner) this._runner.invalidate();
        if (typeof this.invalidateDownstreamNodeRunners === "function") {
          this.invalidateDownstreamNodeRunners();
        }
      }
    } catch (_) {}
  }

  onExecute() {
    const promptsInput = this.getInputData(2);
    if (promptsInput && typeof promptsInput.then === "function") {
      promptsInput
        .then((resolved) => {
          this._invalidateOnPromptsChange(resolved);
          this.setDirtyCanvas(true, true);
        })
        .catch(() => {});
    } else {
      this._invalidateOnPromptsChange(promptsInput);
    }

    super.onExecute();
    this._pMasks = this._p0;
    this._pScores = this._p1;
    this._pBestMaskImage = this._p2;
    this._pOverlayImage = this._p3;
    this._pResult = this._p4;
  }
}

SAMSegmentNode.title = "SAM Segment";
SAMSegmentNode.desc = "Runs SAM segmentation for an image and prompts.";
LiteGraph.registerNodeType("MELT/SAM/Segment", SAMSegmentNode);

class SAMImageDisplayNode extends LiteGraph.LGraphNode {
  constructor() {
    super();
    this.title = "SAM Image Display";
    this.addInput("image_uri", "string");
    this.addOutput("image_uri", "string");

    this._image = null;
    this._imageReady = false;
    this._currentUri = "";
    this._error = "";
    this._lastRequestId = 0;

    this.size = [320, 260];
  }

  onExecute() {
    const maybeInput = this.getInputData(0);

    // Handle promises from async upstream nodes
    if (maybeInput && typeof maybeInput.then === "function") {
      const requestId = ++this._lastRequestId;
      maybeInput
        .then((uri) => {
          if (requestId !== this._lastRequestId) return;
          this._setUri(typeof uri === "string" ? uri : "");
          this.setOutputData(0, this._currentUri || "");
        })
        .catch((err) => {
          if (requestId !== this._lastRequestId) return;
          this._error =
            "Error resolving input: " + ((err && err.message) || String(err));
          this.setDirtyCanvas(true, true);
        });
      this.setOutputData(0, maybeInput);
      return;
    }

    // Handle direct string values
    this._lastRequestId++;
    this._setUri(typeof maybeInput === "string" ? maybeInput : "");

    this.setOutputData(0, this._currentUri || "");
  }

  _setUri(uri) {
    if (!uri) {
      if (this._currentUri || this._image || this._imageReady || this._error) {
        this._currentUri = "";
        this._image = null;
        this._imageReady = false;
        this._error = "";
        this.setDirtyCanvas(true, true);
      }
      return;
    }

    if (uri !== this._currentUri || !this._imageReady) {
      this._currentUri = uri;
      this._loadUri(uri);
    }
  }

  _loadUri(uri) {
    const img = new Image();
    img.onload = () => {
      this._image = img;
      this._imageReady = true;
      this._error = "";
      this.setDirtyCanvas(true, true);
    };
    img.onerror = () => {
      this._image = null;
      this._imageReady = false;
      this._error = "Failed to load image";
      this.setDirtyCanvas(true, true);
    };
    img.src = uri;
  }

  onDrawBackground(ctx) {
    const w = this.size[0];
    const h = this.size[1];
    const top = 26;
    const areaH = h - top - 8;

    ctx.save();
    ctx.fillStyle = "#121212";
    ctx.fillRect(0, top, w, areaH);

    if (this._imageReady && this._image) {
      const img = this._image;
      const sx = (w - 12) / img.width;
      const sy = (areaH - 12) / img.height;
      const scale = Math.min(sx, sy);
      const drawW = Math.max(1, Math.floor(img.width * scale));
      const drawH = Math.max(1, Math.floor(img.height * scale));
      const ox = Math.floor((w - drawW) / 2);
      const oy = top + Math.floor((areaH - drawH) / 2);
      ctx.drawImage(img, ox, oy, drawW, drawH);
    } else {
      ctx.fillStyle = "#8a8a8a";
      ctx.font = "12px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(this._error || "No image", w / 2, top + areaH / 2);
      ctx.textAlign = "left";
    }
    ctx.restore();
  }
}

SAMImageDisplayNode.title = "SAM Image Display";
SAMImageDisplayNode.desc = "Displays image data URI from SAM nodes.";
LiteGraph.registerNodeType("MELT/SAM/ImageDisplay", SAMImageDisplayNode);

class GroundedDINOTextPromptNode extends AsyncMultiOutputNodeBase {
  constructor() {
    super("Grounded DINO Text Prompt");

    this.addOutput("prompts", "object");
    this.addOutput("text_prompt", "string");

    this.addProperty("text_prompt", "person", "string");
    this.addProperty("box_threshold", 0.3, "number");
    this.addProperty("text_threshold", 0.25, "number");

    this.addTextPropertyWidget("Text Prompt", "text_prompt", {
      default: "person",
      emptyValue: "person",
    });

    this.addFloatPropertyWidget("Box Threshold", "box_threshold", {
      min: 0.0,
      max: 1.0,
      step: 0.05,
      default: 0.3,
      precision: 2,
    });

    this.addFloatPropertyWidget("Text Threshold", "text_threshold", {
      min: 0.0,
      max: 1.0,
      step: 0.05,
      default: 0.25,
      precision: 2,
    });

    this.size = [360, 160];
  }

  async fetch() {
    return {
      prompts: {
        text: this.properties.text_prompt,
        box_threshold: this.properties.box_threshold,
        text_threshold: this.properties.text_threshold,
      },
      text_prompt: this.properties.text_prompt,
    };
  }

  get extractors() {
    return [(j) => (j && j.prompts) || {}, (j) => (j && j.text_prompt) || ""];
  }

  get defaults() {
    return [{}, ""];
  }

  onExecute() {
    super.onExecute();
    this._pPrompts = this._p0;
    this._pTextPrompt = this._p1;
  }
}

GroundedDINOTextPromptNode.title = "Grounded DINO Text Prompt";
GroundedDINOTextPromptNode.desc =
  "Builds text prompt for Grounded DINO object detection.";
LiteGraph.registerNodeType(
  "MELT/SAM/GroundedDINOTextPrompt",
  GroundedDINOTextPromptNode,
);

class GroundedDINODetectNode extends AsyncMultiOutputNodeBase {
  constructor() {
    super("Grounded DINO Detect");

    this.addInput("image", "object");
    this.addInput("prompts", "object");

    this.addOutput("boxes", "array");
    this.addOutput("scores", "array");
    this.addOutput("detections", "object");

    this.addProperty("endpoint", "/grounded_dino_detect", "string");

    this.size = [320, 120];
  }

  async fetch(resolvedInput) {
    const image = Array.isArray(resolvedInput) ? resolvedInput[0] : null;
    const prompts = Array.isArray(resolvedInput) ? resolvedInput[1] : null;

    if (!image || !image.image_data) {
      throw new Error("Missing image input with image_data");
    }

    const text_prompt = prompts && prompts.text ? prompts.text : "object";
    const box_threshold =
      prompts && typeof prompts.box_threshold === "number"
        ? prompts.box_threshold
        : 0.3;
    const text_threshold =
      prompts && typeof prompts.text_threshold === "number"
        ? prompts.text_threshold
        : 0.25;

    const payload = {
      image,
      text_prompt,
      box_threshold,
      text_threshold,
    };

    const response = await fetch(
      this.properties.endpoint || "/grounded_dino_detect",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      },
    );

    const text = await response.text();
    let json = null;
    try {
      json = text ? JSON.parse(text) : null;
    } catch (err) {
      json = { __raw_text: text };
    }

    if (!response.ok) {
      throw new Error(
        `HTTP ${response.status} ${response.statusText} - ${JSON.stringify(json)}`,
      );
    }

    return json;
  }

  get extractors() {
    return [
      (j) => (j && j.boxes) || [],
      (j) => (j && j.scores) || [],
      (j) => j || {},
    ];
  }

  get defaults() {
    return [[], [], {}];
  }

  _invalidateOnPromptChange(promptsInput) {
    if (!promptsInput || typeof promptsInput.then === "function") return;

    try {
      const sig = JSON.stringify(promptsInput);
      if (sig !== this._lastPromptsSig) {
        this._lastPromptsSig = sig;
        if (this._runner) this._runner.invalidate();
      }
    } catch (_) {}
  }

  onExecute() {
    const promptsInput = this.getInputData(1);
    if (promptsInput && typeof promptsInput.then === "function") {
      promptsInput
        .then((resolved) => {
          this._invalidateOnPromptChange(resolved);
          this.setDirtyCanvas(true, true);
        })
        .catch(() => {});
    } else {
      this._invalidateOnPromptChange(promptsInput);
    }

    super.onExecute();
    this._pBoxes = this._p0;
    this._pScores = this._p1;
    this._pDetections = this._p2;
  }
}

GroundedDINODetectNode.title = "Grounded DINO Detect";
GroundedDINODetectNode.desc =
  "Detects objects via text prompts using Grounded DINO.";
LiteGraph.registerNodeType(
  "MELT/SAM/GroundedDINODetect",
  GroundedDINODetectNode,
);

class DetectionBoxesToSAMNode extends AsyncMultiOutputNodeBase {
  constructor() {
    super("Detections → SAM Boxes");

    this.addInput("detections", "object");

    this.addOutput("prompts", "object");
    this.addOutput("boxes", "array");

    this.addProperty("use_all_boxes", true, "boolean");
    this.addProperty("top_k", 1, "number");
    this.addProperty("min_score", 0.0, "number");

    this.addBooleanPropertyWidget("Use All Boxes", "use_all_boxes", {
      default: true,
    });

    this.addIntPropertyWidget("Top K", "top_k", {
      min: 1,
      default: 1,
    });

    this.addFloatPropertyWidget("Min Score", "min_score", {
      min: 0.0,
      max: 1.0,
      step: 0.1,
      default: 0.0,
      precision: 2,
    });

    this.size = [340, 150];
  }

  async fetch(resolvedInput) {
    const detections = resolvedInput || {};
    const boxes_raw = detections.boxes || [];
    const scores_raw = detections.scores || [];

    let filtered_boxes = [];
    const min_score = this.properties.min_score;

    for (let i = 0; i < boxes_raw.length; i++) {
      const score = scores_raw[i] !== undefined ? scores_raw[i] : 1.0;
      if (score >= min_score) {
        filtered_boxes.push({ box: boxes_raw[i], score: score, index: i });
      }
    }

    filtered_boxes.sort((a, b) => b.score - a.score);

    let output_boxes = [];
    if (this.properties.use_all_boxes) {
      output_boxes = filtered_boxes.map((item) => item.box);
    } else {
      const k = Math.min(this.properties.top_k || 1, filtered_boxes.length);
      output_boxes = filtered_boxes.slice(0, k).map((item) => item.box);
    }

    return {
      prompts: { boxes: output_boxes, points: [] },
      boxes: output_boxes,
    };
  }

  get extractors() {
    return [
      (j) => (j && j.prompts) || { boxes: [], points: [] },
      (j) => (j && j.boxes) || [],
    ];
  }

  get defaults() {
    return [{ boxes: [], points: [] }, []];
  }

  _invalidateOnDetectionsChange(detectionsInput) {
    if (!detectionsInput || typeof detectionsInput.then === "function") return;

    try {
      const sig = JSON.stringify(detectionsInput);
      if (sig !== this._lastDetectionsSig) {
        this._lastDetectionsSig = sig;
        if (this._runner) this._runner.invalidate();
      }
    } catch (_) {}
  }

  onExecute() {
    const detectionsInput = this.getInputData(0);
    if (detectionsInput && typeof detectionsInput.then === "function") {
      detectionsInput
        .then((resolved) => {
          this._invalidateOnDetectionsChange(resolved);
          this.setDirtyCanvas(true, true);
        })
        .catch(() => {});
    } else {
      this._invalidateOnDetectionsChange(detectionsInput);
    }

    super.onExecute();
    this._pPrompts = this._p0;
    this._pBoxes = this._p1;
  }
}

DetectionBoxesToSAMNode.title = "Detections → SAM Boxes";
DetectionBoxesToSAMNode.desc =
  "Converts Grounded DINO detections to SAM box prompts.";
LiteGraph.registerNodeType(
  "MELT/SAM/DetectionsToSAMBoxes",
  DetectionBoxesToSAMNode,
);

class SAMHealthCheckNode extends AsyncMultiOutputNodeBase {
  constructor() {
    super("SAM Health Check");

    this.addOutput("status", "object");
    this.addOutput("message", "string");

    this.addProperty("endpoint", "/sam_health", "string");

    this.addWidget("button", "Check Now", null, () => {
      if (this._runner) this._runner.invalidate();
    });

    this.size = [340, 140];
  }

  async fetch() {
    const response = await fetch(this.properties.endpoint || "/sam_health", {
      method: "GET",
      headers: { "Content-Type": "application/json" },
    });

    const text = await response.text();
    let json = null;
    try {
      json = text ? JSON.parse(text) : null;
    } catch (err) {
      json = { __raw_text: text };
    }

    if (!response.ok) {
      throw new Error(`HTTP ${response.status} ${response.statusText}`);
    }

    const diag = (json && json.diagnostics) || {};
    let msg = "";
    if (json && json.ok) {
      msg = "✓ All SAM dependencies installed";
    } else {
      const missing = [];
      if (!diag.transformers_available) missing.push("transformers");
      if (!diag.pillow_available) missing.push("pillow");
      if (!diag.torch_available) missing.push("torch");
      if (missing.length > 0) {
        msg = `Missing: ${missing.join(", ")}. Run: pip install ${missing.join(" ")}`;
      }
    }

    return { status: json, message: msg };
  }

  get extractors() {
    return [(j) => (j && j.status) || {}, (j) => (j && j.message) || ""];
  }

  get defaults() {
    return [{}, ""];
  }

  onExecute() {
    super.onExecute();
    this._pStatus = this._p0;
    this._pMessage = this._p1;
  }

  onDrawBackground(ctx) {
    const w = this.size[0];
    const h = this.size[1];
    const titleH = 26;

    ctx.fillStyle = "#1a1a1a";
    ctx.fillRect(0, titleH, w, h - titleH);

    if (this._pMessage && typeof this._pMessage.then === "function") {
      this._pMessage
        .then((msg) => {
          this._displayedMsg = msg;
          this.setDirtyCanvas(true, true);
        })
        .catch(() => {});
    }

    ctx.fillStyle = "#88ff88";
    ctx.font = "11px monospace";
    ctx.textAlign = "center";
    if (this._displayedMsg) {
      const lines = (this._displayedMsg || "").split("\n");
      for (let i = 0; i < lines.length; i++) {
        ctx.fillText(lines[i], w / 2, titleH + 25 + i * 16);
      }
    } else {
      ctx.fillText("Click 'Check Now' or execute graph", w / 2, titleH + 50);
    }
  }
}

SAMHealthCheckNode.title = "SAM Health Check";
SAMHealthCheckNode.desc =
  "Checks if SAM dependencies are installed and available.";
LiteGraph.registerNodeType("MELT/SAM/HealthCheck", SAMHealthCheckNode);
