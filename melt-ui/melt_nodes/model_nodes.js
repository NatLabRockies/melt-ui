// Model nodes
// IO nodes

class SaveModelNode extends AsyncMultiOutputNodeBase {
  constructor() {
    super("Save Model");

    // Inputs
    this.addInput("model", "object");

    // Outputs
    this.addOutput("path", "string");
    this.addOutput("metadata", "object");

    // Properties
    this.addProperty("save_dir", "saved_models", "string");
    this.addProperty("filename", "", "string");
    this.addProperty("overwrite", false, "boolean");
    this.addProperty("include_history", false, "boolean");
    this.addProperty("endpoint", "/save_model", "string");

    // Widgets
    this.addWidget("text", "Save Dir", this.properties.save_dir, (v) => {
      this.properties.save_dir = v;
    });
    this.addWidget("text", "Filename", this.properties.filename, (v) => {
      this.properties.filename = v;
    });
    this.addWidget("toggle", "Overwrite", this.properties.overwrite, (v) => {
      this.properties.overwrite = !!v;
    });
    this.addWidget(
      "toggle",
      "Include History",
      this.properties.include_history,
      (v) => {
        this.properties.include_history = !!v;
      },
    );

    this.size = [300, 200];
  }

  async fetch(resolvedInput) {
    const endpoint = this.properties.endpoint || "/save_model";

    // resolvedInput may be the model itself or [model]
    let model = resolvedInput;
    if (Array.isArray(resolvedInput)) {
      model = resolvedInput[0];
    }
    if (!model) {
      model = this.getInputData(0);
    }

    if (!model || typeof model !== "object" || !model.model_id) {
      return {
        path: "",
        metadata: {},
        error: "Missing or invalid model input",
      };
    }

    const payload = {
      model: model,
      save_dir: this.properties.save_dir,
      filename:
        this.properties.filename && this.properties.filename.trim().length > 0
          ? this.properties.filename
          : null,
      overwrite: !!this.properties.overwrite,
      include_history: !!this.properties.include_history,
    };

    const response = await window.MeltApi.fetch(endpoint, {
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
      const errMsg = `HTTP ${response.status} ${
        response.statusText
      } - ${JSON.stringify(json)}`;
      throw new Error(errMsg);
    }

    return json;
  }

  get extractors() {
    return [(j) => (j && j.path) || "", (j) => (j && j.metadata) || {}];
  }

  get defaults() {
    return ["", {}];
  }

  onExecute() {
    super.onExecute();
    this._pPath = this._p0;
    this._pMetadata = this._p1;
  }

  onPropertyChanged(name, value, prevValue) {
    if (value !== prevValue && this._runner) {
      this._runner.invalidate();
    }
  }
}

SaveModelNode.title = "Save Model";
SaveModelNode.desc =
  "Saves a model from the model store to disk as .safetensors with metadata.";

LiteGraph.registerNodeType("MELT/Model/SaveModelNode", SaveModelNode);

class LoadModelNode extends AsyncMultiOutputNodeBase {
  constructor() {
    super("Load Model");

    // Inputs
    // this.addInput("path", "string");

    // Outputs
    this.addOutput("model", "object");
    this.addOutput("metadata", "object");
    this.addOutput("path", "string");
    this.addOutput("x_normalizer", "object");
    this.addOutput("y_normalizer", "object");

    // Properties
    this.addProperty("path", "", "string");
    this.addProperty("strict", true, "boolean");
    this.addProperty("scaler_info_path", "", "string");
    this.addProperty("auto_load_scaler_info", true, "boolean");
    // this.addProperty("model_id", "", "string");
    this.addProperty("endpoint", "/load_model", "string");

    // Widgets
    this._pathWidget = this.addWidget(
      "text",
      "Model Path",
      this.properties.path,
      (v) => {
        this.properties.path = v;
      },
    );
    this.addWidget("button", "Browse Model...", null, () => {
      this._browseFile(
        "Select .safetensors Model",
        [
          ["Safetensors files", "*.safetensors"],
          ["All files", "*.*"],
        ],
        (p) => {
          this.properties.path = p;
          if (this._pathWidget) this._pathWidget.value = p;
          if (this._runner) this._runner.invalidate();
        },
      );
    });
    this.addWidget("toggle", "Strict", this.properties.strict, (v) => {
      this.properties.strict = !!v;
    });
    this._scalerInfoWidget = this.addWidget(
      "text",
      "Scaler Info Path",
      this.properties.scaler_info_path,
      (v) => {
        this.properties.scaler_info_path = v;
      },
    );
    this.addWidget("button", "Browse Scaler Info...", null, () => {
      this._browseFile(
        "Select scaler_info.txt",
        [
          ["Text files", "*.txt"],
          ["All files", "*.*"],
        ],
        (p) => {
          this.properties.scaler_info_path = p;
          if (this._scalerInfoWidget) this._scalerInfoWidget.value = p;
          if (this._runner) this._runner.invalidate();
        },
      );
    });
    this.addWidget(
      "toggle",
      "Auto Load Scaler Info",
      this.properties.auto_load_scaler_info,
      (v) => {
        this.properties.auto_load_scaler_info = !!v;
      },
    );

    this.size = [330, 240];
  }

  _syncWidgetsFromProperties() {
    if (this._pathWidget) {
      this._pathWidget.value = this.properties.path || "";
    }
    if (this._scalerInfoWidget) {
      this._scalerInfoWidget.value = this.properties.scaler_info_path || "";
    }
  }

  onConfigure(info) {
    if (info && info.properties) {
      const props = info.properties;
      if (typeof props.path === "string") {
        this.properties.path = props.path;
      }
      if (typeof props.scaler_info_path === "string") {
        this.properties.scaler_info_path = props.scaler_info_path;
      }
      if (typeof props.strict === "boolean") {
        this.properties.strict = props.strict;
      }
      if (typeof props.auto_load_scaler_info === "boolean") {
        this.properties.auto_load_scaler_info = props.auto_load_scaler_info;
      }
    }

    this._syncWidgetsFromProperties();
    this.setDirtyCanvas(true, true);
  }

  // Call /browse_file on the backend (opens native OS file picker) and pass
  // the selected path to the callback.
  _browseFile(title, filetypes, callback) {
    const node = this;
    window.MeltApi.fetch("/browse_file", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title, filetypes }),
    })
      .then((r) => r.json())
      .then((j) => {
        if (j && j.path) {
          callback(j.path);
          node.setDirtyCanvas(true, true);
        }
      })
      .catch((e) => console.error("Browse file error:", e));
  }

  async fetch(resolvedInput) {
    const endpoint = this.properties.endpoint || "/load_model";

    // Early exit when no path is set yet
    if (!this.properties.path || !this.properties.path.trim()) {
      return {};
    }

    const payload = {
      path: this.properties.path,
      strict: !!this.properties.strict,
      scaler_info_path:
        this.properties.scaler_info_path &&
        this.properties.scaler_info_path.trim().length > 0
          ? this.properties.scaler_info_path
          : null,
      auto_load_scaler_info: !!this.properties.auto_load_scaler_info,
      model_id:
        this.properties.model_id && this.properties.model_id.trim().length > 0
          ? this.properties.model_id
          : null,
    };

    const response = await window.MeltApi.fetch(endpoint, {
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
      const errMsg = `HTTP ${response.status} ${
        response.statusText
      } - ${JSON.stringify(json)}`;
      throw new Error(errMsg);
    }

    return json;
  }

  //   get extractors() {
  //     const model_dict = (j) => {
  //       return {
  //         model_id: j && j.model_id ? j.model_id : null,
  //         model_meta: j && j.model_meta ? j.model_meta : {},
  //       };
  //     };

  //     return [
  //       model_dict,
  //       (j) => (j && j.metadata) || {},
  //       (j) => (j && j.path) || "",
  //     ];
  //   }
  get extractors() {
    const model_dict = (j) => {
      // Support either {model_id, model_meta} or {model: {model_id, model_meta}}
      const m = (j && j.model) || j || {};
      return {
        model_id: m && m.model_id ? m.model_id : null,
        model_meta: m && m.model_meta ? m.model_meta : {},
      };
    };

    return [
      model_dict,
      (j) => (j && j.metadata) || {},
      (j) => (j && j.path) || "",
      (j) => (j && j.x_normalizer) || {},
      (j) => (j && j.y_normalizer) || {},
    ];
  }

  get defaults() {
    return [{}, {}, "", {}, {}];
  }

  onExecute() {
    super.onExecute();
    this._pModel = this._p0;
    this._pMetadata = this._p1;
    this._pPath = this._p2;
    this._pXNormalizer = this._p3;
    this._pYNormalizer = this._p4;
  }

  onPropertyChanged(name, value, prevValue) {
    if (value !== prevValue && this._runner) {
      this._runner.invalidate();
    }

    if (name === "path" || name === "scaler_info_path") {
      this._syncWidgetsFromProperties();
    }
  }
}

LoadModelNode.title = "Load Model";
LoadModelNode.desc =
  "Loads a .safetensors file, reconstructs the model, and adds it to the model store.";

LiteGraph.registerNodeType("MELT/Model/LoadModelNode", LoadModelNode);
