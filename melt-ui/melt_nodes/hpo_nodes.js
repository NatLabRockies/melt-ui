const HPO_BUILDER_PARAMETER_FALLBACK = [
  "learning_rate",
  "dropout",
  "num_mixtures",
  "l2_reg",
  "batch_size",
  "width",
  "depth",
  "node_list",
  "rnn_type",
  "head_type",
  "num_heads",
  "latent_dims",
  "encoder_node_list",
  "decoder_node_list",
];

const HPO_BUILDER_LABELS = {
  architecture: "Architecture",
  learning_rate: "Learning Rate",
  dropout: "Dropout",
  dropout_rate: "Dropout",
  num_mixtures: "Num Mixtures",
  l2_reg: "L2 Regularization",
  batch_size: "Batch Size",
  width: "Width",
  depth: "Depth",
  node_list: "Node List",
  rnn_type: "RNN Type",
  head_type: "Head Type",
  num_heads: "Attention Heads",
  latent_dims: "Latent Dims",
  encoder_node_list: "Encoder Nodes",
  decoder_node_list: "Decoder Nodes",
};

const HPO_BUILDER_PROPERTY_BY_KEY = {
  dropout: "dropout_rate",
};

const HPO_BUILDER_DEFAULT_SPECS = {
  learning_rate: { type: "choice", values: [0.001, 0.0003, 0.0001] },
  dropout: { type: "choice", values: [0, 0.1, 0.2] },
  num_mixtures: { type: "choice", values: [0, 1, 3] },
  l2_reg: { type: "choice", values: [0, 0.00001, 0.0001] },
  batch_size: { type: "choice", values: [16, 32, 64] },
  width: { type: "choice", values: [16, 32, 64] },
  depth: { type: "choice", values: [1, 2, 3] },
  rnn_type: { type: "choice", values: ["lstm", "gru", "rnn"] },
  head_type: { type: "choice", values: ["last", "mean", "max", "attn"] },
  num_heads: { type: "choice", values: [2, 4, 8] },
  latent_dims: { type: "choice", values: [2, 4, 8, 16] },
};

const HPO_BUILDER_SEARCH_TYPES = [
  "choice",
  "uniform",
  "loguniform",
  "randint",
  "quniform",
  "qloguniform",
];

const HPO_METHOD_LABELS = {
  choice: "Choices",
  uniform: "Linear Range",
  loguniform: "Logarithmic Range",
  randint: "Integer Range",
  quniform: "Quantized Linear Range",
  qloguniform: "Quantized Log Range",
};

const HPO_METHOD_LABELS_BY_KEY = {
  node_list: {
    choice: "Whole Architecture Choices",
    independent_layer_choice: "Independent Layer Choices",
  },
};

const HPO_METHOD_LABEL_TO_ID = Object.fromEntries(
  Object.entries(HPO_METHOD_LABELS).map(([id, label]) => [label, id]),
);

const HPO_ARCHITECTURE_KEY = "architecture";

class MELTHyperparameterTunerNode extends AsyncMultiOutputNodeBase {
  constructor() {
    super("MELT Hyperparameter Tuner");

    this.addInput("x", "array");
    this.addInput("y", "array");
    this.addInput("lengths", "array");

    this.addOutput("best_hyperparameters", "object");
    this.addOutput("tuning_result", "object");
    this.addOutput("trial_history", "object");
    this.addOutput("artifact_path", "string");

    this.addProperty("trainer_family", "static", "string");
    this.addProperty("model_architecture", "ann", "string");
    this.addProperty("metric", "val_loss", "string");
    this.addProperty("mode", "min", "string");
    this.addProperty("num_samples", 10, "number");
    this.addProperty("max_epochs", 10, "number");
    this.addProperty("scheduler", "asha", "string");
    this.addProperty("search_alg", "optuna", "string");
    this.addProperty("max_concurrent", 2, "number");
    this.addProperty("checkpoint_interval", 1, "number");
    this.addProperty("autosave", true, "boolean");
    this.addProperty("endpoint", "/melt_hyperparameter_tuner", "string");

    this.addProperty("val_size", 0.1, "number");
    this.addProperty("test_size", 0.1, "number");
    this.addProperty("norm_type", "none", "string");
    this.addProperty("random_state", 42, "number");
    this.addProperty("shuffle", true, "boolean");
    this.addProperty("batch_size", 32, "number");
    this.addProperty("loss_function", "mse", "string");
    this.addProperty("optimizer", "Adam", "string");
    this.addProperty("learning_rate", 1e-3, "number");
    this.addProperty("dropout_rate", 0.0, "number");
    this.addProperty("batch_norm", false, "boolean");
    this.addProperty("activation_function", "relu", "string");
    this.addProperty("output_activation", "linear", "string");
    this.addProperty("node_list", [8, 8], "array");
    this.addProperty("width", 32, "number");
    this.addProperty("depth", 2, "number");
    this.addProperty("rnn_type", "lstm", "string");
    this.addProperty("head_type", "last", "string");
    this.addProperty("num_heads", 4, "number");
    this.addProperty("ff_dim", 0, "number");
    this.addProperty("max_seq_len", 2048, "number");
    this.addProperty("use_causal_mask", false, "boolean");
    this.addProperty("seq_length", 60, "number");
    this.addProperty("seq_to_one", true, "boolean");
    this.addProperty("suffix_crop", false, "boolean");
    this.addProperty("suffix_crop_min_length", 32, "number");
    this.addProperty("latent_dims", 8, "number");
    this.addProperty("encoder_node_list", [32, 16], "array");
    this.addProperty("decoder_node_list", [16, 32], "array");
    this.addProperty("l1_reg", 0.0, "number");
    this.addProperty("l2_reg", 0.0, "number");
    this.addProperty("num_mixtures", 0, "number");

    this.addProperty("hpo_tunable_key", "learning_rate", "string");
    this.addProperty("hpo_search_type", "choice", "string");
    this.addProperty("hpo_choice_values", "0.001, 0.0003, 0.0001", "string");
    this.addProperty("hpo_range_low", 0, "number");
    this.addProperty("hpo_range_high", 1, "number");
    this.addProperty("hpo_range_q", 1, "number");
    this.addProperty("hpo_include_zero", false, "boolean");
    this.addProperty("hpo_zero_probability", 20, "number");
    this.addProperty("hpo_max_depth", 3, "number");
    this.addProperty("hpo_allowed_widths", "0, 8, 16, 32, 64", "string");
    this.addProperty("hpo_layer_grouping", "single", "string");
    this.addProperty("hpo_tune_architecture", false, "boolean");
    this.addProperty("hpo_architecture_mode", "shared", "string");

    this.addProperty("search_space_builder_json", "{}", "string");
    this.addProperty("search_space_builder_initialized", false, "boolean");
    this.addProperty("resources_json", '{"cpu":1,"gpu":0}', "string");
    this.addProperty(
      "search_space_json",
      JSON.stringify({
        learning_rate: { type: "choice", values: [0.001, 0.0003, 0.0001] },
        dropout: { type: "choice", values: [0, 0.1, 0.2] },
        num_mixtures: { type: "choice", values: [0, 1, 3] },
      }),
      "string",
    );

    this.addProperty("active_tunable_count", 0, "number");
    this.addProperty(
      "active_tunable_summary",
      "No tunable parameters active",
      "string",
    );

    this._templateMetadata = null;
    this._templateByKey = {};
    this._tunableConfigs = {};
    this._draftSpecs = {};
    this._activeEditorKey = null;
    this._editorFieldError = "";
    this._inlineStatus = "";
    this._inlineStatusError = false;

    this._activeSummaryWidget = null;
    this._parameterWidget = null;
    this._searchTypeWidget = null;
    this._choiceValuesWidget = null;
    this._rangeLowWidget = null;
    this._rangeHighWidget = null;
    this._rangeQWidget = null;
    this._includeZeroWidget = null;
    this._zeroProbabilityWidget = null;
    this._maxDepthWidget = null;
    this._allowedWidthsWidget = null;
    this._tuneArchitectureWidget = null;
    this._architectureModeWidget = null;
    this._architectureHelperWidget = null;
    this._primaryActionWidget = null;
    this._inlineStatusWidget = null;

    this._loadBuilderStateFromProperties();

    this.addDropdownPropertyWidget("Trainer Family", "trainer_family", {
      values: ["static", "temporal_rnn", "temporal_transformer", "vae"],
      default: "static",
    });
    this.addDropdownPropertyWidget("Model Architecture", "model_architecture", {
      values: ["ann", "resnet", "bnn"],
      default: "ann",
    });
    this.addDropdownPropertyWidget("Metric Mode", "mode", {
      values: ["min", "max"],
      default: "min",
    });
    this.addDropdownPropertyWidget("Scheduler", "scheduler", {
      values: ["asha", "none"],
      default: "asha",
    });
    this.addDropdownPropertyWidget("Search Algorithm", "search_alg", {
      values: ["optuna", "random"],
      default: "optuna",
    });

    this.addIntPropertyWidget("Trials", "num_samples", { min: 1, default: 10 });
    this.addIntPropertyWidget("Max Epochs", "max_epochs", {
      min: 1,
      default: 10,
    });
    this.addIntPropertyWidget("Max Concurrent", "max_concurrent", {
      min: 1,
      default: 2,
    });
    this.addIntPropertyWidget("Batch Size", "batch_size", {
      min: 1,
      default: 32,
    });
    this.addIntPropertyWidget("Width", "width", { min: 1, default: 32 });
    this.addIntPropertyWidget("Depth", "depth", { min: 1, default: 2 });
    this.addIntPropertyWidget("Sequence Length", "seq_length", {
      min: 1,
      default: 60,
    });
    this.addIntPropertyWidget("Latent Dims", "latent_dims", {
      min: 1,
      default: 8,
    });
    this.addIntPropertyWidget("Num Mixtures", "num_mixtures", {
      min: 0,
      default: 0,
    });

    this.addFloatPropertyWidget("Learning Rate", "learning_rate", {
      min: 1e-6,
      default: 1e-3,
      step: 1e-5,
      precision: 6,
    });
    this.addFloatPropertyWidget("Dropout Rate", "dropout_rate", {
      min: 0,
      max: 0.9,
      default: 0,
      step: 0.1,
      precision: 2,
    });
    this.addBooleanPropertyWidget("Autosave", "autosave", { default: true });

    this._activeSummaryWidget = this._createActiveTunableSummaryWidget();

    this._parameterWidget = this.addDropdownPropertyWidget(
      "Parameter",
      "hpo_tunable_key",
      {
        values: this._templateKeys(),
        default: "learning_rate",
      },
    );

    this._searchTypeWidget = this.addWidget(
      "combo",
      "Tuning Method",
      HPO_METHOD_LABELS.choice,
      (displayLabel) => {
        const methodId = this._methodIdFromDisplayLabel(String(displayLabel));
        this.properties.hpo_search_type = methodId;
        if (this._searchTypeWidget) {
          this._searchTypeWidget.value = this._friendlyMethodLabel(methodId);
        }
        this._setEditorFieldVisibility(methodId);
        this._refreshEditorMeta();
      },
      { values: Object.values(HPO_METHOD_LABELS) },
    );

    this._tuneArchitectureWidget = this.addBooleanPropertyWidget(
      "Tune Architecture",
      "hpo_tune_architecture",
      { default: false },
    );

    this._architectureModeWidget = this.addWidget(
      "combo",
      "Width Behavior",
      "Shared",
      (displayLabel) => {
        const mode = String(displayLabel).trim().toLowerCase();
        this.properties.hpo_architecture_mode =
          mode === "independent" ? "independent" : "shared";
        this._refreshEditorMeta();
        if (
          this.properties.hpo_tunable_key === HPO_ARCHITECTURE_KEY &&
          this.properties.hpo_tune_architecture
        ) {
          this._saveArchitectureFromToggle();
        }
      },
      { values: ["Shared", "Independent"] },
    );

    this._choiceValuesWidget = this.addWidget(
      "text",
      "Choice Values",
      this.properties.hpo_choice_values,
      (v) => {
        this.properties.hpo_choice_values = v;
        this._refreshEditorMeta();
      },
    );

    this._rangeLowWidget = this.addFloatPropertyWidget(
      "Range Low",
      "hpo_range_low",
      {
        default: 0,
        precision: 8,
      },
    );
    this._rangeHighWidget = this.addFloatPropertyWidget(
      "Range High",
      "hpo_range_high",
      {
        default: 1,
        precision: 8,
      },
    );
    this._rangeQWidget = this.addFloatPropertyWidget(
      "Quantization Increment",
      "hpo_range_q",
      {
        min: 0,
        default: 1,
        precision: 8,
      },
    );
    this._includeZeroWidget = this.addBooleanPropertyWidget(
      "Include 0 (turn setting off)",
      "hpo_include_zero",
      { default: false },
    );
    this._zeroProbabilityWidget = this.addFloatPropertyWidget(
      "Off probability (%)",
      "hpo_zero_probability",
      {
        min: 0,
        max: 100,
        default: 20,
        precision: 2,
      },
    );

    this._maxDepthWidget = this.addIntPropertyWidget(
      "Maximum Depth",
      "hpo_max_depth",
      { min: 1, default: 3 },
    );

    this._allowedWidthsWidget = this.addWidget(
      "text",
      "Allowed Widths",
      this.properties.hpo_allowed_widths,
      (v) => {
        this.properties.hpo_allowed_widths = v;
        this._refreshEditorMeta();
        if (
          this.properties.hpo_tunable_key === HPO_ARCHITECTURE_KEY &&
          this.properties.hpo_tune_architecture
        ) {
          this._saveArchitectureFromToggle();
        }
      },
    );

    this._architectureHelperWidget = this.addCustomWidget({
      name: "ArchitectureHelper",
      computeSize(width) {
        return [width, this.hidden ? 0 : 18];
      },
      draw(ctx, _node, width, y) {
        if (!this.value) return;
        ctx.save();
        ctx.font = "11px sans-serif";
        ctx.fillStyle = "#a8a8a8";
        ctx.fillText(this.value, 8, y + 12);
        ctx.restore();
      },
      value: "",
      hidden: true,
    });

    this._primaryActionWidget = this.addWidget(
      "button",
      "Add Tuning",
      null,
      () => {
        this._saveTunableFromEditor();
      },
    );

    this._inlineStatusWidget = this.addCustomWidget({
      name: "InlineStatus",
      computeSize(width) {
        return [width, this.hidden ? 0 : 18];
      },
      draw(ctx, _node, width, y) {
        if (!this.value) return;
        ctx.save();
        ctx.font = "11px sans-serif";
        ctx.fillStyle = this.error ? "#ff7676" : "#9bcf9b";
        ctx.fillText(this.value, 8, y + 12);
        ctx.restore();
      },
      value: "",
      error: false,
      hidden: true,
    });

    this.addWidget("text", "Metric", this.properties.metric, (v) => {
      this.properties.metric = v;
    });
    this.addWidget(
      "text",
      "Resources JSON",
      this.properties.resources_json,
      (v) => {
        this.properties.resources_json = v;
      },
    );

    this._activeEditorKey = this.properties.hpo_tunable_key || "learning_rate";
    this._applyEditorFromSelectedParameter(true);
    this._refreshActiveSummary();
    this._refreshPrimaryActionLabel();
    this._fetchTemplatesAndApply(true);

    this.size = [480, 860];
  }

  _createActiveTunableSummaryWidget() {
    const owner = this;
    const widget = this.addCustomWidget({
      name: "ActiveTunableSummary",
      value: 0,
      _areas: [],
      computeSize(width) {
        const rows = owner._sortedActiveKeys();
        const rowCount = Math.max(1, rows.length);
        return [width, 26 + rowCount * 24];
      },
      draw(ctx, _node, width, y) {
        const rows = owner._sortedActiveKeys();
        this._areas = [];

        ctx.save();
        ctx.fillStyle = "#5f94d6";
        ctx.font = "bold 12px sans-serif";
        ctx.fillText(`Tunable Parameters (${rows.length})`, 8, y + 14);

        if (!rows.length) {
          ctx.fillStyle = "#a8a8a8";
          ctx.font = "11px sans-serif";
          ctx.fillText("None active.", 8, y + 34);
          ctx.restore();
          return;
        }

        let rowY = y + 20;
        for (const key of rows) {
          const spec = owner._tunableConfigs[key];
          const detail = owner._friendlySpecDetail(spec);
          const line = `${owner._labelForKey(key)} - ${detail}`;

          const removeRect = {
            x: width - 18,
            y: rowY + 4,
            w: 12,
            h: 12,
            key,
            action: "remove",
          };
          const rowRect = {
            x: 6,
            y: rowY + 1,
            w: width - 26,
            h: 18,
            key,
            action: "edit",
          };

          ctx.fillStyle = "#d7d7d7";
          ctx.font = "11px sans-serif";
          const maxTextWidth = width - 32;
          let text = line;
          while (
            ctx.measureText(text).width > maxTextWidth &&
            text.length > 6
          ) {
            text = `${text.slice(0, -2)}...`;
          }
          ctx.fillText(text, 8, rowY + 13);

          ctx.strokeStyle = "#c86868";
          ctx.strokeRect(
            removeRect.x,
            removeRect.y,
            removeRect.w,
            removeRect.h,
          );
          ctx.fillStyle = "#ff8d8d";
          ctx.fillText("x", removeRect.x + 4, rowY + 13);

          this._areas.push(removeRect, rowRect);
          rowY += 24;
        }

        ctx.restore();
      },
      mouse(event, pos) {
        if (event.type !== "pointerdown" && event.type !== "mousedown") {
          return false;
        }
        const x = pos[0];
        const y = pos[1];
        for (const area of this._areas || []) {
          if (
            x >= area.x &&
            x <= area.x + area.w &&
            y >= area.y &&
            y <= area.y + area.h
          ) {
            if (area.action === "remove") {
              owner._stopTuningKey(area.key);
            } else {
              owner._editActiveKey(area.key);
            }
            return true;
          }
        }
        return false;
      },
    });
    return widget;
  }

  _setInlineStatus(message, isError = false) {
    this._inlineStatus = String(message || "");
    this._inlineStatusError = !!isError;
    if (this._inlineStatusWidget) {
      this._inlineStatusWidget.value = this._inlineStatus;
      this._inlineStatusWidget.error = this._inlineStatusError;
      this._inlineStatusWidget.hidden = !this._inlineStatus;
    }
    this.setDirtyCanvas(true, true);
  }

  _clearInlineStatus() {
    this._setInlineStatus("", false);
  }

  _safeJsonParse(raw, fallback) {
    if (!raw || !String(raw).trim()) return fallback;
    try {
      return JSON.parse(raw);
    } catch (_err) {
      return fallback;
    }
  }

  _cloneJsonValue(value) {
    return JSON.parse(JSON.stringify(value || {}));
  }

  _isPlainObject(value) {
    return value && typeof value === "object" && !Array.isArray(value);
  }

  _templateKeys() {
    const architectureCfg = this._architectureTuningConfig();
    const hiddenLegacyArchitectureKeys = new Set([
      "width",
      "depth",
      "node_list",
    ]);
    const templates =
      this._templateMetadata && Array.isArray(this._templateMetadata.templates)
        ? this._templateMetadata.templates
            .map((template) => template.key)
            .filter((key) => !hiddenLegacyArchitectureKeys.has(key))
        : [];
    const fallback = HPO_BUILDER_PARAMETER_FALLBACK.filter(
      (key) => !hiddenLegacyArchitectureKeys.has(key),
    );
    const keys = Array.from(new Set([...templates, ...fallback]));
    if (architectureCfg.supported) {
      keys.push(HPO_ARCHITECTURE_KEY);
    }
    return Array.from(new Set(keys));
  }

  _architectureTuningConfig() {
    const cfg =
      (this._templateMetadata && this._templateMetadata.architecture_tuning) ||
      {};
    return {
      supported: !!cfg.supported,
      modes: Array.isArray(cfg.modes) ? cfg.modes : [],
      grouping: cfg.grouping || "single",
      depth_label: cfg.depth_label || "Maximum depth",
      default_max_depth: Number(cfg.default_max_depth || 4),
      default_widths: Array.isArray(cfg.default_widths)
        ? cfg.default_widths
        : [8, 16, 32, 64],
    };
  }

  _isArchitectureKey(key) {
    return String(key) === HPO_ARCHITECTURE_KEY;
  }

  _methodIdFromDisplayLabel(displayLabel) {
    const key = this.properties.hpo_tunable_key || "learning_rate";
    const scoped = HPO_METHOD_LABELS_BY_KEY[key] || {};
    for (const [methodId, label] of Object.entries(scoped)) {
      if (label === displayLabel) return methodId;
    }
    return HPO_METHOD_LABEL_TO_ID[displayLabel] || "choice";
  }

  _sortedActiveKeys() {
    return Object.keys(this._tunableConfigs || {}).sort((a, b) =>
      this._labelForKey(a).localeCompare(this._labelForKey(b)),
    );
  }

  _storeTemplateMetadata(json) {
    this._templateMetadata = json || null;
    this._templateByKey = {};
    for (const template of (json && json.templates) || []) {
      this._templateByKey[template.key] = template;
    }
    this._syncParameterOptions();
    this._updateMethodOptionsForSelectedKey();
  }

  _templateForKey(key) {
    const template = this._templateByKey[key];
    if (template) return template;
    return {
      key,
      label: HPO_BUILDER_LABELS[key] || key,
      default: HPO_BUILDER_DEFAULT_SPECS[key] || { type: "choice", values: [] },
      supported_types: HPO_BUILDER_SEARCH_TYPES,
    };
  }

  _supportedMethodsForKey(key) {
    if (this._isArchitectureKey(key)) {
      return ["layer_structure"];
    }
    const template = this._templateForKey(key);
    let raw = Array.isArray(template.supported_types)
      ? template.supported_types
      : HPO_BUILDER_SEARCH_TYPES;

    // For node_list, support both choice and independent_layer_choice
    if (key === "node_list" && template.supports_independent_layers) {
      raw = [...raw, "independent_layer_choice"];
    }

    const filtered = raw.filter(
      (method) =>
        HPO_BUILDER_SEARCH_TYPES.includes(method) ||
        method === "independent_layer_choice",
    );
    return filtered.length ? filtered : HPO_BUILDER_SEARCH_TYPES;
  }

  _syncParameterOptions() {
    if (!this._parameterWidget) return;
    const values = this._templateKeys();
    this._parameterWidget.options.values = values;
    if (!values.includes(this.properties.hpo_tunable_key)) {
      this.properties.hpo_tunable_key = values[0] || "learning_rate";
      this._parameterWidget.value = this.properties.hpo_tunable_key;
    }
  }

  _labelForKey(key) {
    if (this._isArchitectureKey(key)) {
      return "Architecture";
    }
    // Special case: show custom label for independent layer tuning
    if (key === "node_list" && this._tunableConfigs[key]) {
      const spec = this._tunableConfigs[key];
      if (spec.type === "independent_layer_choice") {
        const grouping = spec.grouping || "single";
        return grouping === "paired_equal"
          ? "Paired Layer Widths"
          : "Layer Widths";
      }
    }
    const template = this._templateForKey(key);
    return template.label || HPO_BUILDER_LABELS[key] || key;
  }

  _friendlyMethodLabel(methodId) {
    const key = this.properties.hpo_tunable_key || "learning_rate";
    if (
      HPO_METHOD_LABELS_BY_KEY[key] &&
      HPO_METHOD_LABELS_BY_KEY[key][methodId]
    ) {
      return HPO_METHOD_LABELS_BY_KEY[key][methodId];
    }
    return HPO_METHOD_LABELS[methodId] || methodId;
  }

  _friendlySpecDetail(spec) {
    if (!spec) return "Fixed value";
    const type = String(spec.type || "choice");
    if (type === "layer_structure") {
      const mode = String(spec.mode || "shared");
      const grouping = String(spec.grouping || "single");
      const maxDepth = Number(spec.max_depth || 1);
      const widths = this._formatChoiceValues(spec.widths || []);
      if (mode === "shared") {
        return `shared width [${widths}], depth 1-${maxDepth}`;
      }
      const unit = grouping === "paired_equal" ? "pairs" : "layers";
      return `up to ${maxDepth} ${unit}, independent widths [${widths}]`;
    }
    if (type === "independent_layer_choice") {
      const grouping = spec.grouping || "single";
      const maxDepth = spec.max_depth || 1;
      const depthLabel = grouping === "paired_equal" ? "pairs" : "layers";
      return `up to ${maxDepth} ${depthLabel}; independently choose ${this._formatChoiceValues(spec.values || [])}`;
    }
    if (type === "choice") {
      return `${this._friendlyMethodLabel(type)}: ${this._formatChoiceValues(spec.values || [])}`;
    }
    if (type === "uniform" || type === "randint") {
      return `${this._friendlyMethodLabel(type)}: ${spec.low} to ${spec.high}`;
    }
    if (type === "loguniform") {
      if (spec.include_zero) {
        const zeroLabel = this._zeroLabelForKey(
          spec.__key_for_summary || this.properties.hpo_tunable_key,
        );
        return `${zeroLabel ? `0 (${zeroLabel})` : "0"} or log range ${spec.low} to ${spec.high}`;
      }
      return `log range ${spec.low} to ${spec.high}`;
    }
    if (type === "qloguniform") {
      if (spec.include_zero) {
        const zeroLabel = this._zeroLabelForKey(
          spec.__key_for_summary || this.properties.hpo_tunable_key,
        );
        return `${zeroLabel ? `0 (${zeroLabel})` : "0"} or log range ${spec.low} to ${spec.high} (step ${spec.q})`;
      }
      return `log range ${spec.low} to ${spec.high} (step ${spec.q})`;
    }
    return `${this._friendlyMethodLabel(type)}: ${spec.low} to ${spec.high} (step ${spec.q})`;
  }

  _allowZeroForKey(key) {
    const template = this._templateForKey(key);
    return !!template.allow_zero;
  }

  _zeroLabelForKey(key) {
    const template = this._templateForKey(key);
    const label = template.zero_label;
    return label == null ? "" : String(label);
  }

  _propertyForKey(key) {
    return HPO_BUILDER_PROPERTY_BY_KEY[key] || key;
  }

  _currentValueForKey(key) {
    const propertyName = this._propertyForKey(key);
    return this.properties[propertyName];
  }

  _getLayerGroupingForKey(key) {
    if (this._isArchitectureKey(key)) {
      return this._architectureTuningConfig().grouping;
    }
    const template = this._templateForKey(key);
    const arch = this.properties.model_architecture || "ann";
    if (!template.layer_grouping_by_architecture) return null;
    return template.layer_grouping_by_architecture[arch] || null;
  }

  _shouldShowIndependentLayersForKey(key) {
    if (key !== "node_list") return false;
    const template = this._templateForKey(key);
    return !!template.supports_independent_layers;
  }

  _parseScalarValue(raw) {
    const text = String(raw ?? "").trim();
    if (!text) return null;
    if (text === "true") return true;
    if (text === "false") return false;
    if (text === "null") return null;
    const numeric = Number(text);
    if (Number.isFinite(numeric)) return numeric;
    try {
      return JSON.parse(text);
    } catch (_err) {
      return text;
    }
  }

  _parseChoiceValues(raw) {
    const text = String(raw || "").trim();
    if (!text) return [];
    if (text.startsWith("[")) {
      const parsed = JSON.parse(text);
      return Array.isArray(parsed) ? parsed : [parsed];
    }
    return text
      .split(",")
      .map((part) => this._parseScalarValue(part))
      .filter((value) => value !== null && value !== "");
  }

  _formatValue(value) {
    if (Array.isArray(value) || this._isPlainObject(value)) {
      return JSON.stringify(value);
    }
    return String(value ?? "");
  }

  _formatChoiceValues(values) {
    if (!Array.isArray(values)) return "";
    if (
      values.some((value) => Array.isArray(value) || this._isPlainObject(value))
    ) {
      return JSON.stringify(values);
    }
    return values.map((value) => this._formatValue(value)).join(", ");
  }

  _loadBuilderStateFromProperties() {
    const builderState = this._safeJsonParse(
      this.properties.search_space_builder_json,
      {},
    );
    const searchSpace = this._safeJsonParse(
      this.properties.search_space_json,
      {},
    );
    const hasBuilderKeys =
      this._isPlainObject(builderState) && Object.keys(builderState).length > 0;
    const hasSearchKeys =
      this._isPlainObject(searchSpace) && Object.keys(searchSpace).length > 0;
    const isBuilderInitialized =
      !!this.properties.search_space_builder_initialized;

    if (isBuilderInitialized) {
      this._tunableConfigs = this._isPlainObject(builderState)
        ? this._cloneJsonValue(builderState)
        : {};
    } else if (hasBuilderKeys) {
      this._tunableConfigs = this._cloneJsonValue(builderState);
    } else if (hasSearchKeys) {
      this._tunableConfigs = this._cloneJsonValue(searchSpace);
    } else {
      this._tunableConfigs = {};
    }

    this._migrateLegacyFixedSpecs();

    for (const [key, spec] of Object.entries(this._tunableConfigs)) {
      this._draftSpecs[key] = this._cloneJsonValue(spec);
    }

    this._persistBuilderState(false);
  }

  _persistBuilderState(updateSearchSpace = true) {
    this.properties.search_space_builder_json = JSON.stringify(
      this._tunableConfigs,
    );
    this.properties.search_space_builder_initialized = true;
    if (updateSearchSpace) {
      this.properties.search_space_json = JSON.stringify(
        this._tunableConfigs,
        null,
        2,
      );
    }
  }

  _migrateLegacyFixedSpecs() {
    for (const [key, spec] of Object.entries(this._tunableConfigs)) {
      if (!this._isPlainObject(spec)) continue;
      if (spec.type !== "fixed") continue;

      const prop = this._propertyForKey(key);
      const hasUnambiguousValue = Object.prototype.hasOwnProperty.call(
        spec,
        "value",
      );
      const propExists = Object.prototype.hasOwnProperty.call(
        this.properties,
        prop,
      );
      if (
        hasUnambiguousValue &&
        propExists &&
        this.properties[prop] === undefined
      ) {
        this.properties[prop] = spec.value;
      }

      delete this._tunableConfigs[key];
    }
  }

  _syncSearchSpaceFromBuilder() {
    this._persistBuilderState(true);
    if (this._runner) this._runner.invalidate();
    this._refreshActiveSummary();
  }

  _refreshPrimaryActionLabel() {
    const key = this.properties.hpo_tunable_key || "learning_rate";
    if (this._isArchitectureKey(key)) {
      if (this._primaryActionWidget) {
        this._primaryActionWidget.hidden = true;
      }
      return;
    }
    if (this._primaryActionWidget) {
      this._primaryActionWidget.hidden = false;
    }
    const isActive = !!this._activeSpecForKey(key);
    if (this._primaryActionWidget) {
      this._primaryActionWidget.name = isActive
        ? "Update Tuning"
        : "Add Tuning";
    }
  }

  _setWidgetValue(widget, propertyName, value) {
    this.properties[propertyName] = value;
    if (widget) widget.value = value;
  }

  _setEditorFieldVisibility(methodId) {
    const key = this.properties.hpo_tunable_key || "learning_rate";
    const isArchitecture = this._isArchitectureKey(key);
    const architectureEnabled = !!this.properties.hpo_tune_architecture;
    const architectureMode = String(
      this.properties.hpo_architecture_mode || "shared",
    );
    const isChoice = methodId === "choice";
    const isSimpleRange = ["uniform", "loguniform", "randint"].includes(
      methodId,
    );
    const isQuantized = ["quniform", "qloguniform"].includes(methodId);
    const isIndependentLayers =
      methodId === "independent_layer_choice" && key === "node_list";
    const canIncludeZero = this._allowZeroForKey(key);
    const isLogType = methodId === "loguniform" || methodId === "qloguniform";

    if (isArchitecture) {
      const cfg = this._architectureTuningConfig();
      if (this._searchTypeWidget) this._searchTypeWidget.hidden = true;
      if (this._choiceValuesWidget) this._choiceValuesWidget.hidden = true;
      if (this._rangeLowWidget) this._rangeLowWidget.hidden = true;
      if (this._rangeHighWidget) this._rangeHighWidget.hidden = true;
      if (this._rangeQWidget) this._rangeQWidget.hidden = true;
      if (this._includeZeroWidget) this._includeZeroWidget.hidden = true;
      if (this._zeroProbabilityWidget)
        this._zeroProbabilityWidget.hidden = true;
      if (this._tuneArchitectureWidget)
        this._tuneArchitectureWidget.hidden = false;
      if (this._architectureModeWidget) {
        this._architectureModeWidget.hidden = !architectureEnabled;
      }
      if (this._maxDepthWidget) {
        this._maxDepthWidget.hidden = !architectureEnabled;
        this._maxDepthWidget.name = cfg.depth_label || "Maximum depth";
      }
      if (this._allowedWidthsWidget)
        this._allowedWidthsWidget.hidden = !architectureEnabled;
      if (this._architectureHelperWidget) {
        this._architectureHelperWidget.hidden = !architectureEnabled;
      }
      this._setArchitectureHelperText(
        architectureEnabled,
        architectureMode,
        cfg.grouping,
      );
      return;
    }

    if (this._tuneArchitectureWidget)
      this._tuneArchitectureWidget.hidden = true;
    if (this._architectureModeWidget)
      this._architectureModeWidget.hidden = true;
    if (this._architectureHelperWidget)
      this._architectureHelperWidget.hidden = true;
    if (this._searchTypeWidget) this._searchTypeWidget.hidden = false;

    if (this._choiceValuesWidget) this._choiceValuesWidget.hidden = !isChoice;
    if (this._rangeLowWidget)
      this._rangeLowWidget.hidden = !(isSimpleRange || isQuantized);
    if (this._rangeHighWidget)
      this._rangeHighWidget.hidden = !(isSimpleRange || isQuantized);
    if (this._rangeQWidget) this._rangeQWidget.hidden = !isQuantized;
    if (this._includeZeroWidget) {
      this._includeZeroWidget.hidden = !(isLogType && canIncludeZero);
      if (this._includeZeroWidget.hidden) {
        this._setWidgetValue(
          this._includeZeroWidget,
          "hpo_include_zero",
          false,
        );
      }
    }
    if (this._zeroProbabilityWidget) {
      const showZeroProbability =
        !!this._includeZeroWidget &&
        !this._includeZeroWidget.hidden &&
        !!this.properties.hpo_include_zero;
      this._zeroProbabilityWidget.hidden = !showZeroProbability;
    }
    if (this._maxDepthWidget)
      this._maxDepthWidget.hidden = !isIndependentLayers;
    if (this._allowedWidthsWidget)
      this._allowedWidthsWidget.hidden = !isIndependentLayers;
  }

  _updateMethodOptionsForSelectedKey() {
    const key = this.properties.hpo_tunable_key || "learning_rate";
    if (this._isArchitectureKey(key)) {
      this.properties.hpo_search_type = "layer_structure";
      if (this._searchTypeWidget) {
        this._searchTypeWidget.options.values = [];
      }
      this._setEditorFieldVisibility("layer_structure");
      return;
    }
    const methods = this._supportedMethodsForKey(key);
    const labels = methods.map((method) => this._friendlyMethodLabel(method));
    if (this._searchTypeWidget) {
      this._searchTypeWidget.options.values = labels;
      const current = this.properties.hpo_search_type;
      const safe = methods.includes(current) ? current : methods[0] || "choice";
      this.properties.hpo_search_type = safe;
      this._searchTypeWidget.value = this._friendlyMethodLabel(safe);
      this._setEditorFieldVisibility(safe);
    }
  }

  _activeSpecForKey(key) {
    return this._tunableConfigs[key] || null;
  }

  _recommendedSpecForKey(key) {
    if (this._isArchitectureKey(key)) {
      const cfg = this._architectureTuningConfig();
      return {
        type: "layer_structure",
        mode: "shared",
        max_depth: Math.max(1, parseInt(cfg.default_max_depth, 10) || 4),
        widths: cfg.default_widths,
        grouping: cfg.grouping,
        zero_behavior: "omit",
      };
    }
    const template = this._templateForKey(key);
    return this._cloneJsonValue(
      template.default ||
        HPO_BUILDER_DEFAULT_SPECS[key] || { type: "choice", values: [] },
    );
  }

  _setEditorSpec(spec) {
    if (this._isArchitectureKey(this.properties.hpo_tunable_key)) {
      const cfg = this._architectureTuningConfig();
      const safe = this._cloneJsonValue(spec || {});
      const isActive = safe && safe.type === "layer_structure";
      const mode = String(
        safe.mode || this.properties.hpo_architecture_mode || "shared",
      ).toLowerCase();
      const widths =
        Array.isArray(safe.widths) && safe.widths.length
          ? safe.widths
          : cfg.default_widths;
      const maxDepth = Number(
        safe.max_depth ||
          this.properties.hpo_max_depth ||
          cfg.default_max_depth,
      );

      this._setWidgetValue(
        this._tuneArchitectureWidget,
        "hpo_tune_architecture",
        isActive,
      );
      this._setWidgetValue(
        this._architectureModeWidget,
        "hpo_architecture_mode",
        mode === "independent" ? "independent" : "shared",
      );
      if (this._architectureModeWidget) {
        this._architectureModeWidget.value =
          this.properties.hpo_architecture_mode === "independent"
            ? "Independent"
            : "Shared";
      }
      this._setWidgetValue(
        this._maxDepthWidget,
        "hpo_max_depth",
        Math.max(1, parseInt(maxDepth, 10) || 1),
      );
      this._setWidgetValue(
        this._allowedWidthsWidget,
        "hpo_allowed_widths",
        this._formatChoiceValues(widths),
      );
      this._setEditorFieldVisibility("layer_structure");
      this._refreshPrimaryActionLabel();
      this._refreshEditorMeta();
      return;
    }

    const safe = this._cloneJsonValue(spec || { type: "choice", values: [] });
    const methods = this._supportedMethodsForKey(
      this.properties.hpo_tunable_key,
    );
    const type = methods.includes(safe.type)
      ? safe.type
      : methods[0] || "choice";

    this._setWidgetValue(this._searchTypeWidget, "hpo_search_type", type);
    if (this._searchTypeWidget) {
      this._searchTypeWidget.value = this._friendlyMethodLabel(type);
    }

    if (type === "independent_layer_choice") {
      this._setWidgetValue(
        this._maxDepthWidget,
        "hpo_max_depth",
        Number(safe.max_depth ?? 3),
      );
      this._setWidgetValue(
        this._allowedWidthsWidget,
        "hpo_allowed_widths",
        this._formatChoiceValues(safe.values || []),
      );
    } else if (type === "choice") {
      this._setWidgetValue(
        this._choiceValuesWidget,
        "hpo_choice_values",
        this._formatChoiceValues(safe.values || []),
      );
      this._setWidgetValue(this._includeZeroWidget, "hpo_include_zero", false);
      this._setWidgetValue(
        this._zeroProbabilityWidget,
        "hpo_zero_probability",
        20,
      );
    } else {
      this._setWidgetValue(
        this._rangeLowWidget,
        "hpo_range_low",
        Number(safe.low ?? 0),
      );
      this._setWidgetValue(
        this._rangeHighWidget,
        "hpo_range_high",
        Number(safe.high ?? 1),
      );
      this._setWidgetValue(
        this._rangeQWidget,
        "hpo_range_q",
        Number(safe.q ?? 1),
      );
      const includeZero =
        (type === "loguniform" || type === "qloguniform") &&
        this._allowZeroForKey(this.properties.hpo_tunable_key)
          ? !!safe.include_zero
          : false;
      this._setWidgetValue(
        this._includeZeroWidget,
        "hpo_include_zero",
        includeZero,
      );
      const zeroProbPct = includeZero
        ? Number.isFinite(Number(safe.zero_probability))
          ? Number(safe.zero_probability) * 100
          : 20
        : 20;
      this._setWidgetValue(
        this._zeroProbabilityWidget,
        "hpo_zero_probability",
        zeroProbPct,
      );
    }

    this._setEditorFieldVisibility(type);
    this._refreshEditorMeta();
    this._refreshPrimaryActionLabel();
  }

  _collectEditorSpec() {
    const key = this.properties.hpo_tunable_key || "learning_rate";
    const type = String(this.properties.hpo_search_type || "choice").trim();

    if (this._isArchitectureKey(key)) {
      const cfg = this._architectureTuningConfig();
      const mode = String(this.properties.hpo_architecture_mode || "shared")
        .trim()
        .toLowerCase();
      try {
        const maxDepth = Math.max(
          1,
          parseInt(this.properties.hpo_max_depth, 10) || 1,
        );
        const widths = this._parseChoiceValues(
          this.properties.hpo_allowed_widths,
        ).map((w) => {
          const n = Number(w);
          if (!Number.isFinite(n) || n < 0 || Math.floor(n) !== n) {
            throw new Error("all widths must be non-negative integers");
          }
          return n;
        });
        if (!widths.length) {
          return {
            spec: null,
            error: "Allowed Widths: must contain at least one value",
          };
        }
        if (!widths.some((w) => w > 0)) {
          return {
            spec: null,
            error: "Allowed Widths: at least one positive width is required",
          };
        }
        if (mode === "shared" && widths.some((w) => w === 0)) {
          return {
            spec: null,
            error: "Shared width behavior cannot include 0 in Allowed Widths",
          };
        }
        return {
          spec: {
            type: "layer_structure",
            mode: mode === "independent" ? "independent" : "shared",
            max_depth: maxDepth,
            widths,
            grouping: cfg.grouping,
            zero_behavior: "omit",
          },
          error: null,
        };
      } catch (err) {
        return { spec: null, error: `Architecture: ${err.message}` };
      }
    }

    if (type === "independent_layer_choice") {
      try {
        const maxDepth = Math.max(
          1,
          parseInt(this.properties.hpo_max_depth) || 1,
        );
        const widths = this._parseChoiceValues(
          this.properties.hpo_allowed_widths,
        );
        if (!widths || !widths.length) {
          return {
            spec: null,
            error: "Allowed Widths: must contain at least one value",
          };
        }
        // Ensure all widths are integers
        const intWidths = widths.map((w) => {
          const n = Number(w);
          if (!Number.isFinite(n) || n < 0 || Math.floor(n) !== n) {
            throw new Error("all widths must be non-negative integers");
          }
          return n;
        });
        if (!intWidths.some((w) => w > 0)) {
          return {
            spec: null,
            error: "Allowed Widths: at least one positive width is required",
          };
        }
        const grouping = this._getLayerGroupingForKey(key);
        if (!grouping) {
          return {
            spec: null,
            error:
              "This parameter does not support independent layer tuning for the selected architecture",
          };
        }
        return {
          spec: {
            type,
            max_depth: maxDepth,
            values: intWidths,
            zero_behavior: "omit",
            grouping,
          },
          error: null,
        };
      } catch (err) {
        return {
          spec: null,
          error: `Independent Layer Choices: ${err.message}`,
        };
      }
    }

    if (!HPO_BUILDER_SEARCH_TYPES.includes(type)) {
      return { spec: null, error: "Tuning Method: unsupported method" };
    }

    if (type === "choice") {
      try {
        const values = this._parseChoiceValues(
          this.properties.hpo_choice_values,
        );
        return { spec: { type, values }, error: null };
      } catch (err) {
        return { spec: null, error: `Choice Values: ${err.message}` };
      }
    }

    const low = Number(this.properties.hpo_range_low);
    const high = Number(this.properties.hpo_range_high);
    if (!Number.isFinite(low) || !Number.isFinite(high)) {
      return { spec: null, error: "Range Low/Range High: must be numeric" };
    }

    const spec = { type, low, high };
    if (type === "quniform" || type === "qloguniform") {
      const q = Number(this.properties.hpo_range_q);
      if (!Number.isFinite(q)) {
        return { spec: null, error: "Range Step: must be numeric" };
      }
      spec.q = q;
    }
    if (
      (type === "loguniform" || type === "qloguniform") &&
      this._allowZeroForKey(key)
    ) {
      spec.include_zero = !!this.properties.hpo_include_zero;
      if (spec.include_zero) {
        const rawPct = Number(this.properties.hpo_zero_probability);
        if (!Number.isFinite(rawPct)) {
          return {
            spec: null,
            error: "Off probability (%): must be numeric",
          };
        }
        spec.zero_probability = rawPct / 100;
      }
    }

    // Keep current validation behavior for VAE mixture constraints.
    if (key === "num_mixtures" && this.properties.trainer_family === "vae") {
      if (type === "randint" && spec.low < 1) {
        return {
          spec: null,
          error: "Range Low: VAE num_mixtures must start at 1",
        };
      }
    }

    return { spec, error: null };
  }

  _validateEditorSpec(key, spec) {
    // Allow independent_layer_choice as valid type
    const validTypes = [
      ...HPO_BUILDER_SEARCH_TYPES,
      "independent_layer_choice",
    ];
    if (this._isArchitectureKey(key)) {
      if (spec.type !== "layer_structure") {
        throw new Error("Architecture tuning requires layer_structure type");
      }
      if (!["shared", "independent"].includes(String(spec.mode))) {
        throw new Error("Architecture mode must be Shared or Independent");
      }
      if (!Array.isArray(spec.widths) || !spec.widths.length) {
        throw new Error("Architecture widths must contain at least one value");
      }
      if (!spec.widths.some((v) => Number(v) > 0)) {
        throw new Error(
          "Architecture widths must contain at least one positive value",
        );
      }
      if (
        String(spec.mode) === "shared" &&
        spec.widths.some((v) => Number(v) === 0)
      ) {
        throw new Error("Shared architecture mode cannot include zero width");
      }
      const conflicts = ["width", "depth", "node_list"].filter(
        (k) => !!this._tunableConfigs[k],
      );
      if (conflicts.length) {
        throw new Error(
          "Remove Width, Depth, and Node List tunables before enabling Architecture tuning.",
        );
      }
      return;
    }

    if (!validTypes.includes(spec.type)) {
      throw new Error(`Unsupported search type '${spec.type}'`);
    }

    if (spec.type === "independent_layer_choice") {
      if (key !== "node_list") {
        throw new Error(
          "Independent layer choice is only supported for node_list",
        );
      }
      if (!Array.isArray(spec.values) || !spec.values.length) {
        throw new Error("At least one width is required");
      }
      if (!spec.values.some((v) => Number(v) > 0)) {
        throw new Error("At least one positive width is required");
      }
      // Check for conflicts with width or depth tuning
      if (this._tunableConfigs.width || this._tunableConfigs.depth) {
        throw new Error(
          "Remove Width and Depth from the tunable list before enabling independent layer widths.",
        );
      }
      if (
        this._tunableConfigs.node_list &&
        this._tunableConfigs.node_list.type !== "independent_layer_choice"
      ) {
        throw new Error(
          "Cannot combine Whole Architecture Choices with Independent Layer Choices.",
        );
      }
      return;
    }

    if (Object.prototype.hasOwnProperty.call(spec, "include_zero")) {
      if (typeof spec.include_zero !== "boolean") {
        throw new Error("Include 0: must be boolean");
      }
      if (!["loguniform", "qloguniform"].includes(spec.type)) {
        throw new Error("Include 0 is only supported for logarithmic methods");
      }
      if (spec.include_zero && !this._allowZeroForKey(key)) {
        throw new Error(
          `${key} does not permit zero in its logarithmic search space`,
        );
      }
      if (spec.include_zero) {
        const zp = Number(spec.zero_probability ?? 0.2);
        if (!Number.isFinite(zp)) {
          throw new Error("Off probability (%): must be numeric");
        }
        if (zp < 0 || zp > 1) {
          throw new Error("Off probability (%): must be between 0 and 100");
        }
        spec.zero_probability = zp;
      } else {
        spec.zero_probability = 0.0;
      }
    }

    if (spec.type === "choice") {
      if (!Array.isArray(spec.values) || !spec.values.length) {
        throw new Error("Choice Values: at least one value is required");
      }
      if (key === "num_mixtures" && this.properties.trainer_family === "vae") {
        if (spec.values.some((value) => Number(value) < 1)) {
          throw new Error(
            "Choice Values: VAE num_mixtures values must be >= 1",
          );
        }
      }
      return;
    }

    if (!Number.isFinite(spec.low) || !Number.isFinite(spec.high)) {
      throw new Error("Range Low/Range High: must be numeric");
    }
    if (spec.low >= spec.high) {
      throw new Error("Range Low/Range High: low must be less than high");
    }
    if (
      (spec.type === "loguniform" || spec.type === "qloguniform") &&
      spec.low <= 0
    ) {
      throw new Error("Range Low: logarithmic ranges require low > 0");
    }
    if (spec.type === "quniform" || spec.type === "qloguniform") {
      if (!Number.isFinite(spec.q) || spec.q <= 0) {
        throw new Error("Range Step: quantized ranges require a positive step");
      }
    }
    if (spec.type === "randint") {
      spec.low = Math.round(spec.low);
      spec.high = Math.round(spec.high);
      if (
        key === "num_mixtures" &&
        this.properties.trainer_family === "vae" &&
        spec.low < 1
      ) {
        throw new Error(
          "Range Low: VAE num_mixtures must start at 1 or higher",
        );
      }
    }
  }

  _refreshEditorMeta() {
    if (this._isArchitectureKey(this.properties.hpo_tunable_key)) {
      const { spec, error } = this._collectEditorSpec();
      this._editorFieldError = error || "";
      if (this._editorFieldError) {
        this._setInlineStatus(this._editorFieldError, true);
      } else {
        const tuneOn = !!this.properties.hpo_tune_architecture;
        if (tuneOn && spec) {
          const mode = spec.mode === "shared" ? "Shared" : "Independent";
          this._setInlineStatus(`Architecture tuning enabled (${mode})`, false);
        } else {
          this._setInlineStatus(this._fixedArchitectureSummary(), false);
        }
      }

      if (spec && !error) {
        this._draftSpecs[HPO_ARCHITECTURE_KEY] = this._cloneJsonValue(spec);
      }
      this._refreshPrimaryActionLabel();
      this.setDirtyCanvas(true, true);
      return;
    }

    const { spec, error } = this._collectEditorSpec();
    this._editorFieldError = error || "";
    if (this._editorFieldError) {
      this._setInlineStatus(this._editorFieldError, true);
    } else if (!this._inlineStatusError) {
      this._clearInlineStatus();
    }

    if (spec && !error) {
      this._draftSpecs[this.properties.hpo_tunable_key || "learning_rate"] =
        this._cloneJsonValue(spec);
    }

    this._refreshPrimaryActionLabel();

    this.setDirtyCanvas(true, true);
  }

  _refreshActiveSummary() {
    const keys = this._sortedActiveKeys();
    this.properties.active_tunable_count = keys.length;
    this.properties.active_tunable_summary =
      keys.length === 0
        ? "No tunable parameters active"
        : keys
            .map(
              (key) =>
                `${this._labelForKey(key)}: ${this._friendlySpecDetail({
                  ...this._tunableConfigs[key],
                  __key_for_summary: key,
                })}`,
            )
            .join(" | ");

    if (
      this._activeSummaryWidget &&
      typeof this._activeSummaryWidget.computeSize === "function"
    ) {
      // force widget to recompute height next draw
      this._activeSummaryWidget.value += 1;
    }

    this.setDirtyCanvas(true, true);
  }

  _applyEditorFromSelectedParameter(force = false) {
    const key = this.properties.hpo_tunable_key || "learning_rate";

    this._activeEditorKey = key;
    this._updateMethodOptionsForSelectedKey();

    if (this._isArchitectureKey(key)) {
      const activeSpec = this._activeSpecForKey(HPO_ARCHITECTURE_KEY);
      this._setEditorSpec(activeSpec || null);
      return;
    }

    const activeSpec = this._activeSpecForKey(key);
    const seed = activeSpec || this._recommendedSpecForKey(key);
    this._draftSpecs[key] = this._cloneJsonValue(seed);
    this._setEditorSpec(seed);
    this._clearInlineStatus();
  }

  _setArchitectureHelperText(enabled, mode, grouping) {
    if (!this._architectureHelperWidget) return;
    if (!enabled) {
      this._architectureHelperWidget.value = "";
      return;
    }
    const independent = String(mode || "shared") === "independent";
    if (!independent) {
      this._architectureHelperWidget.value =
        "Shared mode samples one width and one depth.";
      return;
    }
    if (grouping === "paired_equal") {
      this._architectureHelperWidget.value =
        "Each pair is sampled independently. Both layers in a pair use the same width. Zero removes the pair.";
      return;
    }
    this._architectureHelperWidget.value =
      "Each layer is sampled independently. Zero removes that layer.";
  }

  _fixedArchitectureSummary() {
    const nodeList = this.properties.node_list;
    if (Array.isArray(nodeList) && nodeList.length) {
      return `Fixed architecture: [${nodeList.join(", ")}]`;
    }
    return `Fixed architecture: Width ${this.properties.width} x Depth ${this.properties.depth}`;
  }

  async _saveArchitectureFromToggle() {
    try {
      const enabled = !!this.properties.hpo_tune_architecture;
      const candidate = this._cloneJsonValue(this._tunableConfigs);

      if (!enabled) {
        delete candidate[HPO_ARCHITECTURE_KEY];
      } else {
        const { spec, error } = this._collectEditorSpec();
        if (error || !spec) {
          this._setInlineStatus(
            error || "Architecture settings are invalid",
            true,
          );
          return;
        }
        this._validateEditorSpec(HPO_ARCHITECTURE_KEY, spec);
        candidate[HPO_ARCHITECTURE_KEY] = this._cloneJsonValue(spec);
      }

      await this._validateCandidateSearchSpace(candidate);
      this._tunableConfigs = candidate;
      this._syncSearchSpaceFromBuilder();
      this._refreshEditorMeta();
    } catch (err) {
      this._setInlineStatus(err.message, true);
      this._setWidgetValue(
        this._tuneArchitectureWidget,
        "hpo_tune_architecture",
        !!this._tunableConfigs[HPO_ARCHITECTURE_KEY],
      );
    }
  }

  async _saveTunableFromEditor() {
    try {
      const key = this.properties.hpo_tunable_key || "learning_rate";
      if (this._isArchitectureKey(key)) {
        await this._saveArchitectureFromToggle();
        return;
      }
      const { spec, error } = this._collectEditorSpec();
      if (error || !spec) {
        this._setInlineStatus(error || "Editor input is invalid", true);
        return;
      }

      this._validateEditorSpec(key, spec);

      const candidate = this._cloneJsonValue(this._tunableConfigs);
      candidate[key] = this._cloneJsonValue(spec);
      await this._validateCandidateSearchSpace(candidate);

      this._tunableConfigs = candidate;
      this._draftSpecs[key] = this._cloneJsonValue(spec);
      this._syncSearchSpaceFromBuilder();
      this._clearInlineStatus();
      this._refreshEditorMeta();
    } catch (err) {
      this._setInlineStatus(err.message, true);
    }
  }

  async _validateCandidateSearchSpace(candidate) {
    const json = await this._postJson("/validate_hpo_search_space", {
      trainer_family: this.properties.trainer_family,
      model_architecture: this.properties.model_architecture,
      search_space: candidate,
    });
    if (!json.ok) {
      const msg = (json.errors || []).join("; ") || "Invalid search space";
      throw new Error(msg);
    }
  }

  _stopTuningSelected() {
    const key = this.properties.hpo_tunable_key || "learning_rate";
    this._stopTuningKey(key);
  }

  _stopTuningKey(key) {
    if (!this._tunableConfigs[key]) {
      this._clearInlineStatus();
      return;
    }

    delete this._tunableConfigs[key];
    this._syncSearchSpaceFromBuilder();

    if (this._isArchitectureKey(key)) {
      this._setWidgetValue(
        this._tuneArchitectureWidget,
        "hpo_tune_architecture",
        false,
      );
    }

    const fallback = this._recommendedSpecForKey(key);
    this._draftSpecs[key] = this._cloneJsonValue(fallback);

    if (this.properties.hpo_tunable_key === key) {
      this._setEditorSpec(fallback);
    }
    this._clearInlineStatus();
  }

  _editActiveKey(key) {
    this.properties.hpo_tunable_key = key;
    if (this._parameterWidget) this._parameterWidget.value = key;
    this._applyEditorFromSelectedParameter(true);
  }

  _editManualSearchSpaceJson() {
    const current = this.properties.search_space_json || "{}";
    const next = window.prompt("Advanced Search Space JSON", current);
    if (next === null) return;

    try {
      const parsed = JSON.parse(next || "{}");
      if (!this._isPlainObject(parsed)) {
        throw new Error("Search space must be a JSON object");
      }

      this._tunableConfigs = this._cloneJsonValue(parsed);
      this._persistBuilderState(true);

      this._draftSpecs = {};
      for (const [key, spec] of Object.entries(this._tunableConfigs)) {
        this._draftSpecs[key] = this._cloneJsonValue(spec);
      }
      if (!this._draftSpecs[this.properties.hpo_tunable_key]) {
        this._draftSpecs[this.properties.hpo_tunable_key] =
          this._recommendedSpecForKey(this.properties.hpo_tunable_key);
      }

      this._refreshActiveSummary();
      this._refreshEditorMeta();
      this._clearInlineStatus();
    } catch (err) {
      this._setInlineStatus(`JSON import error: ${err.message}`, true);
    }
  }

  async _fetchTemplatesAndApply(preserveActive = true) {
    try {
      const json = await this._postJson("/melt_hpo_parameter_templates", {
        trainer_family: this.properties.trainer_family,
        model_architecture: this.properties.model_architecture,
      });
      this._storeTemplateMetadata(json);

      if (!preserveActive || !Object.keys(this._tunableConfigs).length) {
        const next = this._cloneJsonValue(json.default_search_space || {});
        if (Object.keys(next).length) {
          this._tunableConfigs = next;
          this._persistBuilderState(true);
        }
      }

      this._syncParameterOptions();
      this._applyEditorFromSelectedParameter(true);
      this._refreshActiveSummary();
    } catch (err) {
      console.error("HPO template load failed", err.detail || err);
      this._setInlineStatus(`Template load failed: ${err.message}`, true);
    }
  }

  _parseJsonProperty(name, fallback) {
    const raw = this.properties[name];
    if (!raw || !String(raw).trim()) return fallback;
    try {
      return JSON.parse(raw);
    } catch (err) {
      throw new Error(`${name} must contain valid JSON: ${err.message}`);
    }
  }

  async _postJson(endpoint, payload) {
    const response = await window.MeltApi.fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const text = await response.text();
    let json = null;
    try {
      json = text ? JSON.parse(text) : null;
    } catch (_err) {
      json = { __raw_text: text };
    }

    if (!response.ok) {
      const message = json && json.error ? json.error : response.statusText;
      const error = new Error(message);
      error.detail = json;
      throw error;
    }

    return json || {};
  }

  async fetch(resolvedInput) {
    const endpoint = this.properties.endpoint || "/melt_hyperparameter_tuner";
    this._syncSearchSpaceFromBuilder();

    let x = null;
    let y = null;
    let lengths = null;
    if (Array.isArray(resolvedInput)) {
      [x, y, lengths] = resolvedInput;
    } else if (resolvedInput && typeof resolvedInput === "object") {
      x = resolvedInput.x;
      y = resolvedInput.y;
      lengths = resolvedInput.lengths;
    }
    if (!x) x = this.getInputData(0);
    if (!y) y = this.getInputData(1);
    if (!lengths) lengths = this.getInputData(2);

    const trainerFamily = this.properties.trainer_family;
    if (!x) {
      return { error: "Missing x input" };
    }
    if (trainerFamily !== "vae" && !y) {
      return { error: "Missing y input" };
    }

    const payload = {
      x,
      y,
      lengths,
      trainer_family: trainerFamily,
      model_architecture: this.properties.model_architecture,
      metric: this.properties.metric,
      mode: this.properties.mode,
      num_samples: parseInt(this.properties.num_samples),
      max_epochs: parseInt(this.properties.max_epochs),
      scheduler: this.properties.scheduler,
      search_alg: this.properties.search_alg,
      max_concurrent: parseInt(this.properties.max_concurrent),
      checkpoint_interval: parseInt(this.properties.checkpoint_interval),
      autosave: !!this.properties.autosave,
      val_size: Number(this.properties.val_size),
      test_size: Number(this.properties.test_size),
      norm_type: this.properties.norm_type,
      random_state: parseInt(this.properties.random_state),
      shuffle: !!this.properties.shuffle,
      batch_size: parseInt(this.properties.batch_size),
      loss_function: this.properties.loss_function,
      optimizer: this.properties.optimizer,
      learning_rate: Number(this.properties.learning_rate),
      dropout_rate: Number(this.properties.dropout_rate),
      batch_norm: !!this.properties.batch_norm,
      activation_function: this.properties.activation_function,
      output_activation: this.properties.output_activation,
      node_list: this.properties.node_list,
      width: parseInt(this.properties.width),
      depth: parseInt(this.properties.depth),
      rnn_type: this.properties.rnn_type,
      head_type: this.properties.head_type,
      num_heads: parseInt(this.properties.num_heads),
      ff_dim: parseInt(this.properties.ff_dim),
      max_seq_len: parseInt(this.properties.max_seq_len),
      use_causal_mask: !!this.properties.use_causal_mask,
      seq_length: parseInt(this.properties.seq_length),
      seq_to_one: !!this.properties.seq_to_one,
      suffix_crop: !!this.properties.suffix_crop,
      suffix_crop_min_length: parseInt(this.properties.suffix_crop_min_length),
      latent_dims: parseInt(this.properties.latent_dims),
      encoder_node_list: this.properties.encoder_node_list,
      decoder_node_list: this.properties.decoder_node_list,
      l1_reg: Number(this.properties.l1_reg),
      l2_reg: Number(this.properties.l2_reg),
      num_mixtures: parseInt(this.properties.num_mixtures),
      search_space: this._parseJsonProperty("search_space_json", {}),
      resources: this._parseJsonProperty("resources_json", { cpu: 1, gpu: 0 }),
    };

    try {
      return await TrainerNodeShared.postJsonWithErrors(endpoint, payload);
    } catch (err) {
      console.error("HPO backend error", err.detail || err);
      const detail = err.detail || {};
      const step = detail.step ? ` (${detail.step})` : "";
      this._setInlineStatus(
        `HPO error${step}: ${detail.error || err.message}`,
        true,
      );
      throw err;
    }
  }

  getExtraMenuOptions(_graphcanvas, options) {
    options.push({
      content: "Edit Search Space JSON...",
      callback: () => this._editManualSearchSpaceJson(),
    });
    return options;
  }

  get extractors() {
    return [
      (j) => (j && (j.trainer_hyperparameters || j.best_hyperparameters)) || {},
      (j) => j || {},
      (j) => (j && j.trial_history) || {},
      (j) => (j && j.artifact_path) || "",
    ];
  }

  get defaults() {
    return [{}, {}, {}, ""];
  }

  onExecute() {
    super.onExecute();
    this._pBestHyperparameters = this._p0;
    this._pTuningResult = this._p1;
    this._pTrialHistory = this._p2;
    this._pArtifactPath = this._p3;
  }

  onPropertyChanged(name, value, prevValue) {
    if (value !== prevValue && this._runner) {
      this._runner.invalidate();
    }

    if (
      [
        "hpo_choice_values",
        "hpo_range_low",
        "hpo_range_high",
        "hpo_range_q",
        "hpo_include_zero",
        "hpo_zero_probability",
        "hpo_search_type",
        "hpo_max_depth",
        "hpo_allowed_widths",
      ].includes(name)
    ) {
      if (name === "hpo_include_zero") {
        this._setEditorFieldVisibility(this.properties.hpo_search_type);
      }
      this._refreshEditorMeta();
    }

    if (name === "hpo_tune_architecture") {
      if (this.properties.hpo_tunable_key === HPO_ARCHITECTURE_KEY) {
        this._setEditorFieldVisibility("layer_structure");
      }
      this._saveArchitectureFromToggle().catch((err) => {
        this._setInlineStatus(err.message, true);
      });
    }

    if (name === "hpo_architecture_mode") {
      if (
        this.properties.hpo_tunable_key === HPO_ARCHITECTURE_KEY &&
        this.properties.hpo_tune_architecture
      ) {
        this._saveArchitectureFromToggle().catch((err) => {
          this._setInlineStatus(err.message, true);
        });
      }
    }

    if (
      ["hpo_max_depth", "hpo_allowed_widths"].includes(name) &&
      this.properties.hpo_tunable_key === HPO_ARCHITECTURE_KEY &&
      this.properties.hpo_tune_architecture
    ) {
      this._saveArchitectureFromToggle().catch((err) => {
        this._setInlineStatus(err.message, true);
      });
    }

    if (name === "hpo_tunable_key") {
      this._applyEditorFromSelectedParameter();
      return;
    }

    if (name === "trainer_family" || name === "model_architecture") {
      this._fetchTemplatesAndApply(true);
      return;
    }

    if (this.properties.hpo_tunable_key === HPO_ARCHITECTURE_KEY) {
      this._setEditorFieldVisibility("layer_structure");
      this._refreshPrimaryActionLabel();
      this.setDirtyCanvas(true, true);
      return;
    }
  }
}

MELTHyperparameterTunerNode.title = "MELT Hyperparameter Tuner";
MELTHyperparameterTunerNode.desc =
  "Runs PT-MELT Ray Tune hyperparameter searches and emits best parameters for trainer nodes.";

LiteGraph.registerNodeType(
  "MELT/HPO/MELTHyperparameterTunerNode",
  MELTHyperparameterTunerNode,
);
