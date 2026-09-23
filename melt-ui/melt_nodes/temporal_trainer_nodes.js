class MELTTemporalSupervisedTrainerNode extends AsyncMultiOutputNodeBase {
  constructor() {
    super("MELT Temporal Supervised Trainer");

    // Inputs
    this.addInput("x", "array");
    this.addInput("y", "array");
    this.addInput("lengths", "array");
    this.addInput("hyperparameters", "object");

    // Outputs
    this.addOutput("model", "object");
    this.addOutput("x_data", "object");
    this.addOutput("y_data", "object");
    this.addOutput("x_data_scaled", "object");
    this.addOutput("y_data_scaled", "object");
    this.addOutput("x_normalizer", "object");
    this.addOutput("y_normalizer", "object");
    this.addOutput("history", "object");

    // Properties
    this.addProperty("val_size", 0.1, "number");
    this.addProperty("test_size", 0.1, "number");
    this.addProperty("norm_type", "none", "string");
    this.addProperty("random_state", 42, "number");
    this.addProperty("shuffle", true, "boolean");
    this.addProperty("batch_size", 32, "number");
    this.addProperty("num_epochs", 100, "number");
    this.addProperty("loss_function", "mse", "string");
    this.addProperty("optimizer", "Adam", "string");
    this.addProperty("learning_rate", 1e-3, "number");
    this.addProperty("dropout_rate", 0.0, "number");
    this.addProperty("activation_function", "relu", "string");
    this.addProperty("output_activation", "linear", "string");
    this.addProperty("width", 32, "number");
    this.addProperty("depth", 2, "number");
    this.addProperty("rnn_type", "lstm", "string");
    this.addProperty("head_type", "last", "string");
    this.addProperty("l1_reg", 0.0, "number");
    this.addProperty("l2_reg", 0.0, "number");
    this.addProperty("lr_scheduler", "ReduceLROnPlateau", "string");
    this.addProperty("num_mixtures", 0, "number");
    this.addProperty("seq_length", 60, "number");
    this.addProperty("seq_to_one", true, "boolean");
    this.addProperty("suffix_crop", false, "boolean");
    this.addProperty("suffix_crop_min_length", 32, "number");
    this.addProperty("endpoint", "/melt_temporal_supervised_trainer", "string");

    // Widgets
    this.addIntPropertyWidget("Width", "width", { min: 1, default: 32 });
    this.addIntPropertyWidget("Depth", "depth", { min: 1, default: 2 });
    this.addDropdownPropertyWidget("RNN Type", "rnn_type", {
      values: ["lstm", "gru", "rnn"],
      default: "lstm",
    });
    this.addDropdownPropertyWidget("RNN Head", "head_type", {
      values: ["last", "attn", "mean", "max"],
      default: "last",
    });
    this.addIntPropertyWidget("N Mixtures", "num_mixtures", {
      min: 0,
      default: 0,
    });
    this.addIntPropertyWidget("Sequence Length", "seq_length", {
      min: 1,
      default: 60,
    });
    this.addBooleanPropertyWidget("Seq-to-One", "seq_to_one", {
      default: true,
    });
    this.addBooleanPropertyWidget("Suffix Crop", "suffix_crop", {
      default: false,
    });
    this.addIntPropertyWidget(
      "Suffix Crop Min Length",
      "suffix_crop_min_length",
      {
        min: 1,
        default: 32,
      },
    );
    this.addFloatPropertyWidget("Validation Size", "val_size", {
      min: 0,
      max: 0.95,
      default: 0.1,
      step: 0.1,
      precision: 2,
    });
    this.addFloatPropertyWidget("Test Size", "test_size", {
      min: 0,
      max: 0.95,
      default: 0.1,
      step: 0.1,
      precision: 2,
    });
    this.addWidget(
      "combo",
      "Normalization Type",
      this.properties.norm_type,
      (v) => (this.properties.norm_type = v),
      {
        values: ["none", "standard", "minmax", "robust", "power", "quantile"],
      },
    );
    this.addIntPropertyWidget("Batch Size", "batch_size", {
      min: 1,
      default: 32,
    });
    this.addIntPropertyWidget("Num Epochs", "num_epochs", {
      min: 1,
      default: 100,
    });
    this.addDropdownPropertyWidget("Loss Function", "loss_function", {
      values: ["mse", "mae"],
      default: "mse",
    });
    this.addDropdownPropertyWidget("Optimizer", "optimizer", {
      values: ["Adam", "SGD", "RMSprop"],
      default: "Adam",
    });
    this.addDropdownPropertyWidget("LR Scheduler", "lr_scheduler", {
      values: ["ReduceLROnPlateau", "StepLR", "ExponentialLR", "None"],
      default: "ReduceLROnPlateau",
    });
    this.addFloatPropertyWidget("Learning Rate", "learning_rate", {
      min: 1e-6,
      default: 1e-3,
      step: 1e-5,
      precision: 6,
    });
    this.addDropdownPropertyWidget(
      "Activation Function",
      "activation_function",
      {
        values: [
          "relu",
          "tanh",
          "sigmoid",
          "selu",
          "elu",
          "leaky_relu",
          "swish",
          "linear",
        ],
        default: "relu",
      },
    );
    this.addDropdownPropertyWidget("Output Activation", "output_activation", {
      values: ["linear", "sigmoid", "softmax", "tanh"],
      default: "linear",
    });
    this.addFloatPropertyWidget("L1 Regularization", "l1_reg", {
      min: 0,
      default: 0.0,
      step: 1e-5,
      precision: 6,
    });
    this.addFloatPropertyWidget("L2 Regularization", "l2_reg", {
      min: 0,
      default: 0.0,
      step: 1e-5,
      precision: 6,
    });
    this.addFloatPropertyWidget("Dropout Rate", "dropout_rate", {
      min: 0,
      max: 0.9,
      default: 0.0,
      step: 0.1,
      precision: 2,
    });
    this.addBooleanPropertyWidget("Shuffle", "shuffle", {
      default: true,
    });
    this.addIntPropertyWidget("Random State", "random_state", {
      min: 0,
      default: 42,
    });

    this.size = [320, 640];
  }

  async fetch(resolvedInput) {
    const endpoint = this.properties && this.properties.endpoint;
    if (!endpoint) {
      return { model: {}, history: {}, error: "No endpoint configured" };
    }

    const { x, y, lengths, hyperparameters } =
      TrainerNodeShared.parseTrainerInputsWithOptions(
        resolvedInput,
        (i) => this.getInputData(i),
        { hasY: true, hasLengths: true, hasHyperparameters: true },
      );

    if (this.properties.seq_to_one === false) {
      return TrainerNodeShared.emptyTrainingResult(
        "Sequence-to-sequence temporal training is not supported yet.",
      );
    }

    const inputError = TrainerNodeShared.validateCommonTrainingInputs(
      this.properties,
      x,
      y,
    );
    if (inputError) {
      return TrainerNodeShared.emptyTrainingResult(inputError);
    }

    const payload = TrainerNodeShared.buildCommonTrainingPayload(
      this.properties,
      x,
      y,
    );
    payload.width = parseInt(this.properties.width);
    payload.depth = parseInt(this.properties.depth);
    payload.rnn_type = this.properties.rnn_type;
    payload.head_type = this.properties.head_type;
    payload.seq_length = parseInt(this.properties.seq_length);
    payload.seq_to_one = this.properties.seq_to_one;
    payload.suffix_crop = !!this.properties.suffix_crop;
    payload.suffix_crop_min_length = parseInt(
      this.properties.suffix_crop_min_length,
    );
    if (hyperparameters) {
      payload.hyperparameters = hyperparameters;
    }

    if (lengths) {
      payload.lengths = lengths;
    }

    return TrainerNodeShared.postJsonWithErrors(endpoint, payload);
  }

  get extractors() {
    return TrainerNodeShared.getTrainingExtractors();
  }

  get defaults() {
    return TrainerNodeShared.defaults;
  }

  onExecute() {
    super.onExecute();
    TrainerNodeShared.bindTrainingPromises(this);
  }

  onPropertyChanged(name, value, prevValue) {
    if (value !== prevValue && this._runner) {
      this._runner.invalidate();
    }
  }
}

MELTTemporalSupervisedTrainerNode.title = "MELT Temporal Supervised Trainer";
MELTTemporalSupervisedTrainerNode.desc =
  "Trains temporal supervised PT-MELT RNN models. For notebook-parity behavior, provide raw 2D x/y and let this node split-scale-window internally; pre-windowed inputs are supported with boundary-overlap trimming.";

LiteGraph.registerNodeType(
  "MELT/Trainer/MELTTemporalSupervisedTrainerNode",
  MELTTemporalSupervisedTrainerNode,
);

class MELTTemporalTransformerTrainerNode extends AsyncMultiOutputNodeBase {
  constructor() {
    super("MELT Temporal Transformer Trainer");

    // Inputs
    this.addInput("x", "array");
    this.addInput("y", "array");
    this.addInput("lengths", "array");
    this.addInput("hyperparameters", "object");

    // Outputs
    this.addOutput("model", "object");
    this.addOutput("x_data", "object");
    this.addOutput("y_data", "object");
    this.addOutput("x_data_scaled", "object");
    this.addOutput("y_data_scaled", "object");
    this.addOutput("x_normalizer", "object");
    this.addOutput("y_normalizer", "object");
    this.addOutput("history", "object");

    // Properties
    this.addProperty("val_size", 0.1, "number");
    this.addProperty("test_size", 0.1, "number");
    this.addProperty("norm_type", "none", "string");
    this.addProperty("random_state", 42, "number");
    this.addProperty("shuffle", true, "boolean");
    this.addProperty("batch_size", 32, "number");
    this.addProperty("num_epochs", 100, "number");
    this.addProperty("loss_function", "mse", "string");
    this.addProperty("optimizer", "Adam", "string");
    this.addProperty("learning_rate", 1e-3, "number");
    this.addProperty("dropout_rate", 0.0, "number");
    this.addProperty("activation_function", "relu", "string");
    this.addProperty("output_activation", "linear", "string");
    this.addProperty("width", 64, "number");
    this.addProperty("depth", 2, "number");
    this.addProperty("head_type", "last", "string");
    this.addProperty("num_heads", 4, "number");
    this.addProperty("ff_dim", 0, "number");
    this.addProperty("max_seq_len", 2048, "number");
    this.addProperty("use_causal_mask", false, "boolean");
    this.addProperty("l1_reg", 0.0, "number");
    this.addProperty("l2_reg", 0.0, "number");
    this.addProperty("lr_scheduler", "ReduceLROnPlateau", "string");
    this.addProperty("num_mixtures", 0, "number");
    this.addProperty("seq_length", 60, "number");
    this.addProperty("seq_to_one", true, "boolean");
    this.addProperty(
      "endpoint",
      "/melt_temporal_transformer_trainer",
      "string",
    );

    // Widgets
    this.addIntPropertyWidget("Width", "width", { min: 1, default: 64 });
    this.addIntPropertyWidget("Depth", "depth", { min: 1, default: 2 });
    this.addDropdownPropertyWidget("Head Type", "head_type", {
      values: ["last", "attn", "mean", "max"],
      default: "last",
    });
    this.addIntPropertyWidget("Num Heads", "num_heads", {
      min: 1,
      default: 4,
    });
    this.addIntPropertyWidget("FF Dim (0 = auto)", "ff_dim", {
      min: 0,
      default: 0,
    });
    this.addIntPropertyWidget("Max Seq Length", "max_seq_len", {
      min: 1,
      default: 2048,
    });
    this.addBooleanPropertyWidget("Use Causal Mask", "use_causal_mask", {
      default: false,
    });
    this.addIntPropertyWidget("N Mixtures", "num_mixtures", {
      min: 0,
      default: 0,
    });
    this.addIntPropertyWidget("Sequence Length", "seq_length", {
      min: 1,
      default: 60,
    });
    this.addBooleanPropertyWidget("Seq-to-One", "seq_to_one", {
      default: true,
    });
    this.addFloatPropertyWidget("Validation Size", "val_size", {
      min: 0,
      max: 0.95,
      default: 0.1,
      step: 0.1,
      precision: 2,
    });
    this.addFloatPropertyWidget("Test Size", "test_size", {
      min: 0,
      max: 0.95,
      default: 0.1,
      step: 0.1,
      precision: 2,
    });
    this.addWidget(
      "combo",
      "Normalization Type",
      this.properties.norm_type,
      (v) => (this.properties.norm_type = v),
      {
        values: ["none", "standard", "minmax", "robust", "power", "quantile"],
      },
    );
    this.addIntPropertyWidget("Batch Size", "batch_size", {
      min: 1,
      default: 32,
    });
    this.addIntPropertyWidget("Num Epochs", "num_epochs", {
      min: 1,
      default: 100,
    });
    this.addDropdownPropertyWidget("Loss Function", "loss_function", {
      values: ["mse", "mae"],
      default: "mse",
    });
    this.addDropdownPropertyWidget("Optimizer", "optimizer", {
      values: ["Adam", "SGD", "RMSprop"],
      default: "Adam",
    });
    this.addDropdownPropertyWidget("LR Scheduler", "lr_scheduler", {
      values: ["ReduceLROnPlateau", "StepLR", "ExponentialLR", "None"],
      default: "ReduceLROnPlateau",
    });
    this.addFloatPropertyWidget("Learning Rate", "learning_rate", {
      min: 1e-6,
      default: 1e-3,
      step: 1e-5,
      precision: 6,
    });
    this.addDropdownPropertyWidget(
      "Activation Function",
      "activation_function",
      {
        values: [
          "relu",
          "tanh",
          "sigmoid",
          "selu",
          "elu",
          "leaky_relu",
          "swish",
          "linear",
        ],
        default: "relu",
      },
    );
    this.addDropdownPropertyWidget("Output Activation", "output_activation", {
      values: ["linear", "sigmoid", "softmax", "tanh"],
      default: "linear",
    });
    this.addFloatPropertyWidget("L1 Regularization", "l1_reg", {
      min: 0,
      default: 0.0,
      step: 1e-5,
      precision: 6,
    });
    this.addFloatPropertyWidget("L2 Regularization", "l2_reg", {
      min: 0,
      default: 0.0,
      step: 1e-5,
      precision: 6,
    });
    this.addFloatPropertyWidget("Dropout Rate", "dropout_rate", {
      min: 0,
      max: 0.9,
      default: 0.0,
      step: 0.1,
      precision: 2,
    });
    this.addBooleanPropertyWidget("Shuffle", "shuffle", {
      default: true,
    });
    this.addIntPropertyWidget("Random State", "random_state", {
      min: 0,
      default: 42,
    });

    this.size = [340, 680];
  }

  async fetch(resolvedInput) {
    const endpoint = this.properties && this.properties.endpoint;
    if (!endpoint) {
      return { model: {}, history: {}, error: "No endpoint configured" };
    }

    const { x, y, lengths, hyperparameters } =
      TrainerNodeShared.parseTrainerInputsWithOptions(
        resolvedInput,
        (i) => this.getInputData(i),
        { hasY: true, hasLengths: true, hasHyperparameters: true },
      );

    if (this.properties.seq_to_one === false) {
      return TrainerNodeShared.emptyTrainingResult(
        "Sequence-to-sequence temporal training is not supported yet.",
      );
    }

    const inputError = TrainerNodeShared.validateCommonTrainingInputs(
      this.properties,
      x,
      y,
    );
    if (inputError) {
      return TrainerNodeShared.emptyTrainingResult(inputError);
    }

    const payload = TrainerNodeShared.buildCommonTrainingPayload(
      this.properties,
      x,
      y,
    );
    payload.width = parseInt(this.properties.width);
    payload.depth = parseInt(this.properties.depth);
    payload.head_type = this.properties.head_type;
    payload.num_heads = parseInt(this.properties.num_heads);
    payload.ff_dim = parseInt(this.properties.ff_dim);
    payload.max_seq_len = parseInt(this.properties.max_seq_len);
    payload.use_causal_mask = !!this.properties.use_causal_mask;
    payload.seq_length = parseInt(this.properties.seq_length);
    payload.seq_to_one = this.properties.seq_to_one;
    if (hyperparameters) {
      payload.hyperparameters = hyperparameters;
    }

    if (lengths) {
      payload.lengths = lengths;
    }

    return TrainerNodeShared.postJsonWithErrors(endpoint, payload);
  }

  get extractors() {
    return TrainerNodeShared.getTrainingExtractors();
  }

  get defaults() {
    return TrainerNodeShared.defaults;
  }

  onExecute() {
    super.onExecute();
    TrainerNodeShared.bindTrainingPromises(this);
  }

  onPropertyChanged(name, value, prevValue) {
    if (value !== prevValue && this._runner) {
      this._runner.invalidate();
    }
  }
}

MELTTemporalTransformerTrainerNode.title = "MELT Temporal Transformer Trainer";
MELTTemporalTransformerTrainerNode.desc =
  "Trains temporal supervised PT-MELT transformer models on the same temporal pipeline used by the RNN trainer for side-by-side comparison.";

LiteGraph.registerNodeType(
  "MELT/Trainer/MELTTemporalTransformerTrainerNode",
  MELTTemporalTransformerTrainerNode,
);
