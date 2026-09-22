class MELTSupervisedTrainerNode extends AsyncMultiOutputNodeBase {
  constructor() {
    super("MELT Supervised Trainer");

    // Inputs
    this.addInput("x", "array");
    this.addInput("y", "array");
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
    this.addProperty("model_architecture", "ann", "string");
    this.addProperty("dropout_rate", 0.0, "number");
    this.addProperty("batch_norm", false, "boolean");
    this.addProperty("activation_function", "relu", "string");
    this.addProperty("output_activation", "linear", "string");
    this.addProperty("node_list", [8, 8], "array");
    this.addProperty("l1_reg", 0.0, "number");
    this.addProperty("l2_reg", 0.0, "number");
    this.addProperty("lr_scheduler", "ReduceLROnPlateau", "string");
    this.addProperty("num_mixtures", 0, "number");

    // Endpoint
    this.addProperty("endpoint", "/melt_supervised_trainer", "string");

    // Widgets
    this._normTypeWidget = null;
    this.addDropdownPropertyWidget("Model Architecture", "model_architecture", {
      values: ["ann", "resnet", "bnn"],
      default: "ann",
    });
    this.addIntArrayPropertyWidget("Node List", "node_list", {
      default: [8, 8],
    });
    this.addIntPropertyWidget("N Mixtures", "num_mixtures", {
      min: 0,
      default: 0,
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
    // Dropdown for normalization type
    this._normTypeWidget = this.addWidget(
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
    this.addBooleanPropertyWidget("Batch Norm", "batch_norm", {
      default: false,
    });
    this.addBooleanPropertyWidget("Shuffle", "shuffle", {
      default: true,
    });
    this.addIntPropertyWidget("Random State", "random_state", {
      min: 0,
      default: 42,
    });

    this.size = [300, 560];
  }

  _syncWidgetsFromProperties() {
    if (this._normTypeWidget) {
      this._normTypeWidget.value = this.properties.norm_type || "none";
    }
  }

  onConfigure(info) {
    if (
      info &&
      info.properties &&
      typeof info.properties.norm_type === "string"
    ) {
      this.properties.norm_type = info.properties.norm_type;
    }

    this._syncWidgetsFromProperties();
  }

  // --- AsyncMultiOutputNodeBase contract ---

  // This is called by AsyncMultiOutputNodeBase.onExecute with resolved inputs.
  async fetch(resolvedInput) {
    const endpoint = this.properties && this.properties.endpoint;
    if (!endpoint) {
      return { model: {}, history: {}, error: "No endpoint configured" };
    }

    const { x, y, hyperparameters } =
      TrainerNodeShared.parseTrainerInputsWithOptions(
        resolvedInput,
        (i) => this.getInputData(i),
        { hasY: true, hasHyperparameters: true },
      );

    const inputError = TrainerNodeShared.validateCommonTrainingInputs(
      this.properties,
      x,
      y,
      { allowZeroValidation: true },
    );
    if (inputError) {
      return TrainerNodeShared.emptyTrainingResult(inputError);
    }

    const payload = TrainerNodeShared.buildCommonTrainingPayload(
      this.properties,
      x,
      y,
    );
    payload.model_architecture = this.properties.model_architecture;
    payload.node_list = this.properties.node_list;
    if (hyperparameters) {
      payload.hyperparameters = hyperparameters;
    }

    return TrainerNodeShared.postJsonWithErrors(endpoint, payload);
  }

  // Map full JSON -> individual outputs
  get extractors() {
    return TrainerNodeShared.getTrainingExtractors();
  }

  // Defaults for outputs if fetch fails / returns nothing
  get defaults() {
    return TrainerNodeShared.defaults;
  }

  // Optional: if you want promise aliases like RegressionDataNode has
  onExecute() {
    // Let the base class handle NodeRunner.runMulti + outputs
    super.onExecute();

    TrainerNodeShared.bindTrainingPromises(this);
  }

  onPropertyChanged(name, value, prevValue) {
    if (value !== prevValue && this._runner) {
      this._runner.invalidate();
    }
  }
}

MELTSupervisedTrainerNode.title = "MELT Supervised Trainer";
MELTSupervisedTrainerNode.desc =
  "Trains static supervised PT-MELT models (ANN, ResNet, BNN) using x and y inputs.";

LiteGraph.registerNodeType(
  "MELT/Trainer/MELTSupervisedTrainerNode",
  MELTSupervisedTrainerNode,
);
