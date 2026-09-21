class MELTVAETrainerNode extends AsyncMultiOutputNodeBase {
  constructor() {
    super("MELT VAE Trainer");

    // Inputs
    this.addInput("x", "array");
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
    this.addProperty("batch_norm", false, "boolean");
    this.addProperty("activation_function", "relu", "string");
    this.addProperty("output_activation", "linear", "string");
    this.addProperty("latent_dims", 8, "number");
    this.addProperty("encoder_node_list", [32, 16], "array");
    this.addProperty("decoder_node_list", [16, 32], "array");
    this.addProperty("l1_reg", 0.0, "number");
    this.addProperty("l2_reg", 0.0, "number");
    this.addProperty("lr_scheduler", "ReduceLROnPlateau", "string");
    this.addProperty("num_mixtures", 1, "number");

    this.addProperty("endpoint", "/melt_vae_trainer", "string");

    // Widgets
    this.addIntPropertyWidget("Latent Dims", "latent_dims", {
      min: 1,
      default: 8,
    });
    this.addIntArrayPropertyWidget("Encoder Nodes", "encoder_node_list", {
      default: [32, 16],
    });
    this.addIntArrayPropertyWidget("Decoder Nodes", "decoder_node_list", {
      default: [16, 32],
    });
    this.addIntPropertyWidget("N Mixtures", "num_mixtures", {
      min: 1,
      default: 1,
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

    this.size = [300, 620];
  }

  async fetch(resolvedInput) {
    const endpoint = this.properties && this.properties.endpoint;
    if (!endpoint) {
      return { model: {}, history: {}, error: "No endpoint configured" };
    }

    const { x, hyperparameters } =
      TrainerNodeShared.parseTrainerInputsWithOptions(
        resolvedInput,
        (i) => this.getInputData(i),
        { hasY: false, hasHyperparameters: true },
      );

    if (!x) {
      return {
        model: {},
        x_data: {},
        y_data: {},
        x_data_scaled: {},
        y_data_scaled: {},
        x_normalizer: {},
        y_normalizer: {},
        history: {},
        error: "Missing x input",
      };
    }

    const payload = TrainerNodeShared.buildCommonTrainingPayload(
      this.properties,
      x,
      x,
    );
    payload.latent_dims = parseInt(this.properties.latent_dims);
    payload.encoder_node_list = this.properties.encoder_node_list;
    payload.decoder_node_list = this.properties.decoder_node_list;
    if (hyperparameters) {
      payload.hyperparameters = hyperparameters;
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

MELTVAETrainerNode.title = "MELT VAE Trainer";
MELTVAETrainerNode.desc =
  "Trains a self-supervised VAE model using x input and VAE-specific hyperparameters.";

LiteGraph.registerNodeType(
  "MELT/Trainer/MELTVAETrainerNode",
  MELTVAETrainerNode,
);

class PlotVAEInitialClustersNode extends PlotNodeBase {
  constructor() {
    super("Plot VAE Initial Clusters", "/plot_vae_initial_clusters", 190);

    this.addInput("x", "array");
    this.addInput("labels", "array");

    this.addProperty("split", "train", "string");
    this.addProperty("feature_x", 0, "number");
    this.addProperty("feature_y", 1, "number");
    this.addProperty("alpha", 0.7, "number");
    this.addProperty("point_size", 18.0, "number");
    this.addProperty("size", [500, 500], "array");

    this.addDropdownPropertyWidget("Split", "split", {
      values: ["train", "val", "test"],
      default: "train",
    });
    this.addIntPropertyWidget("Feature X", "feature_x", {
      min: 0,
      default: 0,
    });
    this.addIntPropertyWidget("Feature Y", "feature_y", {
      min: 0,
      default: 1,
    });
    this.addFloatPropertyWidget("Alpha", "alpha", {
      min: 0.05,
      max: 1.0,
      default: 0.7,
      step: 0.05,
      precision: 2,
    });
    this.addFloatPropertyWidget("Point Size", "point_size", {
      min: 1,
      default: 18.0,
      step: 1,
      precision: 1,
    });
  }

  resolveInput() {
    const x = this.getInputData(0);
    const labels = this.getInputData(1);

    const xIsPromise = x && typeof x.then === "function";
    const labelsIsPromise = labels && typeof labels.then === "function";

    if (x === undefined || x === null) {
      return null;
    }

    if (xIsPromise || labelsIsPromise) {
      return Promise.all([x, labels]);
    }

    return [x, labels];
  }

  buildPayload(resolvedInput) {
    if (!Array.isArray(resolvedInput) || resolvedInput.length < 1) {
      this._error = "Invalid input for VAE initial cluster plot";
      this._imageReady = false;
      this.setDirtyCanvas(true, true);
      return { endpoint: null, payload: null };
    }

    const x = resolvedInput[0];
    const labels = resolvedInput.length > 1 ? resolvedInput[1] : null;
    if (x === undefined || x === null) {
      this._error = "Missing x input";
      this._imageReady = false;
      this.setDirtyCanvas(true, true);
      return { endpoint: null, payload: null };
    }

    return {
      endpoint: this.properties.endpoint || "/plot_vae_initial_clusters",
      payload: {
        x,
        labels: labels === undefined ? null : labels,
        split: this.properties.split || "train",
        feature_x: parseInt(this.properties.feature_x),
        feature_y: parseInt(this.properties.feature_y),
        alpha: Number(this.properties.alpha),
        point_size: Number(this.properties.point_size),
        figsize: [
          this.properties.fig_width || 8,
          this.properties.fig_height || 6,
        ],
        image_format: "pdf",
      },
    };
  }
}

PlotVAEInitialClustersNode.title = "Plot VAE Initial Clusters";
PlotVAEInitialClustersNode.desc =
  "Plot initial feature-space clustering for VAE data; labels are optional.";

LiteGraph.registerNodeType(
  "MELT/Plot/VAEInitialClusterPlot",
  PlotVAEInitialClustersNode,
);

class PlotVAELatentClustersNode extends PlotNodeBase {
  constructor() {
    super("Plot VAE Latent Clusters", "/plot_vae_latent_clusters", 210);

    this.addInput("model", "object");
    this.addInput("x", "array");
    this.addInput("labels", "array");

    this.addProperty("split", "train", "string");
    this.addProperty("latent_x", 0, "number");
    this.addProperty("latent_y", 1, "number");
    this.addProperty("alpha", 0.7, "number");
    this.addProperty("point_size", 18.0, "number");
    this.addProperty("size", [500, 500], "array");

    this.addDropdownPropertyWidget("Split", "split", {
      values: ["train", "val", "test"],
      default: "train",
    });
    this.addIntPropertyWidget("Latent X", "latent_x", {
      min: 0,
      default: 0,
    });
    this.addIntPropertyWidget("Latent Y", "latent_y", {
      min: 0,
      default: 1,
    });
    this.addFloatPropertyWidget("Alpha", "alpha", {
      min: 0.05,
      max: 1.0,
      default: 0.7,
      step: 0.05,
      precision: 2,
    });
    this.addFloatPropertyWidget("Point Size", "point_size", {
      min: 1,
      default: 18.0,
      step: 1,
      precision: 1,
    });
  }

  resolveInput() {
    const model = this.getInputData(0);
    const x = this.getInputData(1);
    const labels = this.getInputData(2);

    const modelIsPromise = model && typeof model.then === "function";
    const xIsPromise = x && typeof x.then === "function";
    const labelsIsPromise = labels && typeof labels.then === "function";

    if (
      model === undefined ||
      model === null ||
      x === undefined ||
      x === null
    ) {
      return null;
    }

    if (modelIsPromise || xIsPromise || labelsIsPromise) {
      return Promise.all([model, x, labels]);
    }

    return [model, x, labels];
  }

  buildPayload(resolvedInput) {
    if (!Array.isArray(resolvedInput) || resolvedInput.length < 2) {
      this._error = "Invalid input for VAE latent cluster plot";
      this._imageReady = false;
      this.setDirtyCanvas(true, true);
      return { endpoint: null, payload: null };
    }

    const model = resolvedInput[0];
    const x = resolvedInput[1];
    const labels = resolvedInput.length > 2 ? resolvedInput[2] : null;

    if (!model || x === undefined || x === null) {
      this._error = "Missing model or x input";
      this._imageReady = false;
      this.setDirtyCanvas(true, true);
      return { endpoint: null, payload: null };
    }

    return {
      endpoint: this.properties.endpoint || "/plot_vae_latent_clusters",
      payload: {
        model,
        x,
        labels: labels === undefined ? null : labels,
        split: this.properties.split || "train",
        latent_x: parseInt(this.properties.latent_x),
        latent_y: parseInt(this.properties.latent_y),
        alpha: Number(this.properties.alpha),
        point_size: Number(this.properties.point_size),
        figsize: [
          this.properties.fig_width || 8,
          this.properties.fig_height || 6,
        ],
        image_format: "pdf",
      },
    };
  }
}

PlotVAELatentClustersNode.title = "Plot VAE Latent Clusters";
PlotVAELatentClustersNode.desc =
  "Encode x with a trained VAE and plot latent-space clusters; labels are optional.";

LiteGraph.registerNodeType(
  "MELT/Plot/VAELatentClusterPlot",
  PlotVAELatentClustersNode,
);

class PlotVAEReconstructionNode extends PlotNodeBase {
  constructor() {
    super("Plot VAE Reconstruction", "/plot_vae_reconstruction", 210);

    this.addInput("model", "object");
    this.addInput("x", "array");
    this.addInput("labels", "array");

    this.addProperty("split", "train", "string");
    this.addProperty("feature_x", 0, "number");
    this.addProperty("feature_y", 1, "number");
    this.addProperty("alpha", 0.7, "number");
    this.addProperty("point_size", 18.0, "number");
    this.addProperty("size", [600, 500], "array");

    this.addDropdownPropertyWidget("Split", "split", {
      values: ["train", "val", "test"],
      default: "train",
    });
    this.addIntPropertyWidget("Feature X", "feature_x", {
      min: 0,
      default: 0,
    });
    this.addIntPropertyWidget("Feature Y", "feature_y", {
      min: 0,
      default: 1,
    });
    this.addFloatPropertyWidget("Alpha", "alpha", {
      min: 0.05,
      max: 1.0,
      default: 0.7,
      step: 0.05,
      precision: 2,
    });
    this.addFloatPropertyWidget("Point Size", "point_size", {
      min: 1,
      default: 18.0,
      step: 1,
      precision: 1,
    });
  }

  resolveInput() {
    const model = this.getInputData(0);
    const x = this.getInputData(1);
    const labels = this.getInputData(2);

    const modelIsPromise = model && typeof model.then === "function";
    const xIsPromise = x && typeof x.then === "function";
    const labelsIsPromise = labels && typeof labels.then === "function";

    if (
      model === undefined ||
      model === null ||
      x === undefined ||
      x === null
    ) {
      return null;
    }

    if (modelIsPromise || xIsPromise || labelsIsPromise) {
      return Promise.all([model, x, labels]);
    }

    return [model, x, labels];
  }

  buildPayload(resolvedInput) {
    if (!Array.isArray(resolvedInput) || resolvedInput.length < 2) {
      this._error = "Invalid input for VAE reconstruction plot";
      this._imageReady = false;
      this.setDirtyCanvas(true, true);
      return { endpoint: null, payload: null };
    }

    const model = resolvedInput[0];
    const x = resolvedInput[1];
    const labels = resolvedInput.length > 2 ? resolvedInput[2] : null;

    if (!model || x === undefined || x === null) {
      this._error = "Missing model or x input";
      this._imageReady = false;
      this.setDirtyCanvas(true, true);
      return { endpoint: null, payload: null };
    }

    return {
      endpoint: this.properties.endpoint || "/plot_vae_reconstruction",
      payload: {
        model,
        x,
        labels: labels === undefined ? null : labels,
        split: this.properties.split || "train",
        feature_x: parseInt(this.properties.feature_x),
        feature_y: parseInt(this.properties.feature_y),
        alpha: Number(this.properties.alpha),
        point_size: Number(this.properties.point_size),
        figsize: [
          this.properties.fig_width || 10,
          this.properties.fig_height || 4,
        ],
        image_format: "pdf",
      },
    };
  }
}

PlotVAEReconstructionNode.title = "Plot VAE Reconstruction";
PlotVAEReconstructionNode.desc =
  "Plot original and reconstructed feature-space points for a trained VAE.";

LiteGraph.registerNodeType(
  "MELT/Plot/VAEReconstructionPlot",
  PlotVAEReconstructionNode,
);

class VAEDecodePlotNode extends PlotNodeBase {
  constructor() {
    super("Plot VAE Decoded Samples", "/plot_vae_decoded", 270);

    this.addInput("model", "object");
    this.addInput("z", "array");
    this.addInput("labels", "array");

    this.addProperty("mode", "auto", "string");
    this.addProperty("n_samples", 200, "number");
    this.addProperty("latent_scale", 1.0, "number");
    this.addProperty("random_state", 42, "number");
    this.addProperty("feature_x", 0, "number");
    this.addProperty("feature_y", 1, "number");
    this.addProperty("feature_z", 2, "number");
    this.addProperty("use_pca", true, "boolean");
    this.addProperty("pairplot", false, "boolean");
    this.addProperty("plot_3d", false, "boolean");
    this.addProperty("max_plot_samples", 2000, "number");
    this.addProperty("alpha", 0.7, "number");
    this.addProperty("point_size", 18.0, "number");
    this.addProperty("size", [520, 520], "array");

    this.addDropdownPropertyWidget("Mode", "mode", {
      values: ["auto", "scatter2d", "pca2d", "pairplot", "scatter3d"],
      default: "auto",
    });
    this.addIntPropertyWidget("N Samples", "n_samples", {
      min: 1,
      default: 200,
    });
    this.addFloatPropertyWidget("Latent Scale", "latent_scale", {
      min: 0.01,
      default: 1.0,
      step: 0.05,
      precision: 2,
    });
    this.addIntPropertyWidget("Random State", "random_state", {
      min: 0,
      default: 42,
    });
    this.addIntPropertyWidget("Feature X", "feature_x", {
      min: 0,
      default: 0,
    });
    this.addIntPropertyWidget("Feature Y", "feature_y", {
      min: 0,
      default: 1,
    });
    this.addIntPropertyWidget("Feature Z", "feature_z", {
      min: 0,
      default: 2,
    });
    this.addBooleanPropertyWidget("Use PCA", "use_pca", {
      default: true,
    });
    this.addBooleanPropertyWidget("Pairplot", "pairplot", {
      default: false,
    });
    this.addBooleanPropertyWidget("3D Plot", "plot_3d", {
      default: false,
    });
    this.addIntPropertyWidget("Max Plot Samples", "max_plot_samples", {
      min: 10,
      default: 2000,
    });
    this.addFloatPropertyWidget("Alpha", "alpha", {
      min: 0.05,
      max: 1.0,
      default: 0.7,
      step: 0.05,
      precision: 2,
    });
    this.addFloatPropertyWidget("Point Size", "point_size", {
      min: 1,
      default: 18.0,
      step: 1,
      precision: 1,
    });
  }

  resolveInput() {
    const model = this.getInputData(0);
    const z = this.getInputData(1);
    const labels = this.getInputData(2);

    const modelIsPromise = model && typeof model.then === "function";
    const zIsPromise = z && typeof z.then === "function";
    const labelsIsPromise = labels && typeof labels.then === "function";

    if (model === undefined || model === null) {
      return null;
    }

    if (modelIsPromise || zIsPromise || labelsIsPromise) {
      return Promise.all([model, z, labels]);
    }

    return [model, z, labels];
  }

  buildPayload(resolvedInput) {
    if (!Array.isArray(resolvedInput) || resolvedInput.length < 1) {
      this._error = "Invalid input for VAE decoded sample plot";
      this._imageReady = false;
      this.setDirtyCanvas(true, true);
      return { endpoint: null, payload: null };
    }

    const model = resolvedInput[0];
    const z = resolvedInput.length > 1 ? resolvedInput[1] : null;
    const labels = resolvedInput.length > 2 ? resolvedInput[2] : null;
    if (!model) {
      this._error = "Missing model input";
      this._imageReady = false;
      this.setDirtyCanvas(true, true);
      return { endpoint: null, payload: null };
    }

    return {
      endpoint: this.properties.endpoint || "/plot_vae_decoded",
      payload: {
        model,
        z: z === undefined ? null : z,
        labels: labels === undefined ? null : labels,
        n_samples: parseInt(this.properties.n_samples),
        latent_scale: Number(this.properties.latent_scale),
        random_state: parseInt(this.properties.random_state),
        feature_x: parseInt(this.properties.feature_x),
        feature_y: parseInt(this.properties.feature_y),
        feature_z: parseInt(this.properties.feature_z),
        mode: this.properties.mode || "auto",
        use_pca: !!this.properties.use_pca,
        pairplot: !!this.properties.pairplot,
        plot_3d: !!this.properties.plot_3d,
        max_plot_samples: parseInt(this.properties.max_plot_samples),
        alpha: Number(this.properties.alpha),
        point_size: Number(this.properties.point_size),
        figsize: [
          this.properties.fig_width || 8,
          this.properties.fig_height || 6,
        ],
        image_format: "pdf",
      },
    };
  }
}

VAEDecodePlotNode.title = "Plot VAE Decoded Samples";
VAEDecodePlotNode.desc =
  "Decode latent vectors (or random latent samples) and visualize decoded synthetic data before export.";

LiteGraph.registerNodeType("MELT/Plot/VAEDecodePlot", VAEDecodePlotNode);

class VAELatentEncodeNode extends AsyncMultiOutputNodeBase {
  constructor() {
    super("VAE Latent Encode");

    this.addInput("model", "object");
    this.addInput("x", "array");
    this.addInput("labels", "array");

    this.addOutput("z", "array");
    this.addOutput("mix_coeffs", "array");
    this.addOutput("means", "array");
    this.addOutput("log_vars", "array");
    this.addOutput("labels", "array");
    this.addOutput("shape", "object");

    this.addProperty("split", "train", "string");
    this.addProperty("include_encoder_stats", true, "boolean");
    this.addProperty("endpoint", "/vae_encode_latent", "string");

    this.addDropdownPropertyWidget("Split", "split", {
      values: ["train", "val", "test"],
      default: "train",
    });
    this.addBooleanPropertyWidget("Include Stats", "include_encoder_stats", {
      default: true,
    });

    this.size = [250, 145];
  }

  async fetch(resolvedInput) {
    const endpoint = this.properties.endpoint || "/vae_encode_latent";

    let model = null;
    let x = null;
    let labels = null;
    if (Array.isArray(resolvedInput)) {
      [model, x, labels] = resolvedInput;
    }

    if (!model || !x) {
      model = this.getInputData(0);
      x = this.getInputData(1);
      labels = this.getInputData(2);
    }

    if (!model || !x) {
      return {
        z: [],
        mix_coeffs: [],
        means: [],
        log_vars: [],
        labels: [],
        shape: {},
        error: "Missing model or x input",
      };
    }

    const payload = {
      model,
      x,
      split: this.properties.split || "train",
      include_encoder_stats: !!this.properties.include_encoder_stats,
    };

    const response = await fetch(endpoint, {
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

    json.__labels_passthrough = labels || [];
    return json;
  }

  get extractors() {
    return [
      (j) => (j && j.z) || [],
      (j) => (j && j.mix_coeffs) || [],
      (j) => (j && j.means) || [],
      (j) => (j && j.log_vars) || [],
      (j) => (j && j.__labels_passthrough) || [],
      (j) => (j && j.shape) || {},
    ];
  }

  get defaults() {
    return [[], [], [], [], [], {}];
  }

  onExecute() {
    super.onExecute();
    this._pZ = this._p0;
    this._pMixCoeffs = this._p1;
    this._pMeans = this._p2;
    this._pLogVars = this._p3;
    this._pLabels = this._p4;
    this._pShape = this._p5;
  }
}

VAELatentEncodeNode.title = "VAE Latent Encode";
VAELatentEncodeNode.desc =
  "Encode x into latent vectors and optional encoder statistics using a trained VAE.";

LiteGraph.registerNodeType("MELT/VAE/LatentEncode", VAELatentEncodeNode);

class VAEReconstructDataNode extends AsyncMultiOutputNodeBase {
  constructor() {
    super("VAE Reconstruct Data");

    this.addInput("model", "object");
    this.addInput("x", "array");

    this.addOutput("x_reconstructed", "array");
    this.addOutput("residual", "array");
    this.addOutput("reconstruction_error", "array");
    this.addOutput("shape", "object");

    this.addProperty("split", "train", "string");
    this.addProperty("error_metric", "mse", "string");
    this.addProperty("endpoint", "/vae_reconstruct_data", "string");

    this.addDropdownPropertyWidget("Split", "split", {
      values: ["train", "val", "test"],
      default: "train",
    });
    this.addDropdownPropertyWidget("Error Metric", "error_metric", {
      values: ["mse", "mae"],
      default: "mse",
    });

    this.size = [260, 120];
  }

  async fetch(resolvedInput) {
    const endpoint = this.properties.endpoint || "/vae_reconstruct_data";

    let model = null;
    let x = null;
    if (Array.isArray(resolvedInput)) {
      [model, x] = resolvedInput;
    }
    if (!model || !x) {
      model = this.getInputData(0);
      x = this.getInputData(1);
    }

    if (!model || !x) {
      return {
        x_reconstructed: [],
        residual: [],
        reconstruction_error: [],
        shape: {},
        error: "Missing model or x input",
      };
    }

    const payload = {
      model,
      x,
      split: this.properties.split || "train",
      error_metric: this.properties.error_metric || "mse",
    };

    const response = await fetch(endpoint, {
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
      (j) => (j && j.x_reconstructed) || [],
      (j) => (j && j.residual) || [],
      (j) => (j && j.reconstruction_error) || [],
      (j) => (j && j.shape) || {},
    ];
  }

  get defaults() {
    return [[], [], [], {}];
  }

  onExecute() {
    super.onExecute();
    this._pXReconstructed = this._p0;
    this._pResidual = this._p1;
    this._pReconstructionError = this._p2;
    this._pShape = this._p3;
  }
}

VAEReconstructDataNode.title = "VAE Reconstruct Data";
VAEReconstructDataNode.desc =
  "Return reconstructed samples, residuals, and per-sample reconstruction error.";

LiteGraph.registerNodeType("MELT/VAE/ReconstructData", VAEReconstructDataNode);

class VAEAnomalyScoreNode extends AsyncMultiOutputNodeBase {
  constructor() {
    super("VAE Anomaly Score");

    this.addInput("model", "object");
    this.addInput("x", "array");

    this.addOutput("scores", "array");
    this.addOutput("anomaly_mask", "array");
    this.addOutput("threshold", "number");
    this.addOutput("ranked_indices", "array");
    this.addOutput("metadata", "object");

    this.addProperty("split", "train", "string");
    this.addProperty("error_metric", "mse", "string");
    this.addProperty("threshold_quantile", 0.99, "number");
    this.addProperty("endpoint", "/vae_anomaly_scores", "string");

    this.addDropdownPropertyWidget("Split", "split", {
      values: ["train", "val", "test"],
      default: "train",
    });
    this.addDropdownPropertyWidget("Error Metric", "error_metric", {
      values: ["mse", "mae"],
      default: "mse",
    });
    this.addFloatPropertyWidget("Threshold Quantile", "threshold_quantile", {
      min: 0.5,
      max: 0.999,
      default: 0.99,
      step: 0.001,
      precision: 3,
    });

    this.size = [280, 150];
  }

  async fetch(resolvedInput) {
    const endpoint = this.properties.endpoint || "/vae_anomaly_scores";

    let model = null;
    let x = null;
    if (Array.isArray(resolvedInput)) {
      [model, x] = resolvedInput;
    }
    if (!model || !x) {
      model = this.getInputData(0);
      x = this.getInputData(1);
    }

    if (!model || !x) {
      return {
        scores: [],
        anomaly_mask: [],
        threshold: 0,
        ranked_indices: [],
        metadata: {},
        error: "Missing model or x input",
      };
    }

    const payload = {
      model,
      x,
      split: this.properties.split || "train",
      error_metric: this.properties.error_metric || "mse",
      threshold_quantile: Number(this.properties.threshold_quantile),
    };

    const response = await fetch(endpoint, {
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
      (j) => (j && j.scores) || [],
      (j) => (j && j.anomaly_mask) || [],
      (j) => (j && j.threshold) || 0,
      (j) => (j && j.ranked_indices) || [],
      (j) => (j && j.metadata) || {},
    ];
  }

  get defaults() {
    return [[], [], 0, [], {}];
  }

  onExecute() {
    super.onExecute();
    this._pScores = this._p0;
    this._pAnomalyMask = this._p1;
    this._pThreshold = this._p2;
    this._pRankedIndices = this._p3;
    this._pMetadata = this._p4;
  }
}

VAEAnomalyScoreNode.title = "VAE Anomaly Score";
VAEAnomalyScoreNode.desc =
  "Compute reconstruction-error anomaly scores and thresholded anomaly masks.";

LiteGraph.registerNodeType("MELT/VAE/AnomalyScore", VAEAnomalyScoreNode);

class VAELatentClusterNode extends AsyncMultiOutputNodeBase {
  constructor() {
    super("VAE Latent Cluster");

    this.addInput("model", "object");
    this.addInput("x", "array");

    this.addOutput("z", "array");
    this.addOutput("cluster_labels", "array");
    this.addOutput("centroids", "array");
    this.addOutput("metadata", "object");
    this.addOutput("shape", "object");

    this.addProperty("split", "train", "string");
    this.addProperty("n_clusters", 5, "number");
    this.addProperty("random_state", 42, "number");
    this.addProperty("n_init", 10, "number");
    this.addProperty("endpoint", "/vae_latent_cluster", "string");

    this.addDropdownPropertyWidget("Split", "split", {
      values: ["train", "val", "test"],
      default: "train",
    });
    this.addIntPropertyWidget("N Clusters", "n_clusters", {
      min: 2,
      default: 5,
    });
    this.addIntPropertyWidget("Random State", "random_state", {
      min: 0,
      default: 42,
    });
    this.addIntPropertyWidget("N Init", "n_init", {
      min: 1,
      default: 10,
    });

    this.size = [270, 160];
  }

  async fetch(resolvedInput) {
    const endpoint = this.properties.endpoint || "/vae_latent_cluster";

    let model = null;
    let x = null;
    if (Array.isArray(resolvedInput)) {
      [model, x] = resolvedInput;
    }
    if (!model || !x) {
      model = this.getInputData(0);
      x = this.getInputData(1);
    }

    if (!model || !x) {
      return {
        z: [],
        cluster_labels: [],
        centroids: [],
        metadata: {},
        shape: {},
        error: "Missing model or x input",
      };
    }

    const payload = {
      model,
      x,
      split: this.properties.split || "train",
      n_clusters: parseInt(this.properties.n_clusters),
      random_state: parseInt(this.properties.random_state),
      n_init: parseInt(this.properties.n_init),
    };

    const response = await fetch(endpoint, {
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
      (j) => (j && j.z) || [],
      (j) => (j && j.cluster_labels) || [],
      (j) => (j && j.centroids) || [],
      (j) => (j && j.metadata) || {},
      (j) => (j && j.shape) || {},
    ];
  }

  get defaults() {
    return [[], [], [], {}, {}];
  }

  onExecute() {
    super.onExecute();
    this._pZ = this._p0;
    this._pClusterLabels = this._p1;
    this._pCentroids = this._p2;
    this._pMetadata = this._p3;
    this._pShape = this._p4;
  }
}

VAELatentClusterNode.title = "VAE Latent Cluster";
VAELatentClusterNode.desc =
  "Encode latent vectors and cluster them with KMeans for downstream workflows.";

LiteGraph.registerNodeType("MELT/VAE/LatentCluster", VAELatentClusterNode);

class VAEDecodeLatentNode extends AsyncMultiOutputNodeBase {
  constructor() {
    super("VAE Decode Latent");

    this.addInput("model", "object");
    this.addInput("z", "array");

    this.addOutput("x_decoded", "array");
    this.addOutput("z", "array");
    this.addOutput("shape", "object");

    this.addProperty("n_samples", 100, "number");
    this.addProperty("latent_scale", 1.0, "number");
    this.addProperty("random_state", 42, "number");
    this.addProperty("endpoint", "/vae_decode_latent", "string");

    this.addIntPropertyWidget("N Samples", "n_samples", {
      min: 1,
      default: 100,
    });
    this.addFloatPropertyWidget("Latent Scale", "latent_scale", {
      min: 0.01,
      default: 1.0,
      step: 0.05,
      precision: 2,
    });
    this.addIntPropertyWidget("Random State", "random_state", {
      min: 0,
      default: 42,
    });

    this.size = [260, 135];
  }

  async fetch(resolvedInput) {
    const endpoint = this.properties.endpoint || "/vae_decode_latent";

    let model = null;
    let z = null;
    if (Array.isArray(resolvedInput)) {
      [model, z] = resolvedInput;
    }
    if (!model) {
      model = this.getInputData(0);
      z = this.getInputData(1);
    }

    if (!model) {
      return {
        x_decoded: [],
        z: [],
        shape: {},
        error: "Missing model input",
      };
    }

    const payload = {
      model,
      z: z || null,
      n_samples: parseInt(this.properties.n_samples),
      latent_scale: Number(this.properties.latent_scale),
      random_state: parseInt(this.properties.random_state),
    };

    const response = await fetch(endpoint, {
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
      (j) => (j && j.x_decoded) || [],
      (j) => (j && j.z) || [],
      (j) => (j && j.shape) || {},
    ];
  }

  get defaults() {
    return [[], [], {}];
  }

  onExecute() {
    super.onExecute();
    this._pXDecoded = this._p0;
    this._pZ = this._p1;
    this._pShape = this._p2;
  }
}

VAEDecodeLatentNode.title = "VAE Decode Latent";
VAEDecodeLatentNode.desc =
  "Decode provided latent vectors (or random latent samples) into synthetic feature vectors.";

LiteGraph.registerNodeType("MELT/VAE/DecodeLatent", VAEDecodeLatentNode);

class VAELatentInterpolateNode extends AsyncMultiOutputNodeBase {
  constructor() {
    super("VAE Latent Interpolate");

    this.addInput("model", "object");
    this.addInput("z_start", "array");
    this.addInput("z_end", "array");

    this.addOutput("z_path", "array");
    this.addOutput("x_path", "array");
    this.addOutput("shape", "object");

    this.addProperty("num_steps", 16, "number");
    this.addProperty("endpoint", "/vae_latent_interpolate", "string");

    this.addIntPropertyWidget("Num Steps", "num_steps", {
      min: 2,
      default: 16,
    });

    this.size = [255, 105];
  }

  async fetch(resolvedInput) {
    const endpoint = this.properties.endpoint || "/vae_latent_interpolate";

    let model = null;
    let zStart = null;
    let zEnd = null;
    if (Array.isArray(resolvedInput)) {
      [model, zStart, zEnd] = resolvedInput;
    }
    if (!model || !zStart || !zEnd) {
      model = this.getInputData(0);
      zStart = this.getInputData(1);
      zEnd = this.getInputData(2);
    }

    if (!model || !zStart || !zEnd) {
      return {
        z_path: [],
        x_path: [],
        shape: {},
        error: "Missing model, z_start, or z_end input",
      };
    }

    const payload = {
      model,
      z_start: zStart,
      z_end: zEnd,
      num_steps: parseInt(this.properties.num_steps),
    };

    const response = await fetch(endpoint, {
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
      (j) => (j && j.z_path) || [],
      (j) => (j && j.x_path) || [],
      (j) => (j && j.shape) || {},
    ];
  }

  get defaults() {
    return [[], [], {}];
  }

  onExecute() {
    super.onExecute();
    this._pZPath = this._p0;
    this._pXPath = this._p1;
    this._pShape = this._p2;
  }
}

VAELatentInterpolateNode.title = "VAE Latent Interpolate";
VAELatentInterpolateNode.desc =
  "Interpolate between two latent vectors and decode each step to feature space.";

LiteGraph.registerNodeType(
  "MELT/VAE/LatentInterpolate",
  VAELatentInterpolateNode,
);
