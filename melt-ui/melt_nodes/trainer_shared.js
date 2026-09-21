(function () {
  const inFlightTrainingControllers = new Set();
  const activeTrainingRunIds = new Set();

  function makeTrainingRunId() {
    if (
      typeof crypto !== "undefined" &&
      typeof crypto.randomUUID === "function"
    ) {
      return crypto.randomUUID();
    }
    return `run_${Date.now()}_${Math.random().toString(16).slice(2)}`;
  }

  function abortInFlightTrainingRequests(reason = "Stopped by user") {
    for (const controller of inFlightTrainingControllers) {
      try {
        controller.abort(reason);
      } catch (err) {
        // Ignore abort errors from already-settled requests.
      }
    }
    inFlightTrainingControllers.clear();

    for (const runId of activeTrainingRunIds) {
      window.MeltApi.fetch("/melt_cancel_training", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ run_id: runId }),
        keepalive: true,
      }).catch(() => {
        // ignore network errors while stopping
      });
    }
    activeTrainingRunIds.clear();
  }

  function parseTrainerInputs(resolvedInput, fallbackGetter, inputCount) {
    let x = null;
    let y = null;
    let lengths = null;

    // For single-input nodes, the resolved input can itself be an array-valued tensor
    // (e.g., x = [[...], [...]]). In that case, treat the whole value as x.
    if (inputCount === 1) {
      if (
        resolvedInput &&
        typeof resolvedInput === "object" &&
        !Array.isArray(resolvedInput) &&
        "x" in resolvedInput
      ) {
        x = resolvedInput.x;
      } else {
        x = resolvedInput;
      }
    } else if (Array.isArray(resolvedInput)) {
      [x, y, lengths] = resolvedInput;
    } else if (resolvedInput && typeof resolvedInput === "object") {
      x = resolvedInput.x;
      y = resolvedInput.y;
      lengths = resolvedInput.lengths;
    }

    if (!x && typeof fallbackGetter === "function") {
      x = fallbackGetter(0);
      if (inputCount > 1) y = fallbackGetter(1);
      if (inputCount > 2) lengths = fallbackGetter(2);
    }

    return { x, y, lengths };
  }

  function parseTrainerInputsWithOptions(
    resolvedInput,
    fallbackGetter,
    options = {},
  ) {
    const hasY = options.hasY !== false;
    const hasLengths = !!options.hasLengths;
    const hasHyperparameters = !!options.hasHyperparameters;
    let x = null;
    let y = null;
    let lengths = null;
    let hyperparameters = null;

    if (Array.isArray(resolvedInput)) {
      let index = 0;
      x = resolvedInput[index++];
      if (hasY) y = resolvedInput[index++];
      if (hasLengths) lengths = resolvedInput[index++];
      if (hasHyperparameters) hyperparameters = resolvedInput[index++];
    } else if (resolvedInput && typeof resolvedInput === "object") {
      if ("x" in resolvedInput) x = resolvedInput.x;
      else if (!hasY && !hasLengths && !hasHyperparameters) x = resolvedInput;
      y = resolvedInput.y;
      lengths = resolvedInput.lengths;
      hyperparameters =
        resolvedInput.hyperparameters ||
        resolvedInput.hyperparams ||
        resolvedInput.best_hyperparameters ||
        null;
    } else {
      x = resolvedInput;
    }

    if (
      (!x ||
        (hasY && !y) ||
        (hasLengths && !lengths) ||
        (hasHyperparameters && !hyperparameters)) &&
      typeof fallbackGetter === "function"
    ) {
      let index = 0;
      if (!x) x = fallbackGetter(index);
      index += 1;
      if (hasY) {
        if (!y) y = fallbackGetter(index);
        index += 1;
      }
      if (hasLengths) {
        if (!lengths) lengths = fallbackGetter(index);
        index += 1;
      }
      if (hasHyperparameters && !hyperparameters) {
        hyperparameters = fallbackGetter(index);
      }
    }

    return { x, y, lengths, hyperparameters };
  }

  function buildCommonTrainingPayload(properties, x, y) {
    return {
      x,
      y,
      val_size: Number(properties.val_size),
      test_size: Number(properties.test_size),
      norm_type: properties.norm_type,
      random_state: parseInt(properties.random_state),
      shuffle: !!properties.shuffle,
      loss_function: properties.loss_function,
      optimizer: properties.optimizer,
      learning_rate: Number(properties.learning_rate),
      dropout_rate: Number(properties.dropout_rate),
      batch_norm: !!properties.batch_norm,
      activation_function: properties.activation_function,
      output_activation: properties.output_activation,
      l1_reg: Number(properties.l1_reg),
      l2_reg: Number(properties.l2_reg),
      lr_scheduler: properties.lr_scheduler,
      num_mixtures: parseInt(properties.num_mixtures),
      batch_size: parseInt(properties.batch_size),
      num_epochs: parseInt(properties.num_epochs),
    };
  }

  function validateCommonTrainingInputs(properties, x, y) {
    if (!x || !y) {
      return "Missing x or y input";
    }

    const valSize = Number(properties.val_size);
    const testSize = Number(properties.test_size);
    if (
      !Number.isFinite(valSize) ||
      !Number.isFinite(testSize) ||
      valSize <= 0 ||
      testSize <= 0 ||
      valSize + testSize >= 1
    ) {
      return "Validation/Test split must satisfy: val_size > 0, test_size > 0, and val_size + test_size < 1";
    }

    const batchSize = parseInt(properties.batch_size);
    const numEpochs = parseInt(properties.num_epochs);
    if (!Number.isFinite(batchSize) || batchSize < 1) {
      return "Batch size must be >= 1";
    }
    if (!Number.isFinite(numEpochs) || numEpochs < 1) {
      return "Num epochs must be >= 1";
    }

    if (Array.isArray(x) && x.length === 0) {
      return "x input is empty";
    }
    if (Array.isArray(y) && y.length === 0) {
      return "y input is empty";
    }
    if (Array.isArray(x) && Array.isArray(y) && x.length !== y.length) {
      return "x and y must have matching sample counts";
    }

    return null;
  }

  function emptyTrainingResult(error) {
    return {
      model: {},
      x_data: {},
      y_data: {},
      x_data_scaled: {},
      y_data_scaled: {},
      x_normalizer: {},
      y_normalizer: {},
      history: {},
      error: error || "Invalid training input",
    };
  }

  async function postJsonWithErrors(endpoint, payload) {
    const runId =
      payload && payload.training_run_id
        ? String(payload.training_run_id)
        : makeTrainingRunId();
    const payloadWithRunId = {
      ...(payload || {}),
      training_run_id: runId,
    };

    const controller = new AbortController();
    inFlightTrainingControllers.add(controller);
    activeTrainingRunIds.add(runId);

    let response;
    try {
      response = await window.MeltApi.fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payloadWithRunId),
        signal: controller.signal,
      });
    } catch (err) {
      if (err && err.name === "AbortError") {
        throw new Error("Training request cancelled by user.");
      }
      throw err;
    } finally {
      inFlightTrainingControllers.delete(controller);
      activeTrainingRunIds.delete(runId);
    }

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
      const error = new Error(errMsg);
      error.detail = json;
      error.status = response.status;
      throw error;
    }

    return json;
  }

  function getTrainingExtractors() {
    const parseSplitPayload = (value) => {
      if (typeof value === "string") {
        try {
          return JSON.parse(value);
        } catch (_err) {
          return value;
        }
      }
      return value;
    };

    const modelDict = (j) => ({
      model_id: j && j.model_id ? j.model_id : null,
      model_meta: j && j.model_meta ? j.model_meta : {},
    });

    return [
      modelDict,
      (j) => parseSplitPayload((j && j.x_data) || {}),
      (j) => parseSplitPayload((j && j.y_data) || {}),
      (j) => parseSplitPayload((j && j.x_data_scaled) || {}),
      (j) => parseSplitPayload((j && j.y_data_scaled) || {}),
      (j) => (j && j.x_normalizer) || {},
      (j) => (j && j.y_normalizer) || {},
      (j) => (j && j.history) || {},
    ];
  }

  function bindTrainingPromises(node) {
    node._pModel = node._p0;
    node.pXdata = node._p1;
    node.pYdata = node._p2;
    node.pXdataScaled = node._p3;
    node.pYdataScaled = node._p4;
    node._pXNormalizer = node._p5;
    node._pYNormalizer = node._p6;
    node._pHistory = node._p7;
  }

  window.TrainerNodeShared = {
    parseTrainerInputs,
    parseTrainerInputsWithOptions,
    buildCommonTrainingPayload,
    validateCommonTrainingInputs,
    emptyTrainingResult,
    abortInFlightTrainingRequests,
    postJsonWithErrors,
    getTrainingExtractors,
    bindTrainingPromises,
    defaults: [{}, {}, {}, {}, {}, {}, {}, {}],
  };
})();
