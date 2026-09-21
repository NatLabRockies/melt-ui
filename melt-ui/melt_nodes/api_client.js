(function (global) {
  "use strict";

  function resolveEndpoint(endpoint) {
    const raw = String(endpoint ?? "").trim();
    if (!raw) {
      throw new Error("MELT API endpoint is empty.");
    }

    const origin = global.location?.origin;
    if (!origin || origin === "null") {
      throw new Error("MELT API requires a valid application origin.");
    }

    let url;
    try {
      url = new URL(raw, `${origin}/`);
    } catch {
      throw new Error(`Invalid MELT API endpoint: ${raw}`);
    }

    if (url.origin !== origin) {
      throw new Error(
        `Blocked cross-origin MELT API endpoint: ${url.origin}`,
      );
    }

    if (url.username || url.password) {
      throw new Error("MELT API endpoints may not contain credentials.");
    }

    return `${url.pathname}${url.search}`;
  }

  function apiFetch(endpoint, options = {}) {
    const requestOptions = { ...options };
    if (!Object.prototype.hasOwnProperty.call(requestOptions, "credentials")) {
      requestOptions.credentials = "same-origin";
    }

    return global.fetch(resolveEndpoint(endpoint), requestOptions);
  }

  global.MeltApi = Object.freeze({
    resolveEndpoint,
    fetch: apiFetch,
  });
})(typeof window !== "undefined" ? window : globalThis);
