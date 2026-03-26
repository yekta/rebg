"use strict";

const { entrypoints } = require("uxp");
const photoshop = require("photoshop");

const { app, core, imaging } = photoshop;

const DEFAULT_BACKEND_URL = "http://localhost:8765";
const SETTINGS_KEY_BACKEND_URL = "rebg.backend_url";

let busy = false;
const BASE64_CHARS =
  "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

function errorMessage(err) {
  if (err && typeof err.message === "string" && err.message.trim()) {
    return err.message;
  }
  return String(err || "Unknown error.");
}

function buildTroubleshootingHint(message) {
  const msg = (message || "").toLowerCase();

  if (msg.includes("permission denied to the url") || msg.includes("manifest entry not found")) {
    return "Network permission mismatch for this URL. Use http://localhost:8765, then remove/re-add plugin in UXP Developer Tool so manifest changes are applied.";
  }

  if (msg.includes("executeasmodal") || msg.includes("modal scope")) {
    return "Photoshop requires this step to run in modal scope. Please retry; if it still fails, restart Photoshop and reload the plugin.";
  }

  if (msg.includes("invalid array length") || msg.includes("pixel buffer")) {
    return "Photoshop returned an unexpected pixel buffer. Try rasterizing the selected layer or using a smaller canvas, then run again.";
  }

  if (msg.includes("rearrange-reduction") || msg.includes("chunks of")) {
    return "Backend hit a shape constraint. Restart backend with the latest code so auto resize/pad is applied.";
  }

  if (msg.includes("to have") && msg.includes("channels")) {
    return "Backend hit a tensor-channel constraint. Restart backend with the latest code so dynamic padding retries are applied.";
  }

  if (
    msg.includes("failed to fetch") ||
    msg.includes("backend error") ||
    msg.includes("networkerror")
  ) {
    return "Make sure local backend is running at the configured URL.";
  }

  return "";
}

function toBase64(uint8) {
  if (!(uint8 instanceof Uint8Array)) {
    throw new Error("Expected Uint8Array for base64 encoding.");
  }

  const parts = [];
  let block = "";
  const flushAt = 4096;
  let i = 0;

  while (i + 2 < uint8.length) {
    const n = (uint8[i] << 16) | (uint8[i + 1] << 8) | uint8[i + 2];
    block +=
      BASE64_CHARS[(n >> 18) & 63] +
      BASE64_CHARS[(n >> 12) & 63] +
      BASE64_CHARS[(n >> 6) & 63] +
      BASE64_CHARS[n & 63];
    i += 3;

    if (block.length >= flushAt) {
      parts.push(block);
      block = "";
    }
  }

  const remaining = uint8.length - i;
  if (remaining === 1) {
    const n = uint8[i];
    block +=
      BASE64_CHARS[(n >> 2) & 63] + BASE64_CHARS[(n & 3) << 4] + "==";
  } else if (remaining === 2) {
    const n = (uint8[i] << 8) | uint8[i + 1];
    block +=
      BASE64_CHARS[(n >> 10) & 63] +
      BASE64_CHARS[(n >> 4) & 63] +
      BASE64_CHARS[(n & 15) << 2] +
      "=";
  }

  if (block) {
    parts.push(block);
  }

  return parts.join("");
}

function fromBase64(b64) {
  let binary;
  try {
    binary = atob(b64);
  } catch (_err) {
    throw new Error("Backend returned invalid base64 mask data.");
  }
  const output = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    output[i] = binary.charCodeAt(i);
  }
  return output;
}

function clampBackendUrl(url) {
  const trimmed = (url || "").trim();
  if (!trimmed) {
    return DEFAULT_BACKEND_URL;
  }

  try {
    const parsed = new URL(trimmed);
    if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
      return DEFAULT_BACKEND_URL;
    }

    // UXP loopback permission matching is more reliable with localhost.
    const hostname = parsed.hostname === "127.0.0.1" ? "localhost" : parsed.hostname;
    const port = parsed.port ? `:${parsed.port}` : "";
    return `${parsed.protocol}//${hostname}${port}`;
  } catch (_err) {
    return trimmed;
  }
}

function normalizePixelBuffer(data) {
  if (data instanceof Uint8Array) {
    return data;
  }

  if (data instanceof ArrayBuffer) {
    return new Uint8Array(data);
  }

  if (ArrayBuffer.isView(data)) {
    return new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  }

  throw new Error("Unsupported pixel data returned by Photoshop.");
}

function getStoredBackendUrl() {
  try {
    if (typeof localStorage !== "undefined") {
      return localStorage.getItem(SETTINGS_KEY_BACKEND_URL);
    }
  } catch (_err) {}
  return null;
}

function setStoredBackendUrl(url) {
  try {
    if (typeof localStorage !== "undefined") {
      localStorage.setItem(SETTINGS_KEY_BACKEND_URL, url);
    }
  } catch (_err) {}
}

function status(rootNode, message, level = "info") {
  const statusEl = rootNode.querySelector("#status");
  if (!statusEl) {
    return;
  }
  statusEl.className = `status ${level}`;
  statusEl.textContent = message;
}

function setBusy(rootNode, value) {
  busy = value;
  const runButton = rootNode.querySelector("#run");
  if (runButton) {
    runButton.disabled = value;
  }
}

function validateSingleLayerSelection() {
  const doc = app.activeDocument;
  if (!doc) {
    throw new Error("No active Photoshop document.");
  }

  const layers = doc.activeLayers || [];
  if (layers.length !== 1) {
    throw new Error("Select exactly one layer.");
  }

  return { doc, layer: layers[0] };
}

function boundsToWidthHeight(bounds) {
  const width = Math.max(0, Math.round(bounds.right - bounds.left));
  const height = Math.max(0, Math.round(bounds.bottom - bounds.top));
  return { width, height };
}

async function readSelectedLayerPixels(layer, doc) {
  const pixels = await imaging.getPixels({
    documentID: doc.id,
    layerID: layer.id,
    componentSize: 8,
    applyAlpha: false,
  });

  if (!pixels || !pixels.sourceBounds || !pixels.imageData) {
    throw new Error("Could not read raster pixels from selected layer.");
  }

  const sourceBounds = pixels.sourceBounds;
  const { width, height } = boundsToWidthHeight(sourceBounds);

  if (width < 1 || height < 1) {
    pixels.imageData.dispose();
    throw new Error("Selected layer has no raster pixels.");
  }

  const data = await pixels.imageData.getData();
  pixels.imageData.dispose();

  const uint8 = normalizePixelBuffer(data);
  const pixelCount = width * height;
  if (pixelCount < 1) {
    throw new Error("Selected layer has invalid bounds.");
  }

  if (uint8.length % pixelCount !== 0) {
    throw new Error(
      `Unexpected pixel buffer size (${uint8.length}) for layer bounds ${width}x${height}.`
    );
  }

  const components = uint8.length / pixelCount;
  if (components < 1 || components > 8) {
    throw new Error(`Unsupported component count returned by Photoshop: ${components}`);
  }

  return {
    width,
    height,
    components,
    left: sourceBounds.left,
    top: sourceBounds.top,
    pixelsB64: toBase64(uint8),
  };
}

async function fetchMaskFromBackend(backendBaseUrl, layerPayload) {
  const endpoint = `${backendBaseUrl.replace(/\/+$/, "")}/segment`;
  const response = await fetch(endpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(layerPayload),
  });

  const text = await response.text();
  if (!response.ok) {
    throw new Error(
      `Backend error ${response.status}: ${text || "No response body"}`
    );
  }

  try {
    return JSON.parse(text);
  } catch (_err) {
    throw new Error("Backend returned invalid JSON.");
  }
}

async function applyMaskToLayer(layerID, maskBytes, width, height, left, top) {
  const maskImageData = await imaging.createImageDataFromBuffer(maskBytes, {
    width,
    height,
    components: 1,
    colorSpace: "Grayscale",
    colorProfile: "Gray Gamma 2.2",
    chunky: true,
  });

  try {
    await core.executeAsModal(
      async () => {
        await imaging.putLayerMask({
          layerID,
          imageData: maskImageData,
          replace: true,
          targetBounds: { left, top },
        });
      },
      { commandName: "Apply Layer Mask" }
    );
  } finally {
    maskImageData.dispose();
  }
}

async function readSelectedLayerPayloadInModal() {
  return core.executeAsModal(
    async () => {
      const { doc, layer } = validateSingleLayerSelection();
      const payload = await readSelectedLayerPixels(layer, doc);
      return { payload, layerID: layer.id };
    },
    { commandName: "Read Layer Pixels" }
  );
}

async function removeBackground(rootNode) {
  if (busy) {
    return;
  }

  setBusy(rootNode, true);

  try {
    const backendInput = rootNode.querySelector("#backendUrl");
    const backendUrl = clampBackendUrl(backendInput.value);

    setStoredBackendUrl(backendUrl);
    status(rootNode, "Reading layer pixels...", "info");

    const { payload, layerID } = await readSelectedLayerPayloadInModal();
    status(rootNode, "Processing image...", "info");

    const result = await fetchMaskFromBackend(backendUrl, payload);

    const maskBytes = fromBase64(result.maskB64 || "");
    if (maskBytes.length !== payload.width * payload.height) {
      throw new Error("Mask size mismatch from backend.");
    }

    status(rootNode, "Applying mask to selected layer...", "info");
    await applyMaskToLayer(
      layerID,
      maskBytes,
      payload.width,
      payload.height,
      payload.left,
      payload.top
    );

    status(
      rootNode,
      "Background removed successfully.",
      "ok"
    );
  } catch (err) {
    const message = errorMessage(err);
    const hint = buildTroubleshootingHint(message);
    status(
      rootNode,
      hint ? `${message}\n\n${hint}` : message,
      "error"
    );
  } finally {
    setBusy(rootNode, false);
  }
}

function renderPanel(rootNode) {
  rootNode.innerHTML = `
    <div class="panel">
      <div class="field">
        <label class="field-label" for="backendUrl">Backend URL</label>
        <sp-textfield
          id="backendUrl"
          spellcheck="false"
          placeholder="http://localhost:8765"
        ></sp-textfield>
      </div>
      <div class="actions">
        <sp-button id="run" variant="secondary">Remove Background</sp-button>
      </div>
      <div id="status" class="status info">Ready.</div>
    </div>
  `;

  const backendEl = rootNode.querySelector("#backendUrl");
  backendEl.value = clampBackendUrl(getStoredBackendUrl());
  backendEl.addEventListener("change", () => {
    backendEl.value = clampBackendUrl(backendEl.value);
    setStoredBackendUrl(backendEl.value);
  });

  const runButton = rootNode.querySelector("#run");
  runButton.addEventListener("click", () => removeBackground(rootNode));
}

entrypoints.setup({
  panels: {
    rebgPanel: {
      create(rootNode) {
        try {
          renderPanel(rootNode);
        } catch (err) {
          console.error("[Rebg] Panel initialization failed:", err);
          rootNode.innerHTML = `
            <div class="panel">
              <div id="status" class="status error">Panel initialization failed:\n${errorMessage(
                err
              )}</div>
            </div>
          `;
        }
      },
      show() {},
      hide() {},
      destroy() {},
    },
  },
});
