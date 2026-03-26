"use strict";

const { entrypoints } = require("uxp");
const photoshop = require("photoshop");

const { app, core, imaging } = photoshop;

const DEFAULT_BACKEND_URL = "http://127.0.0.1:8765";
const SETTINGS_KEY_BACKEND_URL = "birefnet.backend_url";

let busy = false;

function toBase64(uint8) {
  let binary = "";
  const chunkSize = 0x8000;
  for (let i = 0; i < uint8.length; i += chunkSize) {
    const chunk = uint8.subarray(i, i + chunkSize);
    binary += String.fromCharCode.apply(null, chunk);
  }
  return btoa(binary);
}

function fromBase64(b64) {
  const binary = atob(b64);
  const output = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) {
    output[i] = binary.charCodeAt(i);
  }
  return output;
}

function clampBackendUrl(url) {
  const trimmed = (url || "").trim();
  return trimmed || DEFAULT_BACKEND_URL;
}

function status(rootNode, message, level = "info") {
  const statusEl = rootNode.querySelector("#status");
  statusEl.className = `status ${level}`;
  statusEl.textContent = message;
}

function setBusy(rootNode, value) {
  busy = value;
  const runButton = rootNode.querySelector("#run");
  runButton.disabled = value;
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

  const sourceBounds = pixels.sourceBounds;
  const { width, height } = boundsToWidthHeight(sourceBounds);

  if (width < 1 || height < 1) {
    pixels.imageData.dispose();
    throw new Error("Selected layer has no raster pixels.");
  }

  const data = await pixels.imageData.getData();
  pixels.imageData.dispose();

  let uint8;
  if (data instanceof Uint8Array) {
    uint8 = data;
  } else if (ArrayBuffer.isView(data)) {
    uint8 = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  } else {
    uint8 = new Uint8Array(data);
  }
  const pixelCount = width * height;
  const components = Math.max(1, Math.round(uint8.length / pixelCount));

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
      { commandName: "Apply BiRefNet Mask" }
    );
  } finally {
    maskImageData.dispose();
  }
}

async function removeBackground(rootNode) {
  if (busy) {
    return;
  }

  setBusy(rootNode, true);

  try {
    const { doc, layer } = validateSingleLayerSelection();
    const backendInput = rootNode.querySelector("#backendUrl");
    const backendUrl = clampBackendUrl(backendInput.value);

    localStorage.setItem(SETTINGS_KEY_BACKEND_URL, backendUrl);
    status(rootNode, "Reading layer pixels...", "info");

    const payload = await readSelectedLayerPixels(layer, doc);
    status(rootNode, "Running BiRefNet locally...", "info");

    const result = await fetchMaskFromBackend(backendUrl, payload);

    const maskBytes = fromBase64(result.maskB64 || "");
    if (maskBytes.length !== payload.width * payload.height) {
      throw new Error("Mask size mismatch from backend.");
    }

    status(rootNode, "Applying mask to selected layer...", "info");
    await applyMaskToLayer(
      layer.id,
      maskBytes,
      payload.width,
      payload.height,
      payload.left,
      payload.top
    );

    const modelName = result.modelId || "BiRefNet";
    const deviceName = result.device || "local";
    status(
      rootNode,
      `Mask applied successfully.\nModel: ${modelName}\nDevice: ${deviceName}`,
      "ok"
    );
  } catch (err) {
    status(
      rootNode,
      `${err.message}\n\nMake sure local backend is running at the configured URL.`,
      "error"
    );
  } finally {
    setBusy(rootNode, false);
  }
}

function renderPanel(rootNode) {
  rootNode.innerHTML = `
    <div class="container">
      <div class="title">BiRefNet Local Background Removal</div>
      <div class="hint">Select one layer, then run. The plugin creates/replaces the layer mask with a BiRefNet prediction.</div>
      <div class="field">
        <label for="backendUrl">Backend URL</label>
        <input id="backendUrl" type="text" />
      </div>
      <div class="actions">
        <button id="run" type="button">Remove Background</button>
      </div>
      <div id="status" class="status info">Ready.</div>
    </div>
  `;

  const backendEl = rootNode.querySelector("#backendUrl");
  backendEl.value = clampBackendUrl(localStorage.getItem(SETTINGS_KEY_BACKEND_URL));
  backendEl.addEventListener("change", () => {
    backendEl.value = clampBackendUrl(backendEl.value);
    localStorage.setItem(SETTINGS_KEY_BACKEND_URL, backendEl.value);
  });

  const runButton = rootNode.querySelector("#run");
  runButton.addEventListener("click", () => removeBackground(rootNode));
}

entrypoints.setup({
  panels: {
    birefnetPanel: {
      create(rootNode) {
        renderPanel(rootNode);
      },
      show() {},
      hide() {},
      destroy() {},
    },
  },
});
