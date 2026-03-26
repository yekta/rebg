# Rebg Photoshop Plugin (Apple Silicon)

This project adds a Photoshop UXP panel that:

1. Reads pixels from your currently selected layer.
2. Sends them to a local backend running on your Mac.
3. Receives a grayscale foreground mask.
4. Creates/replaces the selected layer's layer mask with that output.

Inference stays local on your machine.

## Project Structure

- `photoshop-plugin/` UXP plugin you load in Adobe UXP Developer Tool.
- `backend/` Local Python service (FastAPI + PyTorch backend).

## Requirements

- macOS on Apple Silicon (M-series).
- Adobe Photoshop (UXP plugin support, PS 24+ recommended).
- Python 3.10+ (3.11 recommended).
- Internet once for initial model download from Hugging Face (after cache, runs offline).

## 1) Start Local Backend

From project root:

```bash
cd backend
./run_backend.sh
```

The script creates `.venv`, installs dependencies, and starts on:

- `http://localhost:8765`

Quick check:

```bash
curl http://localhost:8765/health
```

## 2) Load Photoshop Plugin

1. Open **Adobe UXP Developer Tool**.
2. Choose **Add Plugin** -> select folder: `photoshop-plugin`.
3. Launch the plugin in Photoshop.
4. Open panel: **Plugins -> Rebg**.

## 3) Use It

1. Select exactly one layer in Photoshop.
2. In the panel, keep backend URL as `http://localhost:8765` (or your custom local URL).
3. Click **Remove Background**.
4. Plugin applies the generated mask to that selected layer.

## Backend Configuration (Optional)

Environment variables before starting backend:

- `BIREFNET_MODEL_ID` (default `ZhengPeng7/BiRefNet_dynamic-matting`)
- `BIREFNET_DEVICE` (`auto`, `mps`, or `cpu`; default `auto`)
- `BIREFNET_USE_HALF` (`1` to enable FP16 on MPS, default `0`)
- `BIREFNET_HOST` (default `127.0.0.1`)
- `BIREFNET_PORT` (default `8765`)
- `BIREFNET_MAX_SIDE` (default `2048`; backend auto-downscales larger layers while preserving aspect ratio)
- `BIREFNET_ALIGN_HEIGHT` (default `32`; backend pads height to model-friendly multiples)
- `BIREFNET_ALIGN_WIDTH` (default `32`; backend pads width to model-friendly multiples)

Example:

```bash
cd backend
BIREFNET_MODEL_ID=ZhengPeng7/BiRefNet_HR-matting BIREFNET_DEVICE=mps ./run_backend.sh
```

Large document example:

```bash
cd backend
BIREFNET_MAX_SIDE=1536 ./run_backend.sh
```

## Notes

- First run can take time because model weights are downloaded and cached.
- Very large layers can be slow; backend now auto-resizes for inference, pads to safe alignment, then restores mask back to original layer size.

## Troubleshooting

If UXP Developer Tool says:

- `Validate command successful...`
- `Load command failed...`
- `Plugin Load Failed.`
- `Devtools: Failed to load the devtools plugin.`

check these in Photoshop:

1. `Preferences -> Plugins`: enable `Enable Developer Mode`.
2. `Preferences -> Plugins`: enable `Enable Generator`.
3. Restart Photoshop after changing those settings.

Then in UXP Developer Tool:

1. Remove this plugin entry.
2. Add it again from the `photoshop-plugin` folder.
3. Load again and open panel `Plugins -> Rebg`.

If panel shows `Permission denied to the url ... Manifest entry not found`:

1. Set backend URL to `http://localhost:8765` (not `127.0.0.1`).
2. Remove/re-add plugin in UXP Developer Tool (manifest permissions are read at load time).

If panel shows backend shape/rearrange errors on large images (for example `can't divide axis ... in chunks ...`):

1. Pull latest backend code from this repo.
2. Restart backend so auto-resize/pad logic is active.
3. Optionally reduce `BIREFNET_MAX_SIDE` (for example `1536`) if you still hit memory or speed issues.

If panel shows backend channel mismatch errors (for example `expected ... to have 3072 channels, but got 2976`):

1. Restart backend after pulling latest code (retry logic composes divisibility constraints).
2. Keep `BIREFNET_ALIGN_HEIGHT=32` and `BIREFNET_ALIGN_WIDTH=32` unless you are debugging model internals.
