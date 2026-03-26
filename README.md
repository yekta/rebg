# BiRefNet Local Photoshop Plugin (Apple Silicon)

This project adds a Photoshop UXP panel that:

1. Reads pixels from your currently selected layer.
2. Sends them to a local BiRefNet backend running on your Mac.
3. Receives a grayscale foreground mask.
4. Creates/replaces the selected layer's layer mask with that output.

Inference stays local on your machine.  
Default model: `ZhengPeng7/BiRefNet_dynamic` (official BiRefNet model for general use).

## Project Structure

- `photoshop-plugin/` UXP plugin you load in Adobe UXP Developer Tool.
- `backend/` Local Python service (FastAPI + PyTorch + BiRefNet).

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

- `http://127.0.0.1:8765`

Quick check:

```bash
curl http://127.0.0.1:8765/health
```

## 2) Load Photoshop Plugin

1. Open **Adobe UXP Developer Tool**.
2. Choose **Add Plugin** -> select folder: `photoshop-plugin`.
3. Launch the plugin in Photoshop.
4. Open panel: **Plugins -> BiRefNet Remove BG**.

## 3) Use It

1. Select exactly one layer in Photoshop.
2. In the panel, keep backend URL as `http://127.0.0.1:8765` (or your custom local URL).
3. Click **Remove Background**.
4. Plugin applies the generated mask to that selected layer.

## Backend Configuration (Optional)

Environment variables before starting backend:

- `BIREFNET_MODEL_ID` (default `ZhengPeng7/BiRefNet_dynamic`)
- `BIREFNET_DEVICE` (`auto`, `mps`, or `cpu`; default `auto`)
- `BIREFNET_USE_HALF` (`1` to enable FP16 on MPS, default `0`)
- `BIREFNET_HOST` (default `127.0.0.1`)
- `BIREFNET_PORT` (default `8765`)

Example:

```bash
cd backend
BIREFNET_MODEL_ID=ZhengPeng7/BiRefNet_HR BIREFNET_DEVICE=mps ./run_backend.sh
```

## Notes

- First run can take time because model weights are downloaded and cached.
- Very large layers can be slow; this is expected with local high-quality segmentation.
