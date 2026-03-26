from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForImageSegmentation

MODEL_ID = os.getenv("BIREFNET_MODEL_ID", "ZhengPeng7/BiRefNet_dynamic")
REQUESTED_DEVICE = os.getenv("BIREFNET_DEVICE", "auto").strip().lower()
USE_HALF = os.getenv("BIREFNET_USE_HALF", "0").strip() == "1"

IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class SegmentRequest(BaseModel):
    width: int = Field(..., gt=0, le=12000)
    height: int = Field(..., gt=0, le=12000)
    components: int = Field(..., gt=0, le=8)
    pixelsB64: str = Field(..., min_length=4)


class SegmentResponse(BaseModel):
    width: int
    height: int
    maskB64: str
    modelId: str
    device: str


@dataclass(frozen=True)
class RuntimeConfig:
    model_id: str
    device: str
    use_half: bool


def pick_device(requested: str) -> str:
    if requested in {"mps", "cpu"}:
        if requested == "mps" and not torch.backends.mps.is_available():
            return "cpu"
        return requested

    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def extract_prediction_tensor(output: Any) -> torch.Tensor:
    candidate = output

    if isinstance(candidate, dict):
        if "preds" in candidate:
            candidate = candidate["preds"]
        elif "logits" in candidate:
            candidate = candidate["logits"]
        else:
            candidate = next(iter(candidate.values()))

    if hasattr(candidate, "logits"):
        candidate = candidate.logits

    if isinstance(candidate, (list, tuple)):
        candidate = candidate[-1]

    if not torch.is_tensor(candidate):
        raise RuntimeError("Model output did not contain a tensor prediction.")

    return candidate


def decode_pixels(req: SegmentRequest) -> tuple[np.ndarray, np.ndarray | None]:
    try:
        raw = base64.b64decode(req.pixelsB64, validate=True)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=400, detail=f"Invalid base64 payload: {exc}") from exc

    expected = req.width * req.height * req.components
    if len(raw) != expected:
        raise HTTPException(
            status_code=400,
            detail=f"Pixel buffer size mismatch: got {len(raw)}, expected {expected}.",
        )

    pixels = np.frombuffer(raw, dtype=np.uint8).reshape((req.height, req.width, req.components))

    alpha = None
    if req.components >= 4:
        alpha = pixels[:, :, 3]

    if req.components == 1:
        rgb = np.repeat(pixels[:, :, 0:1], 3, axis=2)
    elif req.components >= 3:
        rgb = pixels[:, :, :3]
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported component count: {req.components}.",
        )

    return rgb, alpha


class BiRefNetRunner:
    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.model = self._load_model(config)

    @staticmethod
    def _load_model(config: RuntimeConfig) -> torch.nn.Module:
        torch.set_float32_matmul_precision("high")
        model = AutoModelForImageSegmentation.from_pretrained(
            config.model_id,
            trust_remote_code=True,
        )
        model.eval()
        model.to(config.device)

        if config.use_half and config.device != "cpu":
            model.half()

        return model

    def _preprocess(self, rgb: np.ndarray) -> torch.Tensor:
        arr = rgb.astype(np.float32) / 255.0
        arr = (arr - IMAGE_MEAN) / IMAGE_STD
        arr = np.transpose(arr, (2, 0, 1))
        tensor = torch.from_numpy(arr).unsqueeze(0)
        if self.config.use_half and self.config.device != "cpu":
            tensor = tensor.half()
        return tensor.to(self.device)

    def segment(self, rgb: np.ndarray, alpha: np.ndarray | None) -> np.ndarray:
        source_h, source_w = rgb.shape[:2]
        input_tensor = self._preprocess(rgb)

        with torch.inference_mode():
            output = self.model(input_tensor)
            pred = extract_prediction_tensor(output)
            pred = torch.sigmoid(pred)

        if pred.ndim == 4:
            pred = pred[:, 0, :, :]
        elif pred.ndim != 3:
            raise RuntimeError(f"Unexpected prediction tensor shape: {tuple(pred.shape)}")

        pred_np = pred[0].detach().float().cpu().numpy()
        pred_np = np.clip(pred_np, 0.0, 1.0)

        if pred_np.shape != (source_h, source_w):
            pred_np = np.array(
                torch.nn.functional.interpolate(
                    torch.from_numpy(pred_np).unsqueeze(0).unsqueeze(0),
                    size=(source_h, source_w),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(0)
                .squeeze(0),
                dtype=np.float32,
            )

        mask = (pred_np * 255.0).astype(np.uint8)

        if alpha is not None:
            mask = (mask.astype(np.float32) * (alpha.astype(np.float32) / 255.0)).astype(np.uint8)

        return mask


runtime = RuntimeConfig(
    model_id=MODEL_ID,
    device=pick_device(REQUESTED_DEVICE),
    use_half=USE_HALF,
)
runner = BiRefNetRunner(runtime)

app = FastAPI(title="BiRefNet Local Backend", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "modelId": runtime.model_id,
        "device": runtime.device,
    }


@app.post("/segment", response_model=SegmentResponse)
def segment(req: SegmentRequest) -> SegmentResponse:
    try:
        rgb, alpha = decode_pixels(req)
        mask = runner.segment(rgb, alpha)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return SegmentResponse(
        width=req.width,
        height=req.height,
        maskB64=base64.b64encode(mask.tobytes()).decode("ascii"),
        modelId=runtime.model_id,
        device=runtime.device,
    )


if __name__ == "__main__":
    host = os.getenv("BIREFNET_HOST", "127.0.0.1")
    port = int(os.getenv("BIREFNET_PORT", "8765"))
    uvicorn.run("server:app", host=host, port=port, reload=False)
