from __future__ import annotations

import base64
import math
import os
import re
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


def env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError:
        value = default
    return max(minimum, value)


MAX_SIDE = env_int("BIREFNET_MAX_SIDE", 2048)
ALIGN_HEIGHT = env_int("BIREFNET_ALIGN_HEIGHT", 32)
ALIGN_WIDTH = env_int("BIREFNET_ALIGN_WIDTH", 32)

IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
SHAPE_CHUNK_RE = re.compile(r"axis of length (\d+) in chunks of (\d+)")


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
    max_side: int
    align_height: int
    align_width: int


@dataclass
class PreparedInput:
    original_h: int
    original_w: int
    scaled_h: int
    scaled_w: int
    padded_h: int
    padded_w: int


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

    @staticmethod
    def _resize_rgb(rgb: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        if rgb.shape[0] == target_h and rgb.shape[1] == target_w:
            return rgb

        tensor = (
            torch.from_numpy(np.ascontiguousarray(rgb))
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
        )
        resized = torch.nn.functional.interpolate(
            tensor, size=(target_h, target_w), mode="bilinear", align_corners=False
        )
        return (
            resized.squeeze(0)
            .permute(1, 2, 0)
            .clamp(0, 255)
            .to(torch.uint8)
            .cpu()
            .numpy()
        )

    @staticmethod
    def _resize_mask(mask: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        if mask.shape[0] == target_h and mask.shape[1] == target_w:
            return mask.astype(np.float32, copy=False)

        tensor = torch.from_numpy(np.ascontiguousarray(mask)).unsqueeze(0).unsqueeze(0).float()
        resized = torch.nn.functional.interpolate(
            tensor, size=(target_h, target_w), mode="bilinear", align_corners=False
        )
        return resized.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32, copy=False)

    @staticmethod
    def _pad_rgb(rgb: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        pad_h = max(0, target_h - rgb.shape[0])
        pad_w = max(0, target_w - rgb.shape[1])
        if pad_h == 0 and pad_w == 0:
            return rgb
        return np.pad(rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")

    @staticmethod
    def _next_multiple(value: int, divisor: int) -> int:
        divisor = max(1, divisor)
        return ((value + divisor - 1) // divisor) * divisor

    def _aligned_size(self, h: int, w: int) -> tuple[int, int]:
        align_h = max(1, self.config.align_height)
        align_w = max(1, self.config.align_width)
        aligned_h = self._next_multiple(h, align_h)
        aligned_w = self._next_multiple(w, align_w)
        return aligned_h, aligned_w

    def _prepare_input(self, rgb: np.ndarray) -> tuple[np.ndarray, PreparedInput]:
        original_h, original_w = rgb.shape[:2]
        max_side = max(original_h, original_w)

        scale = 1.0
        if max_side > self.config.max_side:
            scale = self.config.max_side / float(max_side)

        scaled_h = max(1, int(round(original_h * scale)))
        scaled_w = max(1, int(round(original_w * scale)))
        scaled_rgb = self._resize_rgb(rgb, scaled_h, scaled_w)

        padded_h, padded_w = self._aligned_size(scaled_h, scaled_w)
        padded_rgb = self._pad_rgb(scaled_rgb, padded_h, padded_w)

        prep = PreparedInput(
            original_h=original_h,
            original_w=original_w,
            scaled_h=scaled_h,
            scaled_w=scaled_w,
            padded_h=padded_h,
            padded_w=padded_w,
        )
        return padded_rgb, prep

    def _infer_probability_map(self, rgb: np.ndarray) -> np.ndarray:
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
        return np.clip(pred_np, 0.0, 1.0).astype(np.float32, copy=False)

    def _try_padding_retry(
        self, prepared_rgb: np.ndarray, prep: PreparedInput, error: Exception
    ) -> np.ndarray | None:
        text = str(error)
        required_h = max(1, self.config.align_height)
        required_w = max(1, self.config.align_width)
        needs_retry = False

        for axis_len_s, chunk_s in SHAPE_CHUNK_RE.findall(text):
            axis_len = int(axis_len_s)
            chunk = max(1, int(chunk_s))

            if axis_len == prep.padded_h:
                required_h = math.lcm(required_h, chunk)
                needs_retry = True
            if axis_len == prep.padded_w:
                required_w = math.lcm(required_w, chunk)
                needs_retry = True

        # Fallback for channel-mismatch style failures (e.g. expected 3072, got 2976):
        # enforce the common 32-alignment for both axes before retry.
        if "to have" in text and "channels" in text:
            required_h = math.lcm(required_h, 32)
            required_w = math.lcm(required_w, 32)
            needs_retry = True

        if not needs_retry:
            return None

        target_h = self._next_multiple(prep.padded_h, required_h)
        target_w = self._next_multiple(prep.padded_w, required_w)
        if target_h == prep.padded_h and target_w == prep.padded_w:
            return None

        prep.padded_h = target_h
        prep.padded_w = target_w
        return self._pad_rgb(prepared_rgb, target_h, target_w)

    def _postprocess_prediction(self, pred_np: np.ndarray, prep: PreparedInput) -> np.ndarray:
        if pred_np.shape != (prep.padded_h, prep.padded_w):
            pred_np = self._resize_mask(pred_np, prep.padded_h, prep.padded_w)

        pred_np = pred_np[: prep.scaled_h, : prep.scaled_w]
        if pred_np.shape != (prep.original_h, prep.original_w):
            pred_np = self._resize_mask(pred_np, prep.original_h, prep.original_w)

        return np.clip(pred_np, 0.0, 1.0).astype(np.float32, copy=False)

    def segment(self, rgb: np.ndarray, alpha: np.ndarray | None) -> np.ndarray:
        prepared_rgb, prep = self._prepare_input(rgb)
        last_exc: Exception | None = None
        pred_np: np.ndarray | None = None

        for _ in range(3):
            try:
                pred_np = self._infer_probability_map(prepared_rgb)
                break
            except Exception as exc:
                last_exc = exc
                retry_input = self._try_padding_retry(prepared_rgb, prep, exc)
                if retry_input is None:
                    raise
                prepared_rgb = retry_input
        else:
            if last_exc is not None:
                raise last_exc
            raise RuntimeError("Inference failed after retries.")

        if pred_np is None:
            raise RuntimeError("Inference returned no prediction.")

        pred_np = self._postprocess_prediction(pred_np, prep)
        mask = (pred_np * 255.0).astype(np.uint8)

        if alpha is not None:
            if alpha.shape != mask.shape:
                alpha = np.clip(
                    self._resize_mask(alpha.astype(np.float32), mask.shape[0], mask.shape[1]),
                    0.0,
                    255.0,
                ).astype(np.uint8)
            mask = (mask.astype(np.float32) * (alpha.astype(np.float32) / 255.0)).astype(np.uint8)

        return mask


runtime = RuntimeConfig(
    model_id=MODEL_ID,
    device=pick_device(REQUESTED_DEVICE),
    use_half=USE_HALF,
    max_side=MAX_SIDE,
    align_height=ALIGN_HEIGHT,
    align_width=ALIGN_WIDTH,
)
runner = BiRefNetRunner(runtime)

app = FastAPI(title="BiRefNet Local Backend", version="0.1.0")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "modelId": runtime.model_id,
        "device": runtime.device,
        "maxSide": runtime.max_side,
        "alignHeight": runtime.align_height,
        "alignWidth": runtime.align_width,
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
