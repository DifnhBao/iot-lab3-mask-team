"""WebSocket frame inference API. No synthetic predictions in normal operation."""
import asyncio
import hashlib
import io
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from time import perf_counter_ns

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from server.inference import LABELS, MaskDetector

MAX_BYTES = 5 * 1024 * 1024
MAX_EDGE = 4096
logger = logging.getLogger(__name__)


class FrameHeader(BaseModel):
    model_config = ConfigDict(extra="forbid")
    frame_id: int = Field(ge=0)
    session_id: str = Field(min_length=1, max_length=80)
    source: str = Field(min_length=1, max_length=80)


def decode_frame(data: bytes) -> np.ndarray:
    if not data or len(data) > MAX_BYTES:
        raise ValueError("Empty image or JPEG/PNG exceeds 5 MiB")
    with Image.open(io.BytesIO(data)) as image:
        if image.format not in {"JPEG", "PNG"}:
            raise ValueError("Only JPEG/PNG images are accepted")
        if max(image.size) > MAX_EDGE:
            raise ValueError("Image exceeds 4096 pixels per edge")
        return np.asarray(image.convert("RGB"))[:, :, ::-1].copy()


def create_app(detector=None, model_version="mask-v1") -> FastAPI:
    """An injected detector is used by protocol tests only."""
    @asynccontextmanager
    async def lifespan(app):
        if app.state.detector is None:
            weights = Path(os.environ.get("MASK_MODEL_PATH", "models/best.pt"))
            if weights.is_file():
                try:
                    app.state.detector = await asyncio.to_thread(MaskDetector, str(weights))
                    app.state.model_sha256 = hashlib.sha256(weights.read_bytes()).hexdigest()
                    app.state.unready_reason = ""
                except Exception:
                    logger.exception("Could not initialize trained mask model")
                    app.state.unready_reason = "MODEL_LOAD_FAILED"
            else:
                app.state.unready_reason = "MODEL_FILE_MISSING"
        yield

    app = FastAPI(title="Lab3 Mask Detection", lifespan=lifespan)
    app.state.detector = detector
    app.state.model_sha256 = None
    app.state.unready_reason = "" if detector is not None else "MODEL_NOT_READY"
    inference_lock = asyncio.Lock()

    @app.get("/health")
    async def health():
        ready = app.state.detector is not None
        return JSONResponse({"ready": ready, "model_version": model_version,
                             "model_sha256": app.state.model_sha256, "names": LABELS,
                             "reason": app.state.unready_reason},
                            status_code=200 if ready else 503)

    @app.websocket("/ws/predict")
    async def predict_stream(websocket: WebSocket):
        await websocket.accept()
        if app.state.detector is None:
            await websocket.send_json({"type": "error", "code": app.state.unready_reason,
                                       "message": "Place trained three-class weights at models/best.pt"})
            await websocket.close(code=1013)
            return
        try:
            while True:
                # Protocol: one JSON header, then one binary image, then one result.
                try:
                    header = FrameHeader.model_validate_json(await websocket.receive_text())
                except ValidationError:
                    await websocket.send_json({"type": "error", "code": "INVALID_HEADER"})
                    await websocket.close(code=1008)
                    return
                data = await websocket.receive_bytes()
                handler_start = perf_counter_ns()
                try:
                    image = await asyncio.to_thread(decode_frame, data)
                except (ValueError, UnidentifiedImageError, OSError, Image.DecompressionBombError):
                    await websocket.send_json({"type": "error", "code": "INVALID_IMAGE",
                                               **header.model_dump()})
                    continue
                decode_ms = (perf_counter_ns() - handler_start) / 1_000_000
                try:
                    async with inference_lock:
                        result = await asyncio.to_thread(app.state.detector.predict, image)
                    timing = result["timing_ms"]
                    timing["decode"] = decode_ms
                    timing["server_processing"] = (perf_counter_ns() - handler_start) / 1_000_000
                    await websocket.send_json({"type": "result", **header.model_dump(),
                                               "model_version": model_version,
                                               "image": {"width": image.shape[1], "height": image.shape[0]},
                                               **result})
                except Exception:
                    logger.exception("Inference failed")
                    await websocket.send_json({"type": "error", "code": "INFERENCE_FAILED",
                                               **header.model_dump()})
        except WebSocketDisconnect:
            pass

    return app


app = create_app()
