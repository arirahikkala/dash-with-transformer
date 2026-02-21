from __future__ import annotations

import base64
import logging
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .engine import BytePredictionEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictRequest(BaseModel):
    inputs: list[str]  # base64-encoded byte buffers


class PredictResponse(BaseModel):
    predictions: list[list[float]]


engine = BytePredictionEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await engine.start()
    yield
    await engine.shutdown()


app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    raw_inputs = [base64.b64decode(b64) for b64 in req.inputs]
    t0 = time.perf_counter()
    predictions = await engine.predict_batch(raw_inputs)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info("predict: %d inputs, %.1fms", len(raw_inputs), elapsed_ms)
    return PredictResponse(predictions=predictions)


def main():
    uvicorn.run("llm_dasher.server:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
