from __future__ import annotations

import base64
import logging
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel

from .engine import BytePredictionEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictInput(BaseModel):
    prefix: str  # base64-encoded byte buffer
    min_prob: float = 0.0


class PredictRequest(BaseModel):
    inputs: list[PredictInput]


engine = BytePredictionEngine()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await engine.start()
    yield
    await engine.shutdown()


app = FastAPI(lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=500)


@app.post("/predict")
async def predict(req: PredictRequest):
    raw_inputs = [
        (base64.b64decode(inp.prefix), inp.min_prob)
        for inp in req.inputs
    ]
    t0 = time.perf_counter()
    predictions = await engine.predict_batch(raw_inputs)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info("predict: %d inputs, %.1fms", len(raw_inputs), elapsed_ms)
    return {"predictions": predictions}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="llm-dasher backend server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run("llm_dasher.server:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
