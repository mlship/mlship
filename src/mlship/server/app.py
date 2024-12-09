import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import Any, List, Optional
import logging
import time
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

METRICS_FILE = Path.home() / '.mlship' / 'metrics.json'

class PredictRequest(BaseModel):
    inputs: Any

class PredictResponse(BaseModel):
    predictions: List
    metadata: Optional[dict] = None

def convert_numpy_types(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                         np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

class ModelServer:
    def __init__(self, model):
        self.model = model
        self.app = self._create_app()
        self.request_count = 0
        self.total_latency = 0

    def update_metrics(self):
        """Update server metrics file"""
        avg_latency = self.total_latency / self.request_count if self.request_count > 0 else 0
        metrics = {
            'start_time': getattr(self, 'start_time', time.time()),
            'requests': self.request_count,
            'avg_latency': round(avg_latency, 2)
        }
        with open(METRICS_FILE, 'w') as f:
            json.dump(metrics, f)

    def _create_app(self):
        app = FastAPI(title="MLShip Server")

        @app.post("/predict", response_model=PredictResponse)
        async def predict(request: PredictRequest):
            try:
                start_time = time.time()
                logger.info(f"Processing prediction request")
                predictions = self.model.predict(request.inputs)
                # Convert all numpy types to Python native types
                predictions = convert_numpy_types(predictions)
                
                # Update metrics
                self.request_count += 1
                self.total_latency += (time.time() - start_time) * 1000  # Convert to ms
                self.update_metrics()
                
                return {
                    "predictions": predictions,
                    "metadata": {"model_version": getattr(self.model, "version", "unknown")}
                }
            except Exception as e:
                logger.error(f"Prediction error: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))

        @app.get("/health")
        async def health():
            return {"status": "healthy"}

        return app

    def serve(self, host="0.0.0.0", port=8000):
        logger.info(f"Starting server on {host}:{port}")
        self.start_time = time.time()
        uvicorn.run(self.app, host=host, port=port)