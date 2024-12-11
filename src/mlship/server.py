from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import json
import asyncio
from pathlib import Path

from .models.wrapper import ModelWrapper

app = FastAPI()

# Get the directory containing this file
current_dir = Path(__file__).parent.resolve()

# Mount static files
app.mount("/static", StaticFiles(directory=str(current_dir / "ui" / "static")), name="static")

# Setup templates
templates = Jinja2Templates(directory=str(current_dir / "ui" / "templates"))

# Global model instance
model = None

def get_model_info():
    """Get information about the loaded model."""
    if model is None:
        return {}
    
    return {
        "type": model.type,
        "input_type": model.input_type,
        "output_type": model.output_type,
        "features": model.features if hasattr(model, "features") else [],
        "n_features": model.n_features if hasattr(model, "n_features") else None
    }

@app.get("/ui", response_class=HTMLResponse)
async def get_ui(request: Request):
    """Serve the UI page."""
    model_info = get_model_info()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "model_info": json.dumps(model_info)
        }
    )

@app.get("/api/model-info")
async def get_model_info_api():
    """Get information about the loaded model."""
    return get_model_info()

@app.websocket("/ws/metrics")
async def websocket_metrics(websocket: WebSocket):
    """WebSocket endpoint for real-time metrics."""
    await websocket.accept()
    try:
        while True:
            # Send metrics every second
            metrics = {
                "requests": model.request_count if model else 0,
                "avg_latency": model.average_latency if model else 0
            }
            await websocket.send_json(metrics)
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass

def load_model(model_path: str):
    """Load a model from the given path."""
    global model
    model = ModelWrapper.load(model_path)

def start_server(model_path: str, host: str = "127.0.0.1", port: int = 8000):
    """Start the server with the given model."""
    import uvicorn
    load_model(model_path)
    uvicorn.run(app, host=host, port=port) 