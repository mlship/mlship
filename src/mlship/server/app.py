from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import json
import asyncio
import signal
import sys
from pathlib import Path
import numpy as np
from ..utils.daemon import cleanup_files
from ..models.wrapper import ModelWrapper

# Create FastAPI app
app = FastAPI()

# Get the directory containing this file
current_dir = Path(__file__).parent.resolve()
ui_dir = current_dir.parent / "ui"  # Changed to look in the correct location

# Mount static files
app.mount("/static", StaticFiles(directory=str(ui_dir / "static")), name="static")

# Setup templates
templates = Jinja2Templates(directory=str(ui_dir / "templates"))

# Global model instance and settings
model = None
enable_ui = True  # Enable UI by default

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    print("\nShutting down server...")
    cleanup_files()
    sys.exit(0)

def get_model_info():
    """Get information about the loaded model."""
    global model
    
    if model is None:
        return {
            "error": "No model loaded. Please deploy a model first.",
            "status": "not_loaded",
            "type": None
        }
    
    try:
        info = model.get_model_info()
        info["status"] = "loaded"
        info["request_count"] = model.request_count
        info["average_latency"] = model.average_latency
        
        # Ensure required fields are present
        if "type" not in info:
            info["type"] = model.model_type
        if "input_type" not in info:
            info["input_type"] = model.input_type
        if "output_type" not in info:
            info["output_type"] = model.output_type
        if "features" not in info and hasattr(model, "feature_names"):
            info["features"] = model.feature_names
        
        return info
    except Exception as e:
        print(f"Error getting model info: {str(e)}")
        return {
            "error": str(e),
            "status": "error",
            "type": None
        }

@app.get("/")
async def root():
    """Redirect to UI if enabled, otherwise show API info"""
    if enable_ui:
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/ui")
    return {"message": "MLShip API Server", "ui_enabled": enable_ui}

@app.get("/ui", response_class=HTMLResponse)
async def get_ui(request: Request):
    """Serve the UI page."""
    if not enable_ui:
        return JSONResponse(
            status_code=404,
            content={"error": "UI not enabled. Please run with --ui flag"}
        )
    
    try:
        model_info = get_model_info()
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "model_info": json.dumps(model_info)
            }
        )
    except Exception as e:
        print(f"Error serving UI: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/api/model-info")
async def get_model_info_api():
    """Get information about the loaded model."""
    return get_model_info()

@app.post("/api/predict")
async def predict(request: Request):
    """Make a prediction with the loaded model."""
    if model is None:
        return JSONResponse(
            status_code=400,
            content={"error": "No model loaded"}
        )
    
    try:
        # Get request data
        data = await request.json()
        inputs = data.get("inputs", [])
        
        # Convert inputs to numpy array
        inputs_array = np.array(inputs)
        
        # Make prediction
        predictions = model.predict(inputs_array)
        
        # Convert predictions to list if needed
        if isinstance(predictions, np.ndarray):
            predictions = predictions.tolist()
        
        return {
            "predictions": predictions if isinstance(predictions, list) else [predictions]
        }
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"error": str(e)}
        )

@app.websocket("/ws/metrics")
async def websocket_metrics(websocket: WebSocket):
    """WebSocket endpoint for real-time metrics."""
    await websocket.accept()
    try:
        while True:
            # Get current metrics
            metrics = {
                "requests": model.request_count if model else 0,
                "avg_latency": round(model.average_latency, 2) if model else 0
            }
            await websocket.send_json(metrics)
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass

def load_model(model_path: str):
    """Load a model from the given path."""
    global model
    try:
        from ..server.loader import ModelLoader
        
        # First load the raw model
        raw_model = ModelLoader.load(model_path)
        
        # Create model wrapper
        model = ModelWrapper.from_model(raw_model)
        
        # Get and print model info
        model_info = get_model_info()
        print(f"Model loaded successfully: {model_info}")
        return model_info
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

def start_server(model_path: str, host: str = "127.0.0.1", port: int = 8000, ui: bool = True):
    """Start the server with the given model."""
    import uvicorn
    global enable_ui, model
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Enable UI if requested
    enable_ui = ui
    print(f"UI {'enabled' if ui else 'disabled'}")
    
    # Load model first
    try:
        model_info = load_model(model_path)
        print(f"Starting server with model: {model_info}")
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        raise
    
    # Configure and start uvicorn
    config = uvicorn.Config(
        app=app,
        host=host,
        port=port,
        log_level="info",
        loop="asyncio"
    )
    server = uvicorn.Server(config)
    server.install_signal_handlers = lambda: None  # Disable uvicorn's signal handlers
    server.run()