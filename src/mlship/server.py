from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import json
import asyncio
from pathlib import Path

from .models.wrapper import ModelWrapper
from .server.loader import ModelLoader

app = FastAPI()

# Get the directory containing this file
current_dir = Path(__file__).parent.resolve()

# Global model instance
model = None

# Global UI state
enable_ui = True

def get_model_info():
    """Get information about the loaded model."""
    if model is None:
        return {}
    return model.get_model_info()

def load_model(model_path: str):
    """Load a model from the given path."""
    global model
    try:
        model = ModelLoader.load(model_path)
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

def start_server(model_path: str, host: str = "localhost", port: int = 8000, ui: bool = True):
    """Start the server with the given model."""
    import uvicorn
    global enable_ui
    
    # Set UI state
    enable_ui = ui
    
    # Mount static files and templates if UI is enabled
    if enable_ui:
        app.mount("/static", StaticFiles(directory=str(current_dir / "ui" / "static")), name="static")
        templates = Jinja2Templates(directory=str(current_dir / "ui" / "templates"))
        
        @app.get("/", response_class=HTMLResponse)
        async def root(request: Request):
            """Redirect to UI."""
            return templates.TemplateResponse("index.html", {
                "request": request,
                "model_info": json.dumps(get_model_info())
            })

        @app.get("/ui", response_class=HTMLResponse)
        async def ui(request: Request):
            """Serve the UI."""
            return templates.TemplateResponse("index.html", {
                "request": request,
                "model_info": json.dumps(get_model_info())
            })
    
    # Load model
    if not load_model(model_path):
        raise ValueError("Failed to load model")
    
    # API endpoints
    @app.get("/api/model/info")
    async def get_model_info_api():
        """Get information about the loaded model."""
        if not model:
            raise HTTPException(status_code=404, detail="No model loaded")
        return get_model_info()

    @app.post("/api/model/predict")
    async def predict(request: Request):
        """Make a prediction with the model."""
        if not model:
            raise HTTPException(status_code=404, detail="No model loaded")
        
        try:
            data = await request.json()
            inputs = data.get("inputs")
            if inputs is None:
                raise HTTPException(status_code=400, detail="No inputs provided")
            
            prediction = model.predict(inputs)
            return {"prediction": prediction}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

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
    
    # Start server
    uvicorn.run(app, host=host, port=port) 