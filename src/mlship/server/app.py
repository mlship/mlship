from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import json
import asyncio
import uvicorn
from pathlib import Path
from .loader import ModelLoader
from .dashboard_routes import router as dashboard_router

app = FastAPI()

def setup_ui_routes(app, templates):
    """Set up UI routes."""
    @app.get("/", response_class=HTMLResponse)
    async def root(request: Request):
        """Serve the UI."""
        return templates.TemplateResponse("index.html", {
            "request": request,
            "model_info": json.dumps(get_model_info(app.state.model))
        })

    @app.get("/ui", response_class=HTMLResponse)
    async def ui(request: Request):
        """Serve the UI."""
        return templates.TemplateResponse("index.html", {
            "request": request,
            "model_info": json.dumps(get_model_info(app.state.model))
        })

    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard_ui(request: Request):
        """Serve the dashboard UI."""
        return templates.TemplateResponse("dashboard.html", {
            "request": request
        })

def setup_api_routes(app, model):
    """Set up API routes."""
    # Store model in app state
    app.state.model = model
    
    @app.get("/api/model/info")
    async def get_model_info_api():
        """Get information about the loaded model."""
        return get_model_info(app.state.model)

    @app.post("/api/model/predict")
    async def predict(request: Request):
        """Make a prediction with the model."""
        try:
            data = await request.json()
            inputs = data.get("inputs")
            if inputs is None:
                raise HTTPException(status_code=400, detail="No inputs provided")
            
            # Convert inputs to list of lists if it's not already
            if not isinstance(inputs[0], list):
                inputs = [inputs]
                
            prediction = app.state.model.predict(inputs)
            return {"prediction": prediction[0] if len(prediction) == 1 else prediction}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.websocket("/ws/metrics")
    async def websocket_metrics(websocket: WebSocket):
        """WebSocket endpoint for real-time metrics."""
        await websocket.accept()
        try:
            while True:
                metrics = {
                    "requests": app.state.model.request_count if hasattr(app.state.model, 'request_count') else 0,
                    "avg_latency": app.state.model.average_latency if hasattr(app.state.model, 'average_latency') else 0
                }
                await websocket.send_json(metrics)
                await asyncio.sleep(1)
        except WebSocketDisconnect:
            pass

def get_model_info(model):
    """Get information about the loaded model."""
    if model is None:
        return {}
    try:
        return model.get_info()
    except Exception as e:
        return {"error": str(e)}

def start_server(model_path: str, host: str = "localhost", port: int = 8000, ui: bool = True):
    """Start the server with the given model."""
    # Get the package directory
    package_dir = Path(__file__).parent.parent
    
    # Mount static files if UI is enabled
    if ui:
        app.mount("/static", StaticFiles(directory=str(package_dir / "ui" / "static")), name="static")
        templates = Jinja2Templates(directory=str(package_dir / "ui" / "templates"))
        setup_ui_routes(app, templates)
    
    # Register dashboard routes
    app.include_router(dashboard_router)
    
    # Load model
    try:
        model = ModelLoader.load(model_path)
        if not model:
            raise ValueError("Failed to load model")
            
        # Setup API routes
        setup_api_routes(app, model)
        
        # Start server
        uvicorn.run(app, host=host, port=port)
    except Exception as e:
        raise ValueError(f"Failed to load model: {str(e)}")

__all__ = ['app', 'start_server']