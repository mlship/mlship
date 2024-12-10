from fastapi import FastAPI, WebSocket, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import numpy as np
import json
import time
import asyncio
import logging
import uvicorn
import signal
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from ..utils.constants import METRICS_FILE
from ..utils.daemon import cleanup_files

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PredictRequest(BaseModel):
    inputs: List[List[float]]

class ModelServer:
    def __init__(self, model):
        self.model = model
        self.app = FastAPI(debug=True)
        self.active_connections: List[WebSocket] = []
        self.request_count = 0
        self.total_latency = 0
        self.start_time = time.time()
        self.should_exit = False
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # Set up static files and templates
        self.setup_static_files()
        self.setup_routes()
        
    def setup_static_files(self):
        """Set up static files and templates with proper paths"""
        try:
            # Get the package root directory
            package_dir = Path(__file__).parent.parent
            
            # Set up paths
            static_dir = package_dir / "ui" / "static"
            template_dir = package_dir / "ui" / "templates"
            
            logger.info(f"Static directory: {static_dir} (exists: {static_dir.exists()})")
            logger.info(f"Template directory: {template_dir} (exists: {template_dir.exists()})")
            
            # List contents of directories
            if static_dir.exists():
                logger.info(f"Static directory contents: {list(static_dir.glob('**/*'))}")
            if template_dir.exists():
                logger.info(f"Template directory contents: {list(template_dir.glob('**/*'))}")
            
            # Mount static files
            self.app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
            self.templates = Jinja2Templates(directory=str(template_dir))
            
        except Exception as e:
            logger.error(f"Error setting up static files: {str(e)}")
            raise
        
    def setup_routes(self):
        @self.app.get("/")
        async def redirect_to_ui():
            from fastapi.responses import RedirectResponse
            return RedirectResponse(url="/ui")
            
        @self.app.get("/ui", response_class=HTMLResponse)
        async def ui(request: Request):
            try:
                logger.info(f"Serving UI template for request: {request.url}")
                model_info = self.get_model_info()
                logger.info(f"Model info: {model_info}")
                
                return self.templates.TemplateResponse(
                    "index.html",
                    {
                        "request": request,
                        "model_info": model_info,
                        "base_url": str(request.base_url).rstrip('/')
                    }
                )
            except Exception as e:
                logger.error(f"Error serving UI template: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e))
            
        @self.app.get("/api/model-info")
        def model_info():
            return self.get_model_info()
            
        @self.app.post("/api/predict")
        async def predict(request: PredictRequest):
            try:
                start_time = time.time()
                inputs = np.array(request.inputs)
                predictions = self.model.predict(inputs)
                
                # Update metrics
                self.request_count += 1
                self.total_latency += (time.time() - start_time) * 1000
                self.update_metrics()
                
                return {"predictions": predictions.tolist()}
            except Exception as e:
                logger.error(f"Prediction error: {str(e)}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/health")
        def health():
            return {"status": "healthy"}
            
        @self.app.websocket("/ws/metrics")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self.active_connections.append(websocket)
            try:
                while True:
                    # Send metrics updates every second
                    metrics = {
                        'start_time': self.start_time,
                        'requests': self.request_count,
                        'avg_latency': round(self.total_latency / max(1, self.request_count), 2)
                    }
                    await websocket.send_json(metrics)
                    await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"WebSocket error: {str(e)}")
            finally:
                self.active_connections.remove(websocket)
        
    def get_model_info(self) -> Dict[str, Any]:
        """Extract model information for UI"""
        model_info = {
            "type": type(self.model).__name__,
            "params": {},
            "features": [],
            "n_features": None
        }
        
        # Get model parameters
        if hasattr(self.model, "get_params"):
            model_info["params"] = self.model.get_params()
            
        # Try to get feature names
        if hasattr(self.model, "feature_names_in_"):
            model_info["features"] = self.model.feature_names_in_.tolist()
        
        # Get input shape if available
        if hasattr(self.model, "n_features_in_"):
            model_info["n_features"] = self.model.n_features_in_
            
        return model_info
        
    def update_metrics(self):
        """Update metrics file"""
        metrics = {
            'start_time': self.start_time,
            'requests': self.request_count,
            'avg_latency': round(self.total_latency / max(1, self.request_count), 2)
        }
        with open(METRICS_FILE, 'w') as f:
            json.dump(metrics, f)
            
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info("Received shutdown signal")
        self.should_exit = True
        cleanup_files()
        sys.exit(0)
            
    def serve(self, host="0.0.0.0", port=8000):
        """Start the server with proper signal handling"""
        logger.info(f"Starting server on {host}:{port}")
        
        # Configure uvicorn with proper signal handling
        config = uvicorn.Config(
            app=self.app,
            host=host,
            port=port,
            log_level="debug",
            loop="asyncio",
            reload=False,
            workers=1
        )
        
        server = uvicorn.Server(config)
        server.install_signal_handlers = lambda: None  # Disable uvicorn's signal handlers
        
        try:
            server.run()
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
            cleanup_files()
            sys.exit(0)
        except Exception as e:
            logger.error(f"Server error: {str(e)}")
            cleanup_files()
            sys.exit(1)