from .app import app
from .loader import ModelLoader
from ..models.wrapper import ModelWrapper
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path
import uvicorn

def start_server(model_path: str, host: str = "localhost", port: int = 8000, ui: bool = True):
    """Start the server with the given model."""
    # Get the package directory
    package_dir = Path(__file__).parent.parent
    
    # Mount static files if UI is enabled
    if ui:
        app.mount("/static", StaticFiles(directory=str(package_dir / "ui" / "static")), name="static")
        templates = Jinja2Templates(directory=str(package_dir / "ui" / "templates"))
        
        # Import UI routes
        from .app import setup_ui_routes
        setup_ui_routes(app, templates)
    
    # Load model
    try:
        model = ModelLoader.load(model_path)
        if not model:
            raise ValueError("Failed to load model")
            
        # Import and setup API routes
        from .app import setup_api_routes
        setup_api_routes(app, model)
        
        # Start server
        uvicorn.run(app, host=host, port=port)
    except Exception as e:
        raise ValueError(f"Failed to load model: {str(e)}")

__all__ = ['app', 'start_server', 'ModelLoader']