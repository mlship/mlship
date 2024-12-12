from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
import os
import json
from pathlib import Path
import shutil
import webbrowser
import time
from typing import Optional

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],  # Next.js development servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store deployments in memory (in production, use a database)
deployments = {}

def save_model_path(model_path: str):
    """Save the model path to the config file."""
    if not os.path.exists(model_path):
        raise ValueError(f"Model file not found: {model_path}")
        
    mlship_dir = os.path.join(os.path.expanduser("~"), ".mlship")
    os.makedirs(mlship_dir, exist_ok=True)
    
    config = {"model_path": os.path.abspath(model_path)}
    config_file = os.path.join(mlship_dir, "config.json")
    
    with open(config_file, "w") as f:
        json.dump(config, f)

@app.get("/")
async def root():
    """Redirect to the Next.js frontend."""
    return RedirectResponse(url="http://localhost:3000")

@app.get("/cloud")
async def cloud_redirect():
    """Redirect to the Next.js frontend."""
    return RedirectResponse(url="http://localhost:3000")

@app.get("/api/model")
async def get_model_info():
    """Get information about the model from the CLI command."""
    config_file = os.path.join(os.path.expanduser("~"), ".mlship", "config.json")
    try:
        with open(config_file) as f:
            config = json.load(f)
            model_path = config.get("model_path")
            if model_path and os.path.exists(model_path):
                return {
                    "model_path": model_path,
                    "filename": os.path.basename(model_path),
                    "size": os.path.getsize(model_path)
                }
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return {"model_path": None, "filename": None, "size": None}

@app.post("/api/deploy")
async def deploy_model(
    gpu_type: str,
    auto_scaling: bool = False,
    monitoring: bool = False,
    model_file: Optional[UploadFile] = File(None)
):
    """Deploy a model to the cloud."""
    try:
        # Create .mlship directory if it doesn't exist
        mlship_dir = os.path.join(os.path.expanduser("~"), ".mlship")
        os.makedirs(mlship_dir, exist_ok=True)
        
        # If model file is uploaded through the web interface
        if model_file:
            # Save the uploaded file
            upload_dir = os.path.join(mlship_dir, "uploads")
            os.makedirs(upload_dir, exist_ok=True)
            file_path = os.path.join(upload_dir, model_file.filename)
            
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(model_file.file, buffer)
            
            model_path = file_path
            save_model_path(model_path)
        else:
            # Use the model path from the CLI command
            config_file = os.path.join(mlship_dir, "config.json")
            try:
                with open(config_file) as f:
                    config = json.load(f)
                    model_path = config.get("model_path")
                    if not model_path or not os.path.exists(model_path):
                        raise HTTPException(status_code=400, detail="No valid model file provided")
            except (FileNotFoundError, json.JSONDecodeError):
                raise HTTPException(status_code=400, detail="No model file provided")
        
        # Create deployment record
        deployment_id = len(deployments) + 1
        deployments[deployment_id] = {
            "id": deployment_id,
            "model_path": model_path,
            "filename": os.path.basename(model_path),
            "status": "running",
            "gpu_type": gpu_type,
            "auto_scaling": auto_scaling,
            "monitoring": monitoring,
            "metrics": {
                "memory_usage": "0GB",
                "requests_per_hour": 0
            }
        }
        
        return deployments[deployment_id]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/deployments")
async def list_deployments():
    """List all deployments."""
    return list(deployments.values())

@app.post("/api/deployments/{deployment_id}/stop")
async def stop_deployment(deployment_id: int):
    """Stop a deployment."""
    if deployment_id not in deployments:
        raise HTTPException(status_code=404, detail="Deployment not found")
    
    deployments[deployment_id]["status"] = "stopped"
    return deployments[deployment_id]

def start_cloud_server(host: str = "0.0.0.0", port: int = 8000):
    """Start the cloud deployment server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)

def deploy_from_cli(model_path: str):
    """Handle deployment from CLI command."""
    try:
        # Save the model path
        save_model_path(model_path)
        
        # Open the browser
        webbrowser.open("http://localhost:3000")
        
        # Start the server
        start_cloud_server()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 