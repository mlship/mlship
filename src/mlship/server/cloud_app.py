from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
import os
import json
from pathlib import Path
import shutil
import webbrowser
from enum import Enum
from typing import Optional
from pydantic import BaseModel
from ..cloud.gcp import create_instance, delete_instance, get_instance_ip
from ..config.cloud_config import get_cloud_config

class CloudProvider(str, Enum):
    aws = "aws"
    gcp = "gcp"
    azure = "azure"

class GpuType(str, Enum):
    nvidia_t4 = "nvidia-t4"
    nvidia_a100 = "nvidia-a100"
    nvidia_a100_multi = "nvidia-a100-multi"

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8000", "https://mlship-cloud.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store deployments in memory (in production, use a database)
deployments = {}

FRONTEND_URL = "https://mlship-cloud.vercel.app"

# Allowed ML model file extensions
ALLOWED_EXTENSIONS = {
    '.pt',      # PyTorch
    '.pth',     # PyTorch
    '.h5',      # Keras/TensorFlow
    '.pb',      # TensorFlow (protobuf)
    '.onnx',    # ONNX
    '.pkl',     # Scikit-learn
    '.joblib',  # Scikit-learn
    '.model',   # XGBoost
    '.bin',     # LightGBM
    '.pmml',    # PMML format
    '.mar',     # TorchServe
    '.savedmodel'  # TensorFlow SavedModel
}

def is_valid_model_file(filename: str) -> bool:
    """Check if the file has an allowed ML model extension."""
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

def save_model_path(model_path: str):
    """Save the model path to the config file."""
    if not os.path.exists(model_path):
        raise ValueError(f"Model file not found: {model_path}")
    
    if not is_valid_model_file(model_path):
        raise ValueError(f"Invalid model file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}")
        
    mlship_dir = os.path.join(os.path.expanduser("~"), ".mlship")
    os.makedirs(mlship_dir, exist_ok=True)
    
    config = {"model_path": os.path.abspath(model_path)}
    config_file = os.path.join(mlship_dir, "config.json")
    
    with open(config_file, "w") as f:
        json.dump(config, f)
    print(f"Saved model path to config: {model_path}")

def get_startup_script(model_path: str) -> str:
    """Generate startup script for the VM instance."""
    return f"""#!/bin/bash
# Install required packages
apt-get update
apt-get install -y python3-pip git

# Clone MLship repository
git clone https://github.com/yourusername/mlship.git
cd mlship

# Install dependencies
pip3 install -r requirements.txt

# Copy model file
mkdir -p /model
cp {model_path} /model/

# Start the server
python3 -m mlship serve --port 80
"""

class DeploymentRequest(BaseModel):
    gpu_type: GpuType
    cloud_provider: CloudProvider
    auto_scaling: bool = False

@app.post("/api/deploy")
async def deploy_model(request: DeploymentRequest):
    """Deploy a model to the cloud."""
    print("POST /api/deploy called")
    try:
        config_file = os.path.join(os.path.expanduser("~"), ".mlship", "config.json")
        print(f"Looking for config file at: {config_file}")
        
        if not os.path.exists(config_file):
            print("No config file found")
            raise HTTPException(status_code=400, detail="No model file provided. Please upload a model first.")
            
        with open(config_file) as f:
            config = json.load(f)
            model_path = config.get("model_path")
            print(f"Config loaded, model_path: {model_path}")
            
            if not model_path or not os.path.exists(model_path):
                print(f"Model not found at path: {model_path}")
                raise HTTPException(status_code=400, detail="Model file not found")
            
            if not is_valid_model_file(model_path):
                print(f"Invalid model file type: {model_path}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid model file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
                )
            
            # Create deployment record
            deployment_id = len(deployments) + 1
            deployment = {
                "id": deployment_id,
                "model_path": model_path,
                "filename": os.path.basename(model_path),
                "status": "running",
                "gpu_type": request.gpu_type,
                "cloud_provider": request.cloud_provider,
                "auto_scaling": request.auto_scaling,
                "endpoint": f"http://localhost:8000/api/model/predict",
                "instance_name": f"mlship-{deployment_id}",
                "metrics": {
                    "memory_usage": "0GB",
                    "requests_per_hour": 0
                }
            }
            
            deployments[deployment_id] = deployment
            print(f"Created deployment: {deployment}")
            return deployment
            
    except Exception as e:
        print(f"Error in deploy_model: {str(e)}")
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
    
    deployment = deployments[deployment_id]
    
    if deployment["cloud_provider"] == CloudProvider.gcp:
        # Get cloud configuration
        cloud_config = get_cloud_config()
        
        # Delete the GCP instance
        delete_instance(
            project_id=cloud_config["gcp"]["project_id"],
            zone=cloud_config["gcp"]["zone"],
            instance_name=deployment["instance_name"],
            credentials_file=cloud_config["gcp"]["credentials_file"]
        )
    
    deployment["status"] = "stopped"
    return deployment

def deploy_from_cli(model_path: str):
    """Handle deployment from CLI command."""
    try:
        # Convert to absolute path if relative
        model_path = os.path.abspath(model_path)
        print(f"Deploying model from: {model_path}")
        
        # Save the model path
        save_model_path(model_path)
        print("Model path saved to config")
        
        # Verify the config was saved
        config_file = os.path.join(os.path.expanduser("~"), ".mlship", "config.json")
        if os.path.exists(config_file):
            with open(config_file) as f:
                config = json.load(f)
                print(f"Config loaded: {config}")
        
        # Open the browser to the frontend
        print(f"Opening frontend at: {FRONTEND_URL}")
        webbrowser.open(FRONTEND_URL)
        
        # Start the server
        print("Starting server...")
        start_cloud_server()
    except Exception as e:
        print(f"Error in deploy_from_cli: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def start_cloud_server(host: str = "0.0.0.0", port: int = 8000):
    """Start the cloud deployment server."""
    import uvicorn
    print(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="debug")

@app.get("/api/model")
async def get_model_info():
    """Get information about the model from the CLI command."""
    print("GET /api/model called")
    try:
        config_file = os.path.join(os.path.expanduser("~"), ".mlship", "config.json")
        print(f"Looking for config file at: {config_file}")
        
        if not os.path.exists(config_file):
            print("Config file not found")
            return {"model_path": None, "filename": None, "size": None}
            
        with open(config_file) as f:
            config = json.load(f)
            model_path = config.get("model_path")
            print(f"Config loaded, model_path: {model_path}")
            
            if model_path and os.path.exists(model_path):
                print(f"Found model at: {model_path}")
                return {
                    "model_path": model_path,
                    "filename": os.path.basename(model_path),
                    "size": os.path.getsize(model_path)
                }
            else:
                print(f"Model not found at path: {model_path}")
                return {"model_path": None, "filename": None, "size": None}
    except Exception as e:
        print(f"Error in get_model_info: {str(e)}")
        return {"model_path": None, "filename": None, "size": None}

@app.delete("/api/model")
async def remove_model():
    """Remove the model file and clear the config."""
    print("DELETE /api/model called")
    try:
        config_file = os.path.join(os.path.expanduser("~"), ".mlship", "config.json")
        print(f"Looking for config file at: {config_file}")
        
        if os.path.exists(config_file):
            os.remove(config_file)
            print("Config file removed successfully")
            return {"status": "success"}
            
        print("No config file found")
        return {"status": "no_model_found"}
    except Exception as e:
        print(f"Error in remove_model: {str(e)}")
        return {"status": "error", "detail": str(e)} 