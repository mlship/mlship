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
from ..cloud.gcp import create_instance, delete_instance, get_instance_ip, get_available_zones_with_gpus
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
    
    # Create models directory if it doesn't exist
    models_dir = os.path.join(mlship_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Copy the model file to the models directory
    model_filename = os.path.basename(model_path)
    new_model_path = os.path.join(models_dir, model_filename)
    
    # Only copy if the file doesn't exist or is different
    if not os.path.exists(new_model_path) or not files_are_identical(model_path, new_model_path):
        shutil.copy2(model_path, new_model_path)
    
    config = {"model_path": new_model_path}
    config_file = os.path.join(mlship_dir, "config.json")
    
    with open(config_file, "w") as f:
        json.dump(config, f)
    print(f"Saved model path to config: {new_model_path}")

def files_are_identical(file1: str, file2: str) -> bool:
    """Check if two files are identical by comparing their contents."""
    try:
        with open(file1, 'rb') as f1, open(file2, 'rb') as f2:
            return f1.read() == f2.read()
    except Exception:
        return False

def get_startup_script(model_path: str) -> str:
    """Generate startup script for the VM instance."""
    return f"""#!/bin/bash
# Update and install basic dependencies
apt-get update
apt-get install -y python3-pip git wget

# Install CUDA dependencies
wget https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/cuda-keyring_1.0-1_all.deb
dpkg -i cuda-keyring_1.0-1_all.deb
apt-get update
apt-get install -y cuda-toolkit-11-8

# Set CUDA environment variables
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> /etc/profile.d/cuda.sh
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> /etc/profile.d/cuda.sh
source /etc/profile.d/cuda.sh

# Clone MLship repository
git clone https://github.com/yourusername/mlship.git
cd mlship

# Install Python dependencies
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install -r requirements.txt

# Create model directory and copy model
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
                # Try to find model in models directory
                models_dir = os.path.join(os.path.expanduser("~"), ".mlship", "models")
                if os.path.exists(models_dir):
                    model_files = [f for f in os.listdir(models_dir) if is_valid_model_file(f)]
                    if model_files:
                        model_files.sort(key=lambda x: os.path.getmtime(os.path.join(models_dir, x)), reverse=True)
                        model_path = os.path.join(models_dir, model_files[0])
                        config["model_path"] = model_path
                        with open(config_file, "w") as f:
                            json.dump(config, f)
                        print(f"Updated config with found model: {model_path}")
                    else:
                        print(f"No valid model files found in: {models_dir}")
                        raise HTTPException(status_code=400, detail="Model file not found")
                else:
                    print(f"Models directory not found: {models_dir}")
                    raise HTTPException(status_code=400, detail="Model file not found")
            
            if not is_valid_model_file(model_path):
                print(f"Invalid model file type: {model_path}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid model file type. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}"
                )
            
            # Create deployment record
            deployment_id = len(deployments) + 1
            instance_name = f"mlship-{deployment_id}"
            
            # Get cloud configuration
            cloud_config = get_cloud_config()
            
            # Map GPU types to machine types
            machine_type_map = {
                GpuType.nvidia_t4: "n1-standard-4",
                GpuType.nvidia_a100: "a2-highgpu-1g",
                GpuType.nvidia_a100_multi: "a2-highgpu-4g"
            }
            
            machine_type = machine_type_map[request.gpu_type]
            
            # Create the startup script
            startup_script = get_startup_script(model_path)
            
            # Create GCP instance
            if request.cloud_provider == CloudProvider.gcp:
                try:
                    instance = create_instance(
                        project_id=cloud_config["gcp"]["project_id"],
                        zone=cloud_config["gcp"]["zone"],
                        instance_name=instance_name,
                        machine_type=machine_type,
                        startup_script=startup_script,
                        credentials_base64=cloud_config["gcp"].get("credentials_base64")
                    )
                    
                    # Get instance IP
                    instance_ip = get_instance_ip(instance)
                    endpoint = f"http://{instance_ip}/predict"
                except Exception as e:
                    print(f"Error creating GCP instance: {str(e)}")
                    raise HTTPException(status_code=500, detail=f"Failed to create GCP instance: {str(e)}")
            else:
                raise HTTPException(status_code=400, detail="Only GCP deployments are currently supported")
            
            deployment = {
                "id": deployment_id,
                "model_path": model_path,
                "filename": os.path.basename(model_path),
                "status": "running",
                "gpu_type": request.gpu_type,
                "cloud_provider": request.cloud_provider,
                "auto_scaling": request.auto_scaling,
                "endpoint": endpoint,
                "instance_name": instance_name,
                "instance_ip": instance_ip,
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

# Add inference endpoint
@app.post("/api/deployments/{deployment_id}/predict")
async def predict(deployment_id: int, data: dict = Body(...)):
    """Make a prediction using a deployed model."""
    if deployment_id not in deployments:
        raise HTTPException(status_code=404, detail="Deployment not found")
    
    deployment = deployments[deployment_id]
    if deployment["status"] != "running":
        raise HTTPException(status_code=400, detail="Deployment is not running")
    
    import requests
    try:
        response = requests.post(deployment["endpoint"], json=data)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error making prediction: {str(e)}")

# Add deployment status endpoint
@app.get("/api/deployments/{deployment_id}/status")
async def get_deployment_status(deployment_id: int):
    """Get the status of a deployment."""
    if deployment_id not in deployments:
        raise HTTPException(status_code=404, detail="Deployment not found")
    
    deployment = deployments[deployment_id]
    
    if deployment["cloud_provider"] == CloudProvider.gcp:
        # Get cloud configuration
        cloud_config = get_cloud_config()
        
        # Check instance status
        instance = get_instance(
            project_id=cloud_config["gcp"]["project_id"],
            zone=cloud_config["gcp"]["zone"],
            instance_name=deployment["instance_name"],
            credentials_file=cloud_config["gcp"].get("credentials_file")
        )
        
        if instance:
            deployment["status"] = "running"
            deployment["instance_ip"] = get_instance_ip(instance)
            deployment["endpoint"] = f"http://{deployment['instance_ip']}/predict"
        else:
            deployment["status"] = "stopped"
    
    return deployment

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
            print(f"Found model path in config: {model_path}")
            
            if model_path and os.path.exists(model_path):
                print(f"Found model at: {model_path}")
                return {
                    "model_path": model_path,
                    "filename": os.path.basename(model_path),
                    "size": os.path.getsize(model_path)
                }
            else:
                print(f"Model not found at: {model_path}")
                # Try to find model in models directory
                models_dir = os.path.join(os.path.expanduser("~"), ".mlship", "models")
                if os.path.exists(models_dir):
                    model_files = [f for f in os.listdir(models_dir) if is_valid_model_file(f)]
                    if model_files:
                        # Use the most recently modified file
                        model_files.sort(key=lambda x: os.path.getmtime(os.path.join(models_dir, x)), reverse=True)
                        model_path = os.path.join(models_dir, model_files[0])
                        # Update config with found model
                        config["model_path"] = model_path
                        with open(config_file, "w") as f:
                            json.dump(config, f)
                        print(f"Updated config with found model: {model_path}")
                        return {
                            "model_path": model_path,
                            "filename": os.path.basename(model_path),
                            "size": os.path.getsize(model_path)
                        }
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
            with open(config_file) as f:
                config = json.load(f)
                model_path = config.get("model_path")
                if model_path and os.path.exists(model_path):
                    try:
                        os.remove(model_path)
                    except Exception as e:
                        print(f"Error removing model file: {str(e)}")
            
            # Clear the config
            os.remove(config_file)
            
            # Clean up models directory
            models_dir = os.path.join(os.path.expanduser("~"), ".mlship", "models")
            if os.path.exists(models_dir):
                try:
                    shutil.rmtree(models_dir)
                except Exception as e:
                    print(f"Error removing models directory: {str(e)}")
            
            return {"status": "success"}
            
        print("No config file found")
        return {"status": "no_model_found"}
    except Exception as e:
        print(f"Error in remove_model: {str(e)}")
        return {"status": "error", "detail": str(e)} 

@app.get("/api/gpu-availability")
async def get_gpu_availability():
    """Get GPU availability across all zones."""
    try:
        # Get cloud configuration
        cloud_config = get_cloud_config()
        
        # Get available zones with GPUs
        available_zones = get_available_zones_with_gpus(
            credentials_base64=cloud_config["gcp"].get("credentials_base64")
        )
        
        return available_zones
    except Exception as e:
        print(f"Error getting GPU availability: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 