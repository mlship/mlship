import os
import json
import psutil
import subprocess
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ..utils.constants import PID_FILE, METRICS_FILE

router = APIRouter()

class DeployRequest(BaseModel):
    model_path: str

@router.get("/api/status")
async def get_status():
    """Get server status and metrics"""
    try:
        if not os.path.exists(PID_FILE):
            raise HTTPException(status_code=404, detail="Server is not running")

        with open(PID_FILE) as f:
            pid = int(f.read().strip())

        try:
            process = psutil.Process(pid)
            if process.is_running():
                status = {
                    'pid': pid,
                    'memory_mb': process.memory_info().rss / 1024 / 1024,
                    'cpu_percent': process.cpu_percent(),
                }

                # Add metrics if available
                if os.path.exists(METRICS_FILE):
                    try:
                        with open(METRICS_FILE) as f:
                            metrics = json.load(f)
                            status['metrics'] = metrics
                    except json.JSONDecodeError:
                        pass

                return status
            else:
                raise HTTPException(status_code=404, detail="Server is not running")
        except psutil.NoSuchProcess:
            raise HTTPException(status_code=404, detail="Server is not running")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/api/deploy")
async def deploy_model(request: DeployRequest):
    """Deploy a model using the CLI"""
    try:
        model_path = request.model_path
        
        # Convert to absolute path if relative
        if not os.path.isabs(model_path):
            model_path = os.path.abspath(model_path)

        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Model file not found")

        # Run mlship deploy command
        result = subprocess.run(
            ['mlship', 'deploy', model_path, '--daemon'],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=result.stderr)

        return {"message": "Model deployed successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 