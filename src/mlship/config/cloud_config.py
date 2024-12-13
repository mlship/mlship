"""
Cloud configuration settings for MLship.
Configuration is loaded from environment variables.
"""

import os
import json
from typing import Dict
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = os.path.join(Path(__file__).parent.parent.parent.parent, '.env')
load_dotenv(env_path)

def get_instance_types(provider: str) -> Dict[str, str]:
    """Get instance type mappings for a cloud provider."""
    env_var = f"{provider.upper()}_INSTANCE_TYPES"
    try:
        return json.loads(os.getenv(env_var, "{}"))
    except json.JSONDecodeError:
        return {}

def get_cloud_config() -> Dict:
    """Get cloud configuration from environment variables."""
    return {
        "aws": {
            "access_key_id": os.getenv("AWS_ACCESS_KEY_ID", ""),
            "secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY", ""),
            "region": os.getenv("AWS_REGION", "us-west-2"),
            "instance_types": get_instance_types("aws")
        },
        "gcp": {
            "project_id": os.getenv("GCP_PROJECT_ID", ""),
            "credentials_file": os.getenv("GCP_CREDENTIALS_FILE", ""),
            "zone": os.getenv("GCP_ZONE", "us-central1-a"),
            "instance_types": get_instance_types("gcp")
        },
        "azure": {
            "subscription_id": os.getenv("AZURE_SUBSCRIPTION_ID", ""),
            "resource_group": os.getenv("AZURE_RESOURCE_GROUP", ""),
            "location": os.getenv("AZURE_LOCATION", "eastus"),
            "instance_types": get_instance_types("azure")
        },
        "frontend_url": os.getenv("FRONTEND_URL", "https://mlship-cloud.vercel.app")
    }

def get_provider_config(provider: str) -> Dict:
    """Get configuration details for a specific cloud provider."""
    config = get_cloud_config()
    if provider not in ["aws", "gcp", "azure"]:
        raise ValueError(f"Invalid provider: {provider}. Must be one of: aws, gcp, azure")
    return config[provider]

def get_all_provider_configs() -> Dict:
    """Get configuration details for all cloud providers."""
    config = get_cloud_config()
    return {
        "aws": config["aws"],
        "gcp": config["gcp"],
        "azure": config["azure"],
        "instance_types": {
            "aws": config["aws"]["instance_types"],
            "gcp": config["gcp"]["instance_types"],
            "azure": config["azure"]["instance_types"]
        }
    }

def get_instance_type(provider: str, gpu_type: str) -> str:
    """Get the instance type for a given provider and GPU type."""
    config = get_cloud_config()
    instance_types = config[provider]["instance_types"]
    return instance_types.get(gpu_type, "")

def get_frontend_url() -> str:
    """Get the frontend URL."""
    return get_cloud_config()["frontend_url"]

def get_active_provider() -> str:
    """Get the active cloud provider."""
    return get_cloud_config()["provider"] 