from google.cloud import compute_v1
import os
import time
from typing import Dict, Optional

def create_instance(
    project_id: str,
    zone: str,
    instance_name: str,
    machine_type: str,
    startup_script: str,
    credentials_file: Optional[str] = None
) -> compute_v1.Instance:
    """Create a GCP Compute Engine instance."""
    
    if credentials_file:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_file
    
    instance_client = compute_v1.InstancesClient()
    
    # Configure the machine
    machine_type = f"zones/{zone}/machineTypes/{machine_type}"
    instance_config = {
        "name": instance_name,
        "machine_type": machine_type,
        
        # Specify the boot disk
        "disks": [
            {
                "boot": True,
                "auto_delete": True,
                "initialize_params": {
                    "source_image": "projects/deeplearning-platform-release/global/images/family/common-gpu",
                    "disk_size_gb": 50
                }
            }
        ],
        
        # Specify a network interface
        "network_interfaces": [
            {
                "network": "global/networks/default",
                "access_configs": [{"name": "External NAT"}]
            }
        ],
        
        # Allow the instance to access cloud APIs
        "service_accounts": [
            {
                "email": "default",
                "scopes": ["https://www.googleapis.com/auth/cloud-platform"]
            }
        ],
        
        # Metadata for startup script
        "metadata": {
            "items": [
                {
                    "key": "startup-script",
                    "value": startup_script
                }
            ]
        },
        
        # Enable GPU
        "guest_accelerators": [
            {
                "accelerator_type": f"zones/{zone}/acceleratorTypes/nvidia-tesla-t4",
                "accelerator_count": 1
            }
        ],
        
        # On-host maintenance must be disabled for GPU instances
        "scheduling": {
            "on_host_maintenance": "TERMINATE"
        }
    }
    
    # Create the instance
    operation = instance_client.insert(
        project=project_id,
        zone=zone,
        instance_resource=instance_config
    )
    
    # Wait for the operation to complete
    while not operation.done():
        time.sleep(5)
    
    return instance_client.get(
        project=project_id,
        zone=zone,
        instance=instance_name
    )

def delete_instance(
    project_id: str,
    zone: str,
    instance_name: str,
    credentials_file: Optional[str] = None
) -> None:
    """Delete a GCP Compute Engine instance."""
    
    if credentials_file:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_file
    
    instance_client = compute_v1.InstancesClient()
    
    operation = instance_client.delete(
        project=project_id,
        zone=zone,
        instance=instance_name
    )
    
    # Wait for the operation to complete
    while not operation.done():
        time.sleep(5)

def get_instance(
    project_id: str,
    zone: str,
    instance_name: str,
    credentials_file: Optional[str] = None
) -> Optional[compute_v1.Instance]:
    """Get information about a GCP Compute Engine instance."""
    
    if credentials_file:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_file
    
    instance_client = compute_v1.InstancesClient()
    
    try:
        return instance_client.get(
            project=project_id,
            zone=zone,
            instance=instance_name
        )
    except Exception:
        return None

def get_instance_ip(instance: compute_v1.Instance) -> str:
    """Get the external IP address of an instance."""
    for network_interface in instance.network_interfaces:
        for access_config in network_interface.access_configs:
            if access_config.nat_ip:
                return access_config.nat_ip
    return "" 