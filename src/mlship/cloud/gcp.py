from google.cloud import compute_v1
from google.oauth2 import service_account
from googleapiclient.discovery import build
import os
import time
from typing import Dict, Optional

def get_credentials(credentials_base64: Optional[str] = None):
    """Get Google Cloud credentials from base64-encoded string."""
    if credentials_base64:
        import base64
        import tempfile
        
        # Decode base64 credentials
        credentials_json = base64.b64decode(credentials_base64).decode('utf-8')
        
        # Create temporary file for credentials
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp_file:
            temp_file.write(credentials_json)
            temp_file_path = temp_file.name
        
        try:
            return service_account.Credentials.from_service_account_file(
                temp_file_path,
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)
    return None

def create_instance(
    project_id: str,
    zone: str,
    instance_name: str,
    machine_type: str = "n1-standard-4",
    startup_script: str = "",
    credentials_base64: Optional[str] = None
) -> compute_v1.Instance:
    """Create a GCP instance."""
    try:
        credentials = get_credentials(credentials_base64)
        compute = build('compute', 'v1', credentials=credentials)

        # Get the latest Debian image
        image_response = compute.images().getFromFamily(
            project='debian-cloud',
            family='debian-11'
        ).execute()

        source_disk_image = image_response['selfLink']

        # Configure the machine
        machine_type_url = f"zones/{zone}/machineTypes/{machine_type}"

        # Configure GPU if using GPU machine type
        accelerators = None
        if "a2-" in machine_type:  # A100 GPU
            accelerators = [{
                "acceleratorType": f"zones/{zone}/acceleratorTypes/nvidia-tesla-a100",
                "acceleratorCount": 1
            }]
        elif "n1-" in machine_type:  # T4 GPU
            accelerators = [{
                "acceleratorType": f"zones/{zone}/acceleratorTypes/nvidia-tesla-t4",
                "acceleratorCount": 1
            }]

        config = {
            'name': instance_name,
            'machineType': machine_type_url,
            'disks': [
                {
                    'boot': True,
                    'autoDelete': True,
                    'initializeParams': {
                        'sourceImage': source_disk_image,
                        'diskSizeGb': '50'
                    }
                }
            ],
            'networkInterfaces': [{
                'network': 'global/networks/default',
                'accessConfigs': [
                    {'type': 'ONE_TO_ONE_NAT', 'name': 'External NAT'}
                ]
            }],
            'serviceAccounts': [{
                'email': 'default',
                'scopes': [
                    'https://www.googleapis.com/auth/devstorage.read_write',
                    'https://www.googleapis.com/auth/compute'
                ]
            }],
            'metadata': {
                'items': [{
                    'key': 'startup-script',
                    'value': startup_script
                }]
            }
        }

        # Add GPU configuration if needed
        if accelerators:
            config['guestAccelerators'] = accelerators
            # Add required GPU-specific items
            config['metadata']['items'].extend([
                {
                    'key': 'install-nvidia-driver',
                    'value': 'True'
                }
            ])
            # Add scheduling requirements for GPU
            config['scheduling'] = {
                'onHostMaintenance': 'TERMINATE',
                'automaticRestart': True
            }

        # Create the instance
        operation = compute.instances().insert(
            project=project_id,
            zone=zone,
            body=config
        ).execute()

        # Wait for the operation to complete
        wait_for_operation(compute, project_id, zone, operation['name'])

        # Get the created instance
        instance = compute.instances().get(
            project=project_id,
            zone=zone,
            instance=instance_name
        ).execute()

        return instance

    except Exception as e:
        print(f"Error creating instance: {str(e)}")
        raise

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