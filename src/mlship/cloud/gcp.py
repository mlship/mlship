from google.cloud import compute_v1
from google.oauth2 import service_account
from googleapiclient.discovery import build
import os
import time
from typing import Dict, Optional
import json

# Define available zones for each GPU type
GPU_ZONE_MAPPING = {
    "a2-highgpu-1g": [  # A100 zones
        "us-central1-a",
        "us-central1-b",
        "us-central1-c",
    ],
    "n1-standard-4": [  # T4 zones
        "us-central1-a",
        "us-central1-b",
        "us-central1-c",
        "us-west1-a",
        "us-west1-b",
        "us-east1-b",
        "us-east1-c",
        "us-east4-a",
        "us-east4-b"
    ]
}

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

def get_fallback_config(machine_type: str) -> list:
    """Get fallback configurations for a machine type."""
    if "a2-highgpu" in machine_type:
        return [
            ("n1-standard-4", "nvidia-tesla-t4", 1),  # Fallback to T4
        ]
    return []

def get_available_gpus(compute, project_id: str, zone: str) -> list:
    """Get available GPU types in a zone."""
    try:
        accelerator_types = compute.acceleratorTypes().list(
            project=project_id,
            zone=zone
        ).execute()
        
        return [
            {
                'name': acc['name'].split('/')[-1],  # Get just the GPU name
                'description': acc.get('description', ''),
                'maximumCardsPerInstance': acc.get('maximumCardsPerInstance', 1)
            }
            for acc in accelerator_types.get('items', [])
            if 'nvidia' in acc['name'].lower()  # Only NVIDIA GPUs
        ]
    except Exception as e:
        print(f"Error getting GPUs for zone {zone}: {str(e)}")
        return []

def get_available_zones_with_gpus(credentials_base64: Optional[str] = None) -> Dict:
    """Get all zones with their available GPU types."""
    try:
        credentials = get_credentials(credentials_base64)
        compute = build('compute', 'v1', credentials=credentials)
        
        # Get all zones
        regions_result = compute.regions().list(
            project='mlship'
        ).execute()
        
        available_zones = {}
        
        for region in regions_result.get('items', []):
            region_name = region['name']
            if not region_name.startswith('us-'):  # Only US regions for now
                continue
                
            # Get zones in this region
            zones_result = compute.zones().list(
                project='mlship',
                filter=f'region eq {region["selfLink"]}'
            ).execute()
            
            for zone in zones_result.get('items', []):
                zone_name = zone['name']
                
                # Check GPU availability in this zone
                available_gpus = get_available_gpus(compute, 'mlship', zone_name)
                
                if available_gpus:  # Only include zones that have GPUs
                    available_zones[zone_name] = {
                        'gpus': available_gpus,
                        'status': zone['status'],
                        'region': region_name
                    }
        
        return available_zones
    except Exception as e:
        print(f"Error getting available zones: {str(e)}")
        return {}

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
        print(f"Checking GPU availability...")
        credentials = get_credentials(credentials_base64)
        compute = build('compute', 'v1', credentials=credentials)
        
        # Get available zones with GPUs
        available_zones = get_available_zones_with_gpus(credentials_base64)
        if not available_zones:
            raise Exception("No zones with GPUs are currently available")
        
        print(f"Available zones with GPUs: {json.dumps(available_zones, indent=2)}")
        
        # Try each zone that has the GPU we want
        for zone_name, zone_info in available_zones.items():
            if zone_info['status'] != 'UP':
                continue
                
            # Check if zone has the GPU type we want
            gpu_type = None
            if "a2-" in machine_type:
                gpu_type = "nvidia-tesla-a100"
            elif "n1-" in machine_type:
                gpu_type = "nvidia-tesla-t4"
                
            zone_has_gpu = any(
                gpu['name'] == gpu_type
                for gpu in zone_info['gpus']
            )
            
            if not zone_has_gpu:
                continue
                
            try:
                print(f"Attempting to create instance in zone: {zone_name}")
                
                # Get the latest Debian image
                image_response = compute.images().getFromFamily(
                    project='debian-cloud',
                    family='debian-11'
                ).execute()

                source_disk_image = image_response['selfLink']
                machine_type_url = f"zones/{zone_name}/machineTypes/{machine_type}"

                # Configure GPU
                accelerators = [{
                    "acceleratorType": f"zones/{zone_name}/acceleratorTypes/{gpu_type}",
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

                # Add GPU configuration
                config['guestAccelerators'] = accelerators
                config['metadata']['items'].extend([
                    {
                        'key': 'install-nvidia-driver',
                        'value': 'True'
                    }
                ])
                config['scheduling'] = {
                    'onHostMaintenance': 'TERMINATE',
                    'automaticRestart': True
                }

                # Create the instance
                operation = compute.instances().insert(
                    project=project_id,
                    zone=zone_name,
                    body=config
                ).execute()

                # Wait for the operation to complete
                wait_for_operation(compute, project_id, zone_name, operation['name'])

                # Get the created instance
                instance = compute.instances().get(
                    project=project_id,
                    zone=zone_name,
                    instance=instance_name
                ).execute()

                print(f"Successfully created instance in zone: {zone_name}")
                return instance

            except Exception as e:
                print(f"Failed to create instance in zone {zone_name}: {str(e)}")
                continue
        
        raise Exception("Could not create instance in any available zone")
        
    except Exception as e:
        error_msg = f"Failed to create instance: {str(e)}"
        print(error_msg)
        raise Exception(error_msg)

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

def wait_for_operation(compute, project_id: str, zone: str, operation_name: str):
    """Wait for a GCP operation to complete."""
    print(f"Waiting for operation {operation_name} to finish...")
    while True:
        result = compute.zoneOperations().get(
            project=project_id,
            zone=zone,
            operation=operation_name
        ).execute()

        if result['status'] == 'DONE':
            print("Operation done.")
            if 'error' in result:
                raise Exception(result['error'])
            return result

        time.sleep(5)  # Wait 5 seconds before checking again
  