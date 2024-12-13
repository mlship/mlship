from azure.mgmt.compute import ComputeManagementClient
from azure.identity import DefaultAzureCredential
from typing import Dict, List

def get_available_gpus(subscription_id: str, location: str = "eastus") -> List[Dict]:
    """Get available GPU types in Azure."""
    try:
        credential = DefaultAzureCredential()
        compute_client = ComputeManagementClient(credential, subscription_id)
        
        # Get available VM sizes in the location
        sizes = compute_client.virtual_machine_sizes.list(location)
        
        available_gpus = []
        for size in sizes:
            if size.gpu_count > 0:
                available_gpus.append({
                    'name': f"nvidia-{size.name.lower()}",  # Azure doesn't expose exact GPU model
                    'description': f"GPU on {size.name}",
                    'maximumCardsPerInstance': size.gpu_count,
                    'provider': 'azure',
                    'status': 'UP'  # We assume it's available if listed
                })
        
        return available_gpus
    except Exception as e:
        print(f"Error getting Azure GPUs: {str(e)}")
        return [] 