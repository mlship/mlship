import boto3
from typing import Dict, List

def get_available_gpus(region: str = "us-east-1") -> List[Dict]:
    """Get available GPU types in AWS."""
    try:
        ec2 = boto3.client('ec2', region_name=region)
        
        # Get available instance types with GPUs
        response = ec2.describe_instance_types(
            Filters=[
                {
                    'Name': 'accelerator-manufacturer',
                    'Values': ['nvidia']
                }
            ]
        )
        
        available_gpus = []
        for instance in response['InstanceTypes']:
            if 'GpuInfo' in instance:
                gpu_info = instance['GpuInfo']
                gpu_type = gpu_info['Gpus'][0]['Name']
                
                available_gpus.append({
                    'name': gpu_type,
                    'description': f"{gpu_type} on {instance['InstanceType']}",
                    'maximumCardsPerInstance': gpu_info['Gpus'][0]['Count'],
                    'provider': 'aws',
                    'status': 'UP'  # We assume it's available if listed
                })
        
        return available_gpus
    except Exception as e:
        print(f"Error getting AWS GPUs: {str(e)}")
        return [] 