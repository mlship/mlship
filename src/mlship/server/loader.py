import joblib
import boto3
import tempfile
from pathlib import Path
import logging
import os
from ..utils.config import load_config
from ..models.wrapper import ModelWrapper

logger = logging.getLogger(__name__)

class ModelLoader:
    @staticmethod
    def load(model_path: str):
        """Load a model from the given path."""
        logger.info(f"Loading model from {model_path}")
        try:
            # Convert to absolute path if relative
            if not os.path.isabs(model_path):
                model_path = os.path.abspath(model_path)
                
            if model_path.startswith('s3://'):
                return ModelLoader._load_from_s3(model_path)
                
            # Create a new ModelWrapper instance and load the model
            wrapper = ModelWrapper()
            wrapper.load_model(model_path)
            return wrapper
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    @staticmethod
    def _load_from_s3(path: str):
        """Load a model from S3."""
        config = load_config()
        if not config.get('aws_key') or not config.get('aws_secret'):
            raise ValueError("AWS credentials not configured")

        s3 = boto3.client(
            's3',
            aws_access_key_id=config['aws_key'],
            aws_secret_access_key=config['aws_secret']
        )
        
        bucket, key = path.replace('s3://', '').split('/', 1)
        with tempfile.NamedTemporaryFile() as tmp:
            s3.download_file(bucket, key, tmp.name)
            wrapper = ModelWrapper()
            wrapper.load_model(tmp.name)
            return wrapper