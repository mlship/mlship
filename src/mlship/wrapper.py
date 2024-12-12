import onnxruntime as ort
import numpy as np
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ONNXModelWrapper:
    def __init__(self, model_path):
        """Initialize ONNX model wrapper.
        
        Args:
            model_path (str): Path to the ONNX model file
        """
        self.model_path = Path(model_path)
        if not self.model_path.suffix == '.onnx':
            raise ValueError(f"Model must be an ONNX file, got: {self.model_path}")
        
        if not self.model_path.exists():
            raise ValueError(f"Model file not found: {self.model_path}")
        
        # Load metadata if exists
        self.meta_path = self.model_path.with_suffix('.json')
        if self.meta_path.exists():
            with open(self.meta_path) as f:
                self.metadata = json.load(f)
        else:
            # Create default metadata from model info
            self.metadata = self._create_metadata()
        
        # Load model
        logger.info(f"Loading ONNX model from {self.model_path}")
        try:
            self.session = ort.InferenceSession(str(self.model_path))
            self._validate_model()
            logger.info("Model loaded successfully")
        except Exception as e:
            raise ValueError(f"Failed to load ONNX model: {str(e)}")
    
    def _validate_model(self):
        """Validate model inputs and outputs."""
        # Get input details
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_type = self.session.get_inputs()[0].type
        
        # Get output details
        self.output_name = self.session.get_outputs()[0].name
        self.output_shape = self.session.get_outputs()[0].shape
        self.output_type = self.session.get_outputs()[0].type
        
        logger.info(f"Model input: {self.input_name} {self.input_shape} {self.input_type}")
        logger.info(f"Model output: {self.output_name} {self.output_shape} {self.output_type}")
    
    def _create_metadata(self):
        """Create metadata from model information."""
        session = ort.InferenceSession(str(self.model_path))
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        
        input_schema = {
            "type": "object",
            "properties": {}
        }
        
        # Create input schema
        for inp in inputs:
            input_schema["properties"][inp.name] = {
                "type": "array",
                "shape": inp.shape,
                "dtype": inp.type,
                "description": f"Input tensor of shape {inp.shape}"
            }
        
        output_schema = {
            "type": "object",
            "properties": {}
        }
        
        # Create output schema
        for out in outputs:
            output_schema["properties"][out.name] = {
                "type": "array",
                "shape": out.shape,
                "dtype": out.type,
                "description": f"Output tensor of shape {out.shape}"
            }
        
        return {
            "input_schema": input_schema,
            "output_schema": output_schema,
            "model_type": "onnx"
        }
    
    def predict(self, data):
        """Make prediction with the model.
        
        Args:
            data (dict): Input data matching the model's input schema
            
        Returns:
            dict: Model predictions
        """
        try:
            # Convert inputs to numpy arrays
            inputs = {}
            for name, value in data.items():
                if isinstance(value, list):
                    inputs[name] = np.array(value, dtype=np.float32)
                else:
                    inputs[name] = np.array([value], dtype=np.float32)
            
            # Run prediction
            outputs = self.session.run(None, inputs)
            
            # Format output
            result = {}
            for i, out in enumerate(self.session.get_outputs()):
                result[out.name] = outputs[i].tolist()
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise ValueError(f"Failed to make prediction: {str(e)}")
    
    def get_input_schema(self):
        """Get the input schema for this model."""
        return self.metadata['input_schema']
    
    def get_output_schema(self):
        """Get the output schema for this model."""
        return self.metadata['output_schema'] 