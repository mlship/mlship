import os
import joblib
import numpy as np
import torch
import tensorflow as tf
import onnx
import json
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
import pickle
import xgboost as xgb
from sklearn.base import BaseEstimator
import time

class SimpleNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 10)
        self.linear2 = torch.nn.Linear(10, 1)
        
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        return torch.sigmoid(self.linear2(x))

class ModelWrapper:
    """Base wrapper to add get_model_info method to models"""
    
    SUPPORTED_FORMATS = {
        # PyTorch
        '.pt': 'pytorch',
        '.pth': 'pytorch',
        '.safetensors': 'pytorch',
        '.ckpt': 'pytorch',
        
        # TensorFlow/Keras
        '.pb': 'tensorflow',
        '.h5': 'tensorflow',
        '.tflite': 'tensorflow-lite',
        
        # ONNX
        '.onnx': 'onnx',
        
        # Scikit-learn
        '.joblib': 'sklearn',
        '.pkl': 'pickle',
        
        # Others
        '.model': 'xgboost',
    }

    def __init__(self, model, model_type, input_type, output_type, feature_names=None, classes=None):
        self.model = model
        self.model_type = model_type
        self.input_type = input_type
        self.output_type = output_type
        self.feature_names = feature_names
        self.classes = classes
        self.request_count = 0
        self.total_latency = 0
        self.average_latency = 0

    @classmethod
    def load(cls, model_path: str):
        """Load a model from the given path."""
        model_path = os.path.abspath(model_path)
        
        # Check if it's a directory (could be SavedModel or Hugging Face model)
        if os.path.isdir(model_path):
            return cls._load_directory_model(model_path)
        
        # Get file extension
        ext = os.path.splitext(model_path)[1].lower()
        if ext not in cls.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported model format: {ext}")
        
        model_format = cls.SUPPORTED_FORMATS[ext]
        
        # Load based on format
        if model_format == 'pytorch':
            return cls._load_pytorch_model(model_path)
        elif model_format == 'tensorflow':
            return cls._load_tensorflow_model(model_path)
        elif model_format == 'tensorflow-lite':
            return cls._load_tflite_model(model_path)
        elif model_format == 'onnx':
            return cls._load_onnx_model(model_path)
        elif model_format == 'sklearn':
            return cls._load_sklearn_model(model_path)
        elif model_format == 'pickle':
            return cls._load_pickle_model(model_path)
        elif model_format == 'xgboost':
            return cls._load_xgboost_model(model_path)
        else:
            raise ValueError(f"Format {model_format} not yet implemented")

    @classmethod
    def _load_directory_model(cls, model_path):
        """Load a model from a directory (SavedModel or Hugging Face)."""
        # Check for SavedModel
        if os.path.exists(os.path.join(model_path, 'saved_model.pb')):
            return cls._load_savedmodel(model_path)
        
        # Check for Hugging Face model
        if os.path.exists(os.path.join(model_path, 'config.json')):
            return cls._load_huggingface_model(model_path)
        
        raise ValueError("Unknown directory model format")

    @classmethod
    def _load_pytorch_model(cls, model_path):
        """Load a PyTorch model."""
        if model_path.endswith('.safetensors'):
            from safetensors.torch import load_file
            state_dict = load_file(model_path)
            model = SimpleNet()  # Create a new instance of SimpleNet
            model.load_state_dict(state_dict)
        else:
            model = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Try to get model info
        input_type = 'tensor'
        output_type = 'tensor'
        model_type = 'pytorch'
        
        if hasattr(model, 'config'):
            if hasattr(model.config, 'num_labels'):
                output_type = 'label'
            if hasattr(model.config, 'id2label'):
                classes = list(model.config.id2label.values())
        
        return cls(model, model_type, input_type, output_type)

    @classmethod
    def _load_tensorflow_model(cls, model_path):
        """Load a TensorFlow model."""
        model = tf.keras.models.load_model(model_path)
        
        # Get model info
        input_type = 'tensor'
        output_type = 'tensor'
        model_type = 'tensorflow'
        
        if isinstance(model.output_shape[-1], int):
            if model.output_shape[-1] == 1:
                output_type = 'numeric'
            else:
                output_type = 'label'
                classes = list(range(model.output_shape[-1]))
        
        return cls(model, model_type, input_type, output_type)

    @classmethod
    def _load_tflite_model(cls, model_path):
        """Load a TensorFlow Lite model."""
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        return cls(
            interpreter,
            'tensorflow-lite',
            'tensor',
            'tensor',
            feature_names=[detail['name'] for detail in input_details]
        )

    @classmethod
    def _load_onnx_model(cls, model_path):
        """Load an ONNX model."""
        model = onnx.load(model_path)
        
        # Get input and output info
        input_names = [input.name for input in model.graph.input]
        output_names = [output.name for output in model.graph.output]
        
        return cls(
            model,
            'onnx',
            'tensor',
            'tensor',
            feature_names=input_names
        )

    @classmethod
    def _load_sklearn_model(cls, model_path):
        """Load a scikit-learn model."""
        model = joblib.load(model_path)
        
        # Get model info
        model_type = type(model).__name__
        input_type = 'numeric'
        output_type = 'numeric'
        feature_names = None
        classes = None
        
        if hasattr(model, 'feature_names_in_'):
            feature_names = model.feature_names_in_.tolist()
        
        if hasattr(model, 'classes_'):
            output_type = 'label'
            classes = model.classes_.tolist()
        
        return cls(model, model_type, input_type, output_type, feature_names, classes)

    @classmethod
    def _load_pickle_model(cls, model_path):
        """Load a pickled model."""
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Try to determine model type
        if isinstance(model, BaseEstimator):
            return cls._load_sklearn_model(model_path)
        
        # Generic pickle model
        return cls(
            model,
            'pickle',
            'numeric',
            'numeric'
        )

    @classmethod
    def _load_xgboost_model(cls, model_path):
        """Load an XGBoost model."""
        model = xgb.Booster()
        model.load_model(model_path)
        
        return cls(
            model,
            'xgboost',
            'numeric',
            'numeric'
        )

    @classmethod
    def _load_huggingface_model(cls, model_path):
        """Load a Hugging Face model."""
        # Load config
        with open(os.path.join(model_path, 'config.json')) as f:
            config = json.load(f)
        
        # Load model and tokenizer
        model = AutoModel.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Determine model type from config
        model_type = config.get('model_type', 'transformer')
        
        # Bundle model and tokenizer together
        model_bundle = {
            'model': model,
            'tokenizer': tokenizer,
            'config': config
        }
        
        return cls(
            model_bundle,
            model_type,
            'text',
            'text'
        )

    @classmethod
    def _load_savedmodel(cls, model_path):
        """Load a TensorFlow SavedModel."""
        model = tf.saved_model.load(model_path)
        
        # Try to get signature info
        if hasattr(model, 'signatures'):
            input_names = list(model.signatures['serving_default'].inputs.keys())
            output_names = list(model.signatures['serving_default'].outputs.keys())
        else:
            input_names = None
            output_names = None
        
        return cls(
            model,
            'tensorflow-saved',
            'tensor',
            'tensor',
            feature_names=input_names
        )

    def predict(self, inputs):
        """Make a prediction with the model."""
        start_time = time.time()
        try:
            # Convert inputs to numpy array if needed
            if isinstance(inputs, list):
                inputs = np.array(inputs)
            
            if isinstance(self.model, dict) and 'model' in self.model:
                # Hugging Face model
                tokenizer = self.model['tokenizer']
                model = self.model['model']
                inputs = tokenizer(inputs, return_tensors='pt', padding=True, truncation=True)
                outputs = model(**inputs)
                prediction = outputs.last_hidden_state.mean(dim=1).detach().numpy()
            elif isinstance(self.model, tf.lite.Interpreter):
                # TFLite model
                self.model.set_tensor(self.model.get_input_details()[0]['index'], inputs)
                self.model.invoke()
                prediction = self.model.get_tensor(self.model.get_output_details()[0]['index'])
            elif isinstance(self.model, torch.nn.Module):
                # PyTorch model
                if not isinstance(inputs, torch.Tensor):
                    inputs = torch.FloatTensor(inputs)
                with torch.no_grad():
                    prediction = self.model(inputs).numpy()
            elif isinstance(self.model, xgb.Booster):
                # XGBoost model
                if not isinstance(inputs, xgb.DMatrix):
                    inputs = xgb.DMatrix(inputs)
                prediction = self.model.predict(inputs)
            elif isinstance(self.model, tf.keras.Model):
                # TensorFlow model
                prediction = self.model.predict(inputs, verbose=0)
            else:
                # Standard model prediction (sklearn, etc.)
                if len(inputs.shape) == 1:
                    # Single sample, reshape to 2D
                    inputs = inputs.reshape(1, -1)
                prediction = self.model.predict(inputs)
            
            # Update metrics
            self.request_count += 1
            latency = (time.time() - start_time) * 1000  # Convert to ms
            self.total_latency += latency
            self.average_latency = self.total_latency / self.request_count
            
            # Convert prediction to list for JSON serialization
            if isinstance(prediction, np.ndarray):
                prediction = prediction.tolist()
            
            return prediction
            
        except Exception as e:
            print(f"Error making prediction: {str(e)}")
            raise

    def get_model_info(self):
        """Get information about the model."""
        return {
            "type": self.model_type,
            "input_type": self.input_type,
            "output_type": self.output_type,
            "features": self.feature_names,
            "classes": self.classes,
            "n_features": len(self.feature_names) if self.feature_names else None
        }