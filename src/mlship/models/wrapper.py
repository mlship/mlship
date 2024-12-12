import os
import joblib
import pickle
import numpy as np
import time
from typing import List, Any, Dict, Union
from pathlib import Path
from ..utils.model_architectures import PTRegression, PTImageClassifier

class ModelWrapper:
    def __init__(self):
        self.model = None
        self.model_path = None
        self.model_type = None
        self.framework = None
        self.input_type = None
        self.output_type = None
        self.features = None
        self.classes = None
        self.input_shape = None
        self.output_shape = None
        self.preprocessing = None
        # Initialize metrics
        self.request_count = 0
        self.total_latency = 0
        self.average_latency = 0

    def load_model(self, model_path: str) -> None:
        """Load a model from a file path, supporting multiple formats."""
        self.model_path = os.path.abspath(model_path)
        extension = Path(model_path).suffix.lower()
        
        try:
            if extension in ['.pt', '.pth']:
                self._load_pytorch_model()
            elif extension in ['.pb', '.h5', '.keras']:
                self._load_tensorflow_model()
            elif extension in ['.pkl', '.joblib']:
                self._load_sklearn_model()
            elif extension in ['.onnx']:
                self._load_onnx_model()
            else:
                raise ValueError(f"Unsupported model format: {extension}")
            
            # Extract model information
            self._extract_model_info()
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {self.model_path}: {str(e)}")

    def _load_pytorch_model(self):
        """Load PyTorch model."""
        import torch
        import json
        from pathlib import Path
        
        # Try to load model architecture info
        model_dir = Path(self.model_path).parent
        model_name = Path(self.model_path).stem
        arch_file = model_dir / f"{model_name}_arch.json"
        
        try:
            if arch_file.exists():
                # Load architecture info
                with open(arch_file, 'r') as f:
                    arch_info = json.load(f)
                
                # Import the model class
                if arch_info['type'] == 'PTRegression':
                    model = PTRegression()
                elif arch_info['type'] == 'PTImageClassifier':
                    model = PTImageClassifier()
                else:
                    raise ValueError(f"Unknown model type: {arch_info['type']}")
                
                # Load state dict
                state_dict = torch.load(self.model_path, weights_only=True)
                model.load_state_dict(state_dict)
                self.model = model
            else:
                # Try loading as a full model (legacy support)
                self.model = torch.load(self.model_path, weights_only=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load PyTorch model: {str(e)}")
            
        self.framework = 'pytorch'
        if hasattr(self.model, 'eval'):
            self.model.eval()
        # Set device to CPU for consistency
        if hasattr(self.model, 'cpu'):
            self.model = self.model.cpu()

    def _load_tensorflow_model(self):
        """Load TensorFlow/Keras model."""
        import tensorflow as tf
        extension = Path(self.model_path).suffix.lower()
        if extension == '.pb':
            self.model = tf.saved_model.load(self.model_path)
        else:
            self.model = tf.keras.models.load_model(self.model_path)
        self.framework = 'tensorflow'

    def _load_sklearn_model(self):
        """Load scikit-learn model."""
        try:
            self.model = joblib.load(self.model_path)
        except:
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
        self.framework = 'sklearn'

    def _load_onnx_model(self):
        """Load ONNX model."""
        import onnxruntime as ort
        self.model = ort.InferenceSession(self.model_path)
        self.framework = 'onnx'

    def _extract_model_info(self) -> None:
        """Extract model information after loading."""
        if self.model is None:
            raise ValueError("No model loaded")

        self.model_type = type(self.model).__name__
        
        # Framework-specific info extraction
        if self.framework == 'sklearn':
            self._extract_sklearn_info()
        elif self.framework == 'pytorch':
            self._extract_pytorch_info()
        elif self.framework == 'tensorflow':
            self._extract_tensorflow_info()
        elif self.framework == 'onnx':
            self._extract_onnx_info()

    def _extract_sklearn_info(self):
        """Extract info from scikit-learn model."""
        self.input_type = 'numeric'
        
        # Try to get feature names
        if hasattr(self.model, 'feature_names_in_'):
            self.features = self.model.feature_names_in_.tolist()
            # Set input shape based on number of features
            self.input_shape = (None, len(self.features))
        elif hasattr(self.model, 'n_features_in_'):
            self.features = [f'feature_{i+1}' for i in range(self.model.n_features_in_)]
            self.input_shape = (None, self.model.n_features_in_)
        else:
            self.features = []
            self.input_shape = (None, 0)
            
        # Try to get class information for classifiers
        if hasattr(self.model, 'classes_'):
            self.classes = self.model.classes_.tolist()
            self.output_type = 'classification'
            self.output_shape = (None, len(self.classes))
        else:
            self.classes = None
            self.output_type = 'regression'
            self.output_shape = (None, 1)
            
        # Try to get preprocessing info
        if hasattr(self.model, 'preprocessing'):
            self.preprocessing = self.model.preprocessing

    def _extract_pytorch_info(self):
        """Extract info from PyTorch model."""
        self.input_type = 'tensor'
        
        # Get input/output shape
        if hasattr(self.model, 'input_shape'):
            self.input_shape = self.model.input_shape
            # For non-image models, create feature names based on input dimension
            if not isinstance(self.model, PTImageClassifier):
                self.features = [f'feature_{i+1}' for i in range(self.input_shape[1])]
        
        if hasattr(self.model, 'output_shape'):
            self.output_shape = self.model.output_shape
            
        # Try to determine if classification model
        if hasattr(self.model, 'num_classes'):
            self.output_type = 'classification'
            self.classes = list(range(self.model.num_classes))
        else:
            self.output_type = 'regression'

    def _extract_tensorflow_info(self):
        """Extract info from TensorFlow model."""
        self.input_type = 'tensor'
        
        # Get input/output shapes
        if hasattr(self.model, 'input_shape'):
            self.input_shape = self.model.input_shape
            # Check if this is the text classifier model
            if self.model_path.endswith('tensorflow_text_classifier.h5'):
                self.input_type = 'text'
                self.features = ['text']  # Single text input
        if hasattr(self.model, 'output_shape'):
            self.output_shape = self.model.output_shape
            if self.output_shape[-1] > 1:
                self.output_type = 'classification'
                self.classes = list(range(self.output_shape[-1]))
            else:
                self.output_type = 'regression'
                
        # Try to get preprocessing info
        if hasattr(self.model, 'preprocessing'):
            self.preprocessing = self.model.preprocessing

    def _extract_onnx_info(self):
        """Extract info from ONNX model."""
        self.input_type = 'tensor'
        inputs = self.model.get_inputs()
        outputs = self.model.get_outputs()
        
        # Get input/output info
        self.input_shape = inputs[0].shape
        self.output_shape = outputs[0].shape
        self.features = [input.name for input in inputs]
        
        # Try to determine if classification
        if len(outputs) == 1 and outputs[0].shape[-1] > 1:
            self.output_type = 'classification'
            self.classes = list(range(outputs[0].shape[-1]))
        else:
            self.output_type = 'regression'

    def predict(self, inputs: List[List[float]]) -> List[Any]:
        """Make predictions using the loaded model."""
        if self.model is None:
            raise ValueError("No model loaded")
            
        # Convert inputs based on framework
        start_time = time.time()
        try:
            if self.framework == 'sklearn':
                X = np.array(inputs)
                predictions = self.model.predict(X)
            elif self.framework == 'pytorch':
                import torch
                X = torch.tensor(inputs, dtype=torch.float32)
                
                # Reshape input if it's an image model
                if isinstance(self.model, PTImageClassifier):
                    if X.shape != (1, 3, 32, 32):
                        # Reshape from [batch_size, flattened] to [batch_size, channels, height, width]
                        X = X.reshape(-1, 3, 32, 32)
                
                with torch.no_grad():
                    predictions = self.model(X).numpy()
            elif self.framework == 'tensorflow':
                import tensorflow as tf
                
                # Special handling for text classifier
                if self.input_type == 'text':
                    # Convert text input to sequence of integers
                    # For simplicity, we'll use a basic character-level encoding
                    # In production, you should use a proper tokenizer
                    text = inputs[0][0]  # Get the text from the input
                    # Convert to sequence of character codes
                    char_sequence = [ord(c) % 256 for c in str(text)]
                    # Pad or truncate to length 100
                    if len(char_sequence) > 100:
                        char_sequence = char_sequence[:100]
                    else:
                        char_sequence = char_sequence + [0] * (100 - len(char_sequence))
                    # Convert to tensor with batch dimension
                    X = tf.convert_to_tensor([char_sequence], dtype=tf.float32)
                else:
                    X = tf.convert_to_tensor(inputs, dtype=tf.float32)
                
                if hasattr(self.model, 'predict'):
                    predictions = self.model.predict(X, verbose=0)
                else:
                    predictions = self.model(X).numpy()
            elif self.framework == 'onnx':
                input_name = self.model.get_inputs()[0].name
                # Convert inputs to numpy array with correct shape
                X = np.array(inputs, dtype=np.float32)
                # Get expected input shape
                expected_shape = self.model.get_inputs()[0].shape
                # If input shape has fixed dimensions (after batch size), ensure input matches
                if len(expected_shape) > 1 and all(isinstance(dim, int) for dim in expected_shape[1:]):
                    if X.shape[1] != expected_shape[1]:
                        # Repeat the input values to match expected shape
                        X = np.repeat(X, expected_shape[1] // X.shape[1], axis=1)
                predictions = self.model.run(None, {input_name: X})[0]
            else:
                raise ValueError(f"Unsupported framework: {self.framework}")
            
            end_time = time.time()
            
            # Update metrics
            self.request_count += 1
            latency = (end_time - start_time) * 1000  # Convert to milliseconds
            self.total_latency += latency
            self.average_latency = self.total_latency / self.request_count
            
            return predictions.tolist()
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")

    def get_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if self.model is None:
            raise ValueError("No model loaded")
            
        return {
            "type": self.model_type,
            "framework": self.framework,
            "input_type": self.input_type,
            "output_type": self.output_type,
            "features": self.features,
            "classes": self.classes,
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "preprocessing": self.preprocessing,
            "model_path": self.model_path,
            "metrics": {
                "requests": self.request_count,
                "avg_latency": round(self.average_latency, 2)
            }
        }
