import os
import joblib
import pickle
import numpy as np
import time
from typing import List, Any, Dict, Union
from pathlib import Path

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
        # Initialize metrics
        self.request_count = 0
        self.total_latency = 0
        self.average_latency = 0

    def load_model(self, model_path: str) -> None:
        """Load a model from a file path, supporting multiple formats."""
        self.model_path = os.path.abspath(model_path)
        extension = Path(model_path).suffix.lower()
        
        try:
            # Try loading with joblib first (most common)
            try:
                self.model = joblib.load(self.model_path)
                self.framework = 'sklearn'
                print(f"Successfully loaded model from {self.model_path} using joblib")
            except Exception as joblib_error:
                try:
                    # If joblib fails, try pickle
                    with open(self.model_path, 'rb') as f:
                        self.model = pickle.load(f)
                    self.framework = 'sklearn'
                    print(f"Successfully loaded model from {self.model_path} using pickle")
                except Exception as pickle_error:
                    # Try importing other frameworks only if needed
                    if extension in ['.pt', '.pth']:
                        import torch
                        self.model = torch.load(self.model_path)
                        self.framework = 'pytorch'
                        if hasattr(self.model, 'eval'):
                            self.model.eval()
                    elif extension in ['.h5', '.keras']:
                        import tensorflow as tf
                        self.model = tf.keras.models.load_model(self.model_path)
                        self.framework = 'tensorflow'
                    elif extension in ['.onnx']:
                        import onnxruntime as ort
                        self.model = ort.InferenceSession(self.model_path)
                        self.framework = 'onnx'
                    else:
                        raise RuntimeError(
                            f"Failed to load model. "
                            f"Joblib error: {str(joblib_error)}. "
                            f"Pickle error: {str(pickle_error)}"
                        )
            
            # Extract model information
            self._extract_model_info()
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {self.model_path}: {str(e)}")

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
        # Try to get feature names
        if hasattr(self.model, 'feature_names_in_'):
            self.features = self.model.feature_names_in_.tolist()
        else:
            self.features = []
            
        # Try to get class information for classifiers
        if hasattr(self.model, 'classes_'):
            self.classes = self.model.classes_.tolist()
            self.output_type = 'classification'
        else:
            self.classes = None
            self.output_type = 'regression'
            
        self.input_type = 'numeric'

    def _extract_pytorch_info(self):
        """Extract info from PyTorch model."""
        self.input_type = 'tensor'
        if hasattr(self.model, 'num_classes'):
            self.output_type = 'classification'
            self.classes = list(range(self.model.num_classes))
        else:
            self.output_type = 'unknown'

    def _extract_tensorflow_info(self):
        """Extract info from TensorFlow model."""
        self.input_type = 'tensor'
        if hasattr(self.model, 'output_shape'):
            output_shape = self.model.output_shape
            if output_shape[-1] > 1:
                self.output_type = 'classification'
                self.classes = list(range(output_shape[-1]))
            else:
                self.output_type = 'regression'

    def _extract_onnx_info(self):
        """Extract info from ONNX model."""
        self.input_type = 'tensor'
        inputs = self.model.get_inputs()
        outputs = self.model.get_outputs()
        self.features = [input.name for input in inputs]
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
                with torch.no_grad():
                    predictions = self.model(X).numpy()
            elif self.framework == 'tensorflow':
                import tensorflow as tf
                X = tf.convert_to_tensor(inputs, dtype=tf.float32)
                predictions = self.model.predict(X, verbose=0)
            elif self.framework == 'onnx':
                input_name = self.model.get_inputs()[0].name
                predictions = self.model.run(None, {input_name: np.array(inputs).astype(np.float32)})[0]
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
            "model_path": self.model_path,
            "metrics": {
                "requests": self.request_count,
                "avg_latency": round(self.average_latency, 2)
            }
        }
