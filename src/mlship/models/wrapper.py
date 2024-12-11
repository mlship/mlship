import os
import joblib
import pickle
import numpy as np
import time
from typing import List, Any, Dict, Union

class ModelWrapper:
    def __init__(self):
        self.model = None
        self.model_path = None
        self.model_type = None
        self.input_type = None
        self.output_type = None
        self.features = None
        self.classes = None
        # Initialize metrics
        self.request_count = 0
        self.total_latency = 0
        self.average_latency = 0

    def load_model(self, model_path: str) -> None:
        """Load a model from a file path, supporting both joblib and pickle formats."""
        self.model_path = os.path.abspath(model_path)
        
        # Try loading with joblib first
        try:
            self.model = joblib.load(self.model_path)
            print(f"Successfully loaded model from {self.model_path} using joblib")
        except Exception as joblib_error:
            try:
                # If joblib fails, try pickle
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                print(f"Successfully loaded model from {self.model_path} using pickle")
            except Exception as pickle_error:
                raise RuntimeError(
                    f"Failed to load model from {self.model_path}. "
                    f"Joblib error: {str(joblib_error)}. "
                    f"Pickle error: {str(pickle_error)}"
                )
        
        # Extract model information
        self._extract_model_info()

    def _extract_model_info(self) -> None:
        """Extract model information after loading."""
        if self.model is None:
            raise ValueError("No model loaded")

        self.model_type = type(self.model).__name__
        
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
            
        # Set input type based on feature names or default to numeric
        self.input_type = 'numeric'

    def predict(self, inputs: List[List[float]]) -> List[Any]:
        """Make predictions using the loaded model."""
        if self.model is None:
            raise ValueError("No model loaded")
            
        # Convert inputs to numpy array
        X = np.array(inputs)
        
        # Make predictions and track metrics
        start_time = time.time()
        try:
            predictions = self.model.predict(X)
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