import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from transformers import AutoTokenizer, AutoModel
import os
from pathlib import Path

def create_image_classifier():
    """Create a simple image classification model."""
    print("Creating image classification model...")
    
    # Use a pre-trained ResNet18 model
    model = models.resnet18(pretrained=True)
    model.eval()
    
    # Create example input
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Export to ONNX
    output_path = "example_models/image_classifier.onnx"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['image'],
        output_names=['predictions'],
        dynamic_axes={
            'image': {0: 'batch_size'},
            'predictions': {0: 'batch_size'}
        }
    )
    print(f"Saved image classifier to {output_path}")

def create_text_classifier():
    """Create a text classification model."""
    print("Creating text classification model...")
    
    # Simple sentiment classifier
    class SentimentClassifier(nn.Module):
        def __init__(self, vocab_size=30522, embedding_dim=768, hidden_dim=256):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, 2)  # binary classification
            
        def forward(self, input_ids, attention_mask):
            embedded = self.embedding(input_ids)
            packed_output, (hidden, cell) = self.lstm(embedded)
            logits = self.fc(hidden[-1])
            return logits

    model = SentimentClassifier()
    model.eval()
    
    # Create example inputs
    dummy_input_ids = torch.randint(0, 30522, (1, 128))
    dummy_attention_mask = torch.ones(1, 128)
    
    # Export to ONNX
    output_path = "example_models/text_classifier.onnx"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask),
        output_path,
        input_names=['input_ids', 'attention_mask'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
            'logits': {0: 'batch_size'}
        }
    )
    print(f"Saved text classifier to {output_path}")

def create_tabular_regressor():
    """Create a tabular data regression model."""
    print("Creating tabular regression model...")
    
    class TabularRegressor(nn.Module):
        def __init__(self, input_dim=10):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1)
            )
            
        def forward(self, x):
            return self.layers(x)

    model = TabularRegressor()
    model.eval()
    
    # Create example input
    dummy_input = torch.randn(1, 10)
    
    # Export to ONNX
    output_path = "example_models/tabular_regressor.onnx"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['features'],
        output_names=['prediction'],
        dynamic_axes={
            'features': {0: 'batch_size'},
            'prediction': {0: 'batch_size'}
        }
    )
    print(f"Saved tabular regressor to {output_path}")

def create_time_series_forecaster():
    """Create a time series forecasting model."""
    print("Creating time series forecasting model...")
    
    class TimeSeriesForecaster(nn.Module):
        def __init__(self, input_size=1, hidden_size=64, num_layers=2):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True
            )
            self.fc = nn.Linear(hidden_size, 1)
            
        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            return self.fc(lstm_out[:, -1, :])

    model = TimeSeriesForecaster()
    model.eval()
    
    # Create example input (sequence_length=50, features=1)
    dummy_input = torch.randn(1, 50, 1)
    
    # Export to ONNX
    output_path = "example_models/time_series_forecaster.onnx"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['sequence'],
        output_names=['forecast'],
        dynamic_axes={
            'sequence': {0: 'batch_size', 1: 'sequence_length'},
            'forecast': {0: 'batch_size'}
        }
    )
    print(f"Saved time series forecaster to {output_path}")

if __name__ == "__main__":
    # Create example models directory
    os.makedirs("example_models", exist_ok=True)
    
    # Generate different types of models
    create_image_classifier()
    create_text_classifier()
    create_tabular_regressor()
    create_time_series_forecaster()
    
    print("\nAll example models have been created in the 'example_models' directory.") 