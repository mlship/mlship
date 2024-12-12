import os
import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from PIL import Image
import joblib
import onnx
import onnxruntime
from pathlib import Path
import json

# PyTorch model definitions
class PTRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(5, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )
        self.input_shape = (None, 5)
        self.output_shape = (None, 1)
    
    def forward(self, x):
        return self.layers(x)

class PTImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(32 * 8 * 8, 10)
        self.input_shape = (None, 3, 32, 32)
        self.output_shape = (None, 10)
        self.num_classes = 10
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(-1, 32 * 8 * 8)
        return self.fc(x)

class SimpleONNXModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3, 1)
    
    def forward(self, x):
        return self.fc(x)

def create_output_dir():
    """Create output directory for test models."""
    output_dir = Path('test_models')
    output_dir.mkdir(exist_ok=True)
    return output_dir

def generate_sklearn_models(output_dir):
    """Generate scikit-learn models for regression and classification."""
    print("Generating scikit-learn models...")
    
    # Regression model
    X_reg = np.random.rand(1000, 5)
    y_reg = 2 * X_reg[:, 0] + 3 * X_reg[:, 1] - X_reg[:, 2] + np.random.normal(0, 0.1, 1000)
    reg_model = RandomForestRegressor(n_estimators=10, random_state=42)
    reg_model.fit(X_reg, y_reg)
    reg_model.feature_names_in_ = np.array(['x1', 'x2', 'x3', 'x4', 'x5'])
    joblib.dump(reg_model, output_dir / 'sklearn_regression.joblib')
    print("- Created regression model: sklearn_regression.joblib")
    
    # Classification model
    X_clf = np.random.rand(1000, 3)
    y_clf = (X_clf[:, 0] + X_clf[:, 1] > 1).astype(int)
    clf_model = RandomForestClassifier(n_estimators=10, random_state=42)
    clf_model.fit(X_clf, y_clf)
    clf_model.feature_names_in_ = np.array(['feature1', 'feature2', 'feature3'])
    joblib.dump(clf_model, output_dir / 'sklearn_classification.pkl')
    print("- Created classification model: sklearn_classification.pkl")

def generate_pytorch_models(output_dir):
    """Generate PyTorch models for different tasks."""
    print("Generating PyTorch models...")
    
    # Simple regression model
    pt_reg = PTRegression()
    pt_reg.eval()  # Set to evaluation mode
    
    # Save state dict and architecture info
    torch.save(pt_reg.state_dict(), output_dir / 'pytorch_regression.pt')
    with open(output_dir / 'pytorch_regression_arch.json', 'w') as f:
        json.dump({
            'type': 'PTRegression',
            'input_shape': pt_reg.input_shape,
            'output_shape': pt_reg.output_shape
        }, f)
    print("- Created regression model: pytorch_regression.pt")
    
    # Image classification model
    pt_img = PTImageClassifier()
    pt_img.eval()  # Set to evaluation mode
    
    # Save state dict and architecture info
    torch.save(pt_img.state_dict(), output_dir / 'pytorch_image_classifier.pth')
    with open(output_dir / 'pytorch_image_classifier_arch.json', 'w') as f:
        json.dump({
            'type': 'PTImageClassifier',
            'input_shape': pt_img.input_shape,
            'output_shape': pt_img.output_shape,
            'num_classes': pt_img.num_classes
        }, f)
    print("- Created image classifier: pytorch_image_classifier.pth")

def generate_tensorflow_models(output_dir):
    """Generate TensorFlow models for different tasks."""
    print("Generating TensorFlow models...")
    
    # Text classification model
    text_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(100,)),
        tf.keras.layers.Embedding(10000, 16),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    text_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    text_model.save(output_dir / 'tensorflow_text_classifier.h5')
    print("- Created text classifier: tensorflow_text_classifier.h5")
    
    # Time series model
    time_series_model = tf.keras.Sequential([
        tf.keras.layers.LSTM(32, input_shape=(10, 1)),
        tf.keras.layers.Dense(1)
    ])
    time_series_model.compile(optimizer='adam', loss='mse')
    tf.saved_model.save(time_series_model, str(output_dir / 'tensorflow_timeseries'))
    print("- Created time series model: tensorflow_timeseries/")

def generate_huggingface_model(output_dir):
    """Generate a fine-tuned BERT model for text classification."""
    print("Generating Hugging Face model...")
    
    try:
        # Load a small pre-trained model
        model_name = "prajjwal1/bert-tiny"  # Very small BERT model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        
        # Save the model and tokenizer
        model_dir = output_dir / 'huggingface_text_classifier'
        model_dir.mkdir(exist_ok=True)
        model.save_pretrained(str(model_dir))
        tokenizer.save_pretrained(str(model_dir))
        print("- Created text classifier: huggingface_text_classifier/")
    except Exception as e:
        print(f"Warning: Could not generate Hugging Face model: {str(e)}")

def generate_onnx_model(output_dir):
    """Generate an ONNX model."""
    print("Generating ONNX model...")
    
    try:
        model = SimpleONNXModel()
        model.eval()  # Set to evaluation mode
        
        # Create dummy input
        dummy_input = torch.randn(1, 3)
        
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_dir / 'simple_model.onnx',
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        print("- Created ONNX model: simple_model.onnx")
    except Exception as e:
        print(f"Warning: Could not generate ONNX model: {str(e)}")

def main():
    """Generate all test models."""
    output_dir = create_output_dir()
    
    # Generate models for each framework
    generate_sklearn_models(output_dir)
    generate_pytorch_models(output_dir)
    generate_tensorflow_models(output_dir)
    generate_huggingface_model(output_dir)
    generate_onnx_model(output_dir)
    
    print("\nModel generation complete!")
    print(f"Models saved in: {output_dir.absolute()}")
    print("\nGenerated models:")
    for path in output_dir.glob('**/*'):
        if path.is_file():
            print(f"- {path.relative_to(output_dir)}")

if __name__ == '__main__':
    main() 