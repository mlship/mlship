import os
import numpy as np
import torch
import tensorflow as tf
import onnx
from sklearn.ensemble import RandomForestClassifier
import joblib
import xgboost as xgb
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
from pathlib import Path

class SimpleNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(2, 10)
        self.linear2 = torch.nn.Linear(10, 1)
        
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        return torch.sigmoid(self.linear2(x))

def create_test_models(output_dir='test_models'):
    """Create test models in different formats."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create training data
    np.random.seed(42)
    X = np.random.rand(1000, 2)
    y = (X[:, 1] > X[:, 0]).astype(int)
    
    # 1. Scikit-learn model
    print("Creating scikit-learn model...")
    sklearn_model = RandomForestClassifier(n_estimators=10, random_state=42)
    sklearn_model.fit(X, y)
    sklearn_model.feature_names_in_ = np.array(['first_number', 'second_number'])
    joblib.dump(sklearn_model, os.path.join(output_dir, 'model.joblib'))
    
    # 2. PyTorch model
    print("Creating PyTorch model...")
    torch_model = SimpleNet()
    torch_model.train()
    optimizer = torch.optim.Adam(torch_model.parameters())
    criterion = torch.nn.BCELoss()
    
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y).reshape(-1, 1)
    
    for _ in range(100):
        optimizer.zero_grad()
        output = torch_model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()
    
    torch.save(torch_model, os.path.join(output_dir, 'model.pt'))
    
    # Save as safetensors
    from safetensors.torch import save_file
    state_dict = torch_model.state_dict()
    save_file(state_dict, os.path.join(output_dir, 'model.safetensors'))
    
    # 3. TensorFlow model
    print("Creating TensorFlow model...")
    tf_model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(2,)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    tf_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    tf_model.fit(X, y, epochs=10, verbose=0)
    tf_model.save(os.path.join(output_dir, 'model.h5'))
    
    # Save as SavedModel
    tf.saved_model.save(tf_model, os.path.join(output_dir, 'saved_model'))
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
    tflite_model = converter.convert()
    with open(os.path.join(output_dir, 'model.tflite'), 'wb') as f:
        f.write(tflite_model)
    
    # 4. ONNX model
    print("Creating ONNX model...")
    dummy_input = torch.randn(1, 2)
    torch.onnx.export(torch_model, dummy_input, os.path.join(output_dir, 'model.onnx'))
    
    # 5. XGBoost model
    print("Creating XGBoost model...")
    dtrain = xgb.DMatrix(X, label=y)
    params = {
        'max_depth': 3,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss'
    }
    xgb_model = xgb.train(params, dtrain, num_boost_round=10)
    xgb_model.save_model(os.path.join(output_dir, 'model.model'))
    
    # 6. Hugging Face model
    print("Creating Hugging Face model...")
    hf_model_path = os.path.join(output_dir, 'hf_model')
    os.makedirs(hf_model_path, exist_ok=True)
    
    # Create a simple config
    config = {
        "architectures": ["SimpleClassifier"],
        "model_type": "simple",
        "num_labels": 2,
        "id2label": {0: "first_larger", 1: "second_larger"}
    }
    
    with open(os.path.join(hf_model_path, 'config.json'), 'w') as f:
        json.dump(config, f)
    
    # Save PyTorch model as Hugging Face model
    torch.save(torch_model.state_dict(), os.path.join(hf_model_path, 'pytorch_model.bin'))
    
    print("\nTest models created successfully in directory:", output_dir)
    print("\nAvailable models:")
    for model_file in os.listdir(output_dir):
        print(f"- {model_file}")

if __name__ == '__main__':
    create_test_models() 