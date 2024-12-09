from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np
import os

def create_test_model(output_path='test_model.pkl'):
    """Create a simple test model"""
    # Convert to absolute path
    output_path = os.path.abspath(output_path)
    
    # Create a simple random forest model
    model = RandomForestClassifier(n_estimators=2, random_state=42)
    
    # Create some dummy data
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 0, 1])
    
    # Train the model
    model.fit(X, y)
    
    # Save the model
    joblib.dump(model, output_path)
    
    return output_path

if __name__ == '__main__':
    model_path = create_test_model()
    print(f"Test model created at: {model_path}") 