from sklearn.ensemble import RandomForestClassifier
import joblib
import numpy as np
import os

def create_test_model(output_path='test_model.pkl'):
    """Create a test model using RandomForest that predicts 1 if second number > first number, 0 otherwise"""
    # Convert to absolute path
    output_path = os.path.abspath(output_path)
    
    # Create training data
    np.random.seed(42)
    
    # Generate regular grid points
    x1 = np.linspace(0, 10, 50)
    x2 = np.linspace(0, 10, 50)
    X1, X2 = np.meshgrid(x1, x2)
    X_grid = np.column_stack([X1.ravel(), X2.ravel()])
    y_grid = (X_grid[:, 1] > X_grid[:, 0]).astype(int)
    
    # Generate edge cases
    n_edge = 1000
    base = np.random.uniform(0, 10, n_edge)
    small_diffs = np.random.uniform(0.0001, 0.1, n_edge)
    
    # Cases where second number is slightly larger
    X_larger = np.column_stack([base, base + small_diffs])
    y_larger = np.ones(n_edge)
    
    # Cases where second number is slightly smaller
    X_smaller = np.column_stack([base, base - small_diffs])
    y_smaller = np.zeros(n_edge)
    
    # Equal number cases
    X_equal = np.column_stack([base, base])
    y_equal = np.zeros(n_edge)
    
    # Combine all datasets
    X = np.vstack([
        X_grid,
        np.repeat(X_larger, 10, axis=0),  # Repeat edge cases to give them more weight
        np.repeat(X_smaller, 10, axis=0),
        np.repeat(X_equal, 10, axis=0),
        # Add specific test cases multiple times
        np.tile([[0, 0.1], [0.1, 0], [0, 0]], (100, 1)),
        np.tile([[9.9, 10], [10, 9.9], [10, 10]], (100, 1))
    ])
    
    # Create labels
    y = np.hstack([
        y_grid,
        np.repeat(y_larger, 10),
        np.repeat(y_smaller, 10),
        np.repeat(y_equal, 10),
        np.tile([1, 0, 0], 100),  # Labels for [0, 0.1], [0.1, 0], [0, 0]
        np.tile([1, 0, 0], 100)   # Labels for [9.9, 10], [10, 9.9], [10, 10]
    ])
    
    # Create and train the model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'
    )
    
    # Add feature names
    model.feature_names_in_ = np.array(['first_number', 'second_number'])
    
    # Train the model
    model.fit(X, y)
    
    # Test cases
    test_cases = [
        ([3, 4], 1),   # second > first
        ([4, 3], 0),   # second < first
        ([1, 5], 1),   # second >> first
        ([5, 1], 0),   # second << first
        ([0, 0.1], 1), # tiny difference
        ([0.1, 0], 0), # tiny difference
        ([10, 9.9], 0), # tiny difference at upper range
        ([9.9, 10], 1), # tiny difference at upper range
    ]
    
    print("Testing predictions:")
    for input_data, expected in test_cases:
        pred = model.predict([input_data])[0]
        proba = model.predict_proba([input_data])[0]
        print(f"Input {input_data} -> Predicted {pred} (Expected {expected})")
        print(f"  Confidence: {max(proba):.3f}")
        assert pred == expected, f"Model prediction {pred} does not match expected {expected}"
    
    # Save the model
    joblib.dump(model, output_path)
    print(f"Model saved successfully at: {output_path}")
    
    return output_path

if __name__ == '__main__':
    model_path = create_test_model()
    print(f"Test model created at: {model_path}") 