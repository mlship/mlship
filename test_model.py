from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import joblib

# Generate a random dataset
X, y = make_classification(n_samples=100, n_features=4, random_state=42)

# Create and train a simple model
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X, y)

# Save the model
joblib.dump(model, 'test_model.joblib') 