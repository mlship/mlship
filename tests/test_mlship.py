import pytest
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from mlship.server import ModelServer
from fastapi.testclient import TestClient

def test_imports():
    """Test that we can import required packages"""
    import numpy
    import scipy
    import sklearn
    assert True

def test_randomforest():
    """Test that we can create a RandomForestClassifier"""
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=2)
    assert model is not None

@pytest.fixture
def test_model():
    """Create a simple test model"""
    model = RandomForestClassifier(n_estimators=2)
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])
    model.fit(X, y)
    return model

@pytest.fixture
def model_path(test_model, tmp_path):
    """Save test model to temporary file"""
    model_path = tmp_path / "model.pkl"
    joblib.dump(test_model, model_path)
    return str(model_path)

@pytest.fixture
def client(test_model):
    """Create test client"""
    server = ModelServer(test_model)
    return TestClient(server.app)

def test_predict_endpoint(client):
    """Test prediction endpoint"""
    response = client.post(
        "/predict",
        json={"inputs": [[1, 2]]}
    )
    assert response.status_code == 200
    assert "predictions" in response.json()

def test_health_endpoint(client):
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"