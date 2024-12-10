import os
import json
import pytest
from click.testing import CliRunner
from mlship.cli import cli
from mlship.utils.constants import PID_FILE, METRICS_FILE, LOG_FILE, CONFIG_FILE
import time
import requests
import subprocess
import signal
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from mlship.utils.create_test_model import create_test_model

@pytest.fixture
def runner():
    return CliRunner()

@pytest.fixture
def cleanup():
    # Clean up before and after tests
    for file in [PID_FILE, METRICS_FILE, LOG_FILE, CONFIG_FILE]:
        if os.path.exists(file):
            os.remove(file)
    yield
    for file in [PID_FILE, METRICS_FILE, LOG_FILE, CONFIG_FILE]:
        if os.path.exists(file):
            os.remove(file)

def test_deploy_command_basic(runner, cleanup):
    """Test basic deployment without daemon mode"""
    result = runner.invoke(cli, ['deploy', 'test_model.pkl'])
    assert result.exit_code == 0
    assert "Model deployed successfully" in result.output
    assert os.path.exists(PID_FILE)

def test_deploy_command_daemon(runner, cleanup):
    """Test deployment in daemon mode"""
    result = runner.invoke(cli, ['deploy', 'test_model.pkl', '--daemon'])
    assert result.exit_code == 0
    assert "Model deployed at http://localhost:8000" in result.output
    assert os.path.exists(PID_FILE)
    assert os.path.exists(METRICS_FILE)

def test_deploy_command_missing_model(runner, cleanup):
    """Test deployment with missing model file"""
    result = runner.invoke(cli, ['deploy', 'nonexistent.pkl'])
    assert result.exit_code == 1
    assert "Model file not found" in result.output

def test_deploy_command_server_running(runner, cleanup):
    """Test deployment when server is already running"""
    # First deployment
    runner.invoke(cli, ['deploy', 'test_model.pkl'])
    # Second deployment should fail
    result = runner.invoke(cli, ['deploy', 'test_model.pkl'])
    assert result.exit_code == 1
    assert "Server is already running" in result.output

def test_status_command_no_server(runner, cleanup):
    """Test status command when no server is running"""
    result = runner.invoke(cli, ['status'])
    assert result.exit_code == 0
    assert "Server not running" in result.output

def test_status_command_with_server(runner, cleanup):
    """Test status command with running server"""
    # Deploy first
    runner.invoke(cli, ['deploy', 'test_model.pkl', '--daemon'])
    # Check status
    result = runner.invoke(cli, ['status'])
    assert result.exit_code == 0
    assert "Server running" in result.output
    assert "Uptime" in result.output
    assert "Requests" in result.output

def test_logs_command_no_logs(runner, cleanup):
    """Test logs command when no logs exist"""
    result = runner.invoke(cli, ['logs'])
    assert result.exit_code == 0
    assert "No logs available" in result.output

def test_logs_command_with_logs(runner, cleanup):
    """Test logs command with existing logs"""
    test_log = "Test log message"
    with open(LOG_FILE, 'w') as f:
        f.write(test_log)

    result = runner.invoke(cli, ['logs'])
    assert result.exit_code == 0
    assert test_log in result.output

def test_stop_command_no_server(runner, cleanup):
    """Test stop command when no server is running"""
    result = runner.invoke(cli, ['stop'])
    assert result.exit_code == 0
    assert "No server running" in result.output

def test_stop_command_with_server(runner, cleanup):
    """Test stop command with running server"""
    # Deploy first
    runner.invoke(cli, ['deploy', 'test_model.pkl'])
    # Stop server
    result = runner.invoke(cli, ['stop'])
    assert result.exit_code == 0
    assert "Server stopped" in result.output
    assert not os.path.exists(PID_FILE)

def test_stop_command_stale_pid(runner, cleanup):
    """Test stop command with stale PID file"""
    # Create a PID file
    with open(PID_FILE, 'w') as f:
        f.write("12345")  # Non-existent PID
    # Stop server
    result = runner.invoke(cli, ['stop'])
    assert result.exit_code == 0
    assert not os.path.exists(PID_FILE)

def test_configure_command(runner):
    """Test configure command"""
    result = runner.invoke(cli, ['configure'], input='test-key\ntest-secret\n')
    assert result.exit_code == 0
    assert "Configuration saved successfully" in result.output
    assert os.path.exists(CONFIG_FILE)
    
    # Verify config content
    with open(CONFIG_FILE) as f:
        config = json.load(f)
        assert config['aws_key'] == 'test-key'
        assert config['aws_secret'] == 'test-secret'

# Constants
TEST_PORT = 8001  # Use a different port for testing
TEST_MODEL_PATH = "test_model.pkl"
BASE_URL = f"http://localhost:{TEST_PORT}"
API_URL = f"{BASE_URL}/api"
UI_URL = f"{BASE_URL}/ui"

@pytest.fixture(scope="session")
def test_model():
    """Create a test model for testing"""
    model_path = create_test_model(TEST_MODEL_PATH)
    yield model_path
    # Cleanup
    if os.path.exists(model_path):
        os.remove(model_path)

@pytest.fixture(scope="session")
def server_process(test_model):
    """Start the server for testing"""
    # Start server
    process = subprocess.Popen(
        ["mlship", "deploy", test_model, "--port", str(TEST_PORT), "--ui"],
        preexec_fn=os.setsid
    )
    
    # Wait for server to start
    time.sleep(2)
    
    yield process
    
    # Cleanup
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        process.wait(timeout=5)
    except (subprocess.TimeoutExpired, ProcessLookupError):
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    finally:
        subprocess.run(["mlship", "stop"])

@pytest.fixture(scope="session")
def chrome_driver():
    """Set up Chrome WebDriver for UI testing"""
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run in headless mode
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    driver = webdriver.Chrome(options=chrome_options)
    yield driver
    driver.quit()

def test_api_health(server_process):
    """Test API health endpoint"""
    response = requests.get(f"{API_URL}/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_api_model_info(server_process):
    """Test model info endpoint"""
    response = requests.get(f"{API_URL}/model-info")
    assert response.status_code == 200
    info = response.json()
    assert "type" in info
    assert "params" in info
    assert "features" in info
    assert info["n_features"] == 2

def test_api_prediction(server_process):
    """Test prediction endpoint"""
    # Test cases
    test_cases = [
        {"inputs": [[3, 4]], "expected": 1},  # second > first
        {"inputs": [[4, 3]], "expected": 0},  # second < first
        {"inputs": [[1, 2]], "expected": 1},  # second > first
        {"inputs": [[2, 1]], "expected": 0},  # second < first
    ]
    
    for case in test_cases:
        response = requests.post(
            f"{API_URL}/predict",
            json={"inputs": case["inputs"]}
        )
        assert response.status_code == 200
        result = response.json()
        assert "predictions" in result
        assert result["predictions"][0] == case["expected"]

def test_ui_loads(server_process, chrome_driver):
    """Test that UI loads correctly"""
    chrome_driver.get(UI_URL)
    
    # Wait for React to load
    WebDriverWait(chrome_driver, 10).until(
        EC.presence_of_element_located((By.ID, "root"))
    )
    
    # Check main components
    assert chrome_driver.find_element(By.CLASS_NAME, "text-4xl").text == "MLship Dashboard"
    assert chrome_driver.find_element(By.XPATH, "//h2[text()='Make Prediction']")
    assert chrome_driver.find_element(By.XPATH, "//h2[text()='Prediction History']")
    assert chrome_driver.find_element(By.XPATH, "//h2[text()='Real-time Metrics']")
    assert chrome_driver.find_element(By.XPATH, "//h2[text()='Model Information']")

def test_ui_prediction(server_process, chrome_driver):
    """Test making predictions through the UI"""
    chrome_driver.get(UI_URL)
    
    # Wait for React to load
    WebDriverWait(chrome_driver, 10).until(
        EC.presence_of_element_located((By.ID, "root"))
    )
    
    # Find input fields
    inputs = chrome_driver.find_elements(By.TAG_NAME, "input")
    assert len(inputs) == 2
    
    # Test cases
    test_cases = [
        {"values": ["3", "4"], "expected": "1"},  # second > first
        {"values": ["4", "3"], "expected": "0"},  # second < first
    ]
    
    for case in test_cases:
        # Enter values
        for input_field, value in zip(inputs, case["values"]):
            input_field.clear()
            input_field.send_keys(value)
        
        # Click predict button
        predict_button = chrome_driver.find_element(By.XPATH, "//button[text()='Predict']")
        predict_button.click()
        
        # Wait for prediction
        WebDriverWait(chrome_driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "bg-gray-50"))
        )
        
        # Check prediction
        prediction_elements = chrome_driver.find_elements(By.CLASS_NAME, "bg-gray-50")
        latest_prediction = prediction_elements[0]  # Most recent prediction
        assert case["expected"] in latest_prediction.text
        assert f"[{case['values'][0]}, {case['values'][1]}]" in latest_prediction.text

def test_ui_metrics_update(server_process, chrome_driver):
    """Test that metrics update after predictions"""
    chrome_driver.get(UI_URL)
    
    # Wait for React to load
    WebDriverWait(chrome_driver, 10).until(
        EC.presence_of_element_located((By.ID, "root"))
    )
    
    # Get initial request count
    initial_requests = chrome_driver.find_element(
        By.XPATH,
        "//p[contains(text(), 'Requests')]/following-sibling::p"
    ).text
    initial_count = int(initial_requests)
    
    # Make a prediction
    inputs = chrome_driver.find_elements(By.TAG_NAME, "input")
    inputs[0].send_keys("3")
    inputs[1].send_keys("4")
    predict_button = chrome_driver.find_element(By.XPATH, "//button[text()='Predict']")
    predict_button.click()
    
    # Wait for metrics to update
    time.sleep(2)  # Wait for WebSocket update
    
    # Check that request count increased
    new_requests = chrome_driver.find_element(
        By.XPATH,
        "//p[contains(text(), 'Requests')]/following-sibling::p"
    ).text
    new_count = int(new_requests)
    assert new_count > initial_count

def test_ui_prediction_history(server_process, chrome_driver):
    """Test that prediction history accumulates"""
    chrome_driver.get(UI_URL)
    
    # Wait for React to load
    WebDriverWait(chrome_driver, 10).until(
        EC.presence_of_element_located((By.ID, "root"))
    )
    
    # Make multiple predictions
    test_cases = [
        ["3", "4"],
        ["4", "3"],
        ["1", "2"]
    ]
    
    inputs = chrome_driver.find_elements(By.TAG_NAME, "input")
    predict_button = chrome_driver.find_element(By.XPATH, "//button[text()='Predict']")
    
    for values in test_cases:
        # Enter values
        for input_field, value in zip(inputs, values):
            input_field.clear()
            input_field.send_keys(value)
        
        predict_button.click()
        time.sleep(0.5)  # Wait for prediction to complete
    
    # Check prediction history
    predictions = chrome_driver.find_elements(By.CLASS_NAME, "bg-gray-50")
    assert len(predictions) >= len(test_cases)
    
    # Check that predictions are in reverse order (newest first)
    for i, case in enumerate(reversed(test_cases[:3])):  # Check last 3 predictions
        prediction_text = predictions[i].text
        assert f"[{case[0]}, {case[1]}]" in prediction_text