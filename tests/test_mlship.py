import pytest
import os
import json
import time
import requests
import subprocess
import signal
import psutil
from pathlib import Path
from click.testing import CliRunner
from mlship.cli import cli
from mlship.utils.constants import PID_FILE, METRICS_FILE, LOG_FILE, CONFIG_FILE
from mlship.utils.create_test_model import create_test_model
from mlship.utils.daemon import cleanup_files

# Constants
TEST_PORT = 8001  # Use a different port for testing
TEST_MODEL_PATH = "test_model.pkl"
BASE_URL = f"http://localhost:{TEST_PORT}"
API_URL = f"{BASE_URL}/api"
UI_URL = f"{BASE_URL}/ui"

def kill_process_tree(pid):
    """Kill a process and all its children"""
    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        
        # Kill children first
        for child in children:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
                
        # Kill parent
        try:
            parent.kill()
        except psutil.NoSuchProcess:
            pass
            
        # Wait for processes to die
        gone, alive = psutil.wait_procs([parent] + children, timeout=3)
        
        # Force kill if any are still alive
        for p in alive:
            try:
                p.kill()
            except psutil.NoSuchProcess:
                pass
                
    except psutil.NoSuchProcess:
        pass

def cleanup_processes():
    """Clean up any running mlship processes"""
    # Find all mlship processes
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline'] or [])
            if ('mlship' in cmdline and 'deploy' in cmdline) or 'uvicorn' in cmdline:
                kill_process_tree(proc.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

@pytest.fixture(scope="session", autouse=True)
def global_cleanup():
    """Global cleanup that runs before and after all tests"""
    # Clean up before tests
    cleanup_processes()
    cleanup_files()
    
    yield
    
    # Clean up after tests
    cleanup_processes()
    cleanup_files()
    
    # Remove test model if it exists
    if os.path.exists(TEST_MODEL_PATH):
        os.remove(TEST_MODEL_PATH)

@pytest.fixture
def runner():
    return CliRunner()

@pytest.fixture
def test_model():
    """Create a test model for testing"""
    model_path = create_test_model(TEST_MODEL_PATH)
    yield model_path
    if os.path.exists(model_path):
        os.remove(model_path)

@pytest.fixture
def cleanup():
    """Clean up files before and after each test"""
    cleanup_files()
    yield
    cleanup_files()

def test_deploy_command_basic(runner, test_model, cleanup):
    """Test basic deployment without daemon mode"""
    # Start in background to avoid blocking
    process = subprocess.Popen(
        ["mlship", "deploy", test_model, "--port", str(TEST_PORT)],
        preexec_fn=os.setsid
    )
    
    try:
        # Wait for server to start
        time.sleep(2)
        
        # Check if server is running
        response = requests.get(f"http://localhost:{TEST_PORT}/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        
    finally:
        # Clean up
        if process:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(timeout=5)

def test_deploy_command_daemon(runner, test_model, cleanup):
    """Test deployment in daemon mode"""
    result = runner.invoke(cli, ['deploy', test_model, '--daemon', '--port', str(TEST_PORT)])
    assert result.exit_code == 0
    assert f"🚀 API: http://localhost:{TEST_PORT}" in result.output
    
    # Wait for server to start
    time.sleep(2)
    
    # Check if server is running
    response = requests.get(f"http://localhost:{TEST_PORT}/health")
    assert response.status_code == 200
    
    # Clean up
    runner.invoke(cli, ['stop'])

def test_deploy_command_missing_model(runner, cleanup):
    """Test deployment with missing model file"""
    result = runner.invoke(cli, ['deploy', 'nonexistent.pkl'])
    assert result.exit_code == 1
    assert "Model file not found" in result.output

def test_deploy_command_server_running(runner, test_model, cleanup):
    """Test deployment when server is already running"""
    # First deployment
    process = subprocess.Popen(
        ["mlship", "deploy", test_model, "--port", str(TEST_PORT)],
        preexec_fn=os.setsid
    )
    
    try:
        # Wait for server to start
        time.sleep(2)
        
        # Second deployment should fail
        result = runner.invoke(cli, ['deploy', test_model])
        assert result.exit_code == 1
        assert "Server is already running" in result.output
        
    finally:
        # Clean up
        if process:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(timeout=5)

def test_status_command_no_server(runner, cleanup):
    """Test status command when no server is running"""
    result = runner.invoke(cli, ['status'])
    assert result.exit_code == 0
    assert "Server not running" in result.output

def test_status_command_with_server(runner, test_model, cleanup):
    """Test status command with running server"""
    # Deploy first
    process = subprocess.Popen(
        ["mlship", "deploy", test_model, "--port", str(TEST_PORT)],
        preexec_fn=os.setsid
    )
    
    try:
        # Wait for server to start
        time.sleep(2)
        
        # Check status
        result = runner.invoke(cli, ['status'])
        assert result.exit_code == 0
        assert "Server running" in result.output
        
    finally:
        # Clean up
        if process:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait(timeout=5)

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

def test_stop_command_with_server(runner, test_model, cleanup):
    """Test stop command with running server"""
    # Deploy first
    process = subprocess.Popen(
        ["mlship", "deploy", test_model, "--port", str(TEST_PORT)],
        preexec_fn=os.setsid
    )
    
    try:
        # Wait for server to start
        time.sleep(2)
        
        # Stop server
        result = runner.invoke(cli, ['stop'])
        assert result.exit_code == 0
        assert "Server stopped" in result.output
        
        # Verify server is stopped
        with pytest.raises(requests.ConnectionError):
            requests.get(f"http://localhost:{TEST_PORT}/health")
            
    finally:
        # Clean up if stop failed
        try:
            if process and process.poll() is None:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                process.wait(timeout=5)
        except:
            pass

def test_stop_command_stale_pid(runner, cleanup):
    """Test stop command with stale PID file"""
    # Create a PID file with non-existent PID
    with open(PID_FILE, 'w') as f:
        f.write("99999")  # Non-existent PID
        
    result = runner.invoke(cli, ['stop'])
    assert result.exit_code == 0
    assert not os.path.exists(PID_FILE)

def test_configure_command(runner, cleanup):
    """Test configure command"""
    result = runner.invoke(cli, ['configure'], input='test-key\ntest-secret\n')
    assert result.exit_code == 0
    assert "Configuration saved successfully" in result.output
    
    # Verify config content
    with open(CONFIG_FILE) as f:
        config = json.load(f)
        assert config['aws_key'] == 'test-key'
        assert config['aws_secret'] == 'test-secret'