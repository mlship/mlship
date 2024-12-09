import os
import json
import pytest
from click.testing import CliRunner
from mlship.cli import cli
from mlship.utils.constants import PID_FILE, METRICS_FILE, LOG_FILE, CONFIG_FILE

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