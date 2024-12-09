import click
import logging
import sys
import subprocess
import time
import os
from .server.loader import ModelLoader
from .server.app import ModelServer
from .utils.config import save_config
from .utils.daemon import (
    start_daemon, is_process_running, get_pid, save_pid, 
    get_metrics, update_metrics, stop_daemon, view_logs
)
from .utils.constants import LOG_FILE, PID_FILE, METRICS_FILE
from .utils.create_test_model import create_test_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@click.group()
def cli():
    """MLShip - Simple ML Model Deployment"""
    pass

@cli.command()
@click.argument('model_path')
@click.option('--port', default=8000, help='Port to run the server on')
@click.option('--host', default='localhost', help='Host to run the server on')
@click.option('--daemon', is_flag=True, help='Run server in background')
@click.option('--foreground', is_flag=True, hidden=True, help='Internal flag for daemon mode')
def deploy(model_path, port, host, daemon, foreground):
    """Deploy a model as an API"""
    try:
        # Convert model path to absolute path if relative
        if not os.path.isabs(model_path):
            model_path = os.path.abspath(model_path)
        
        # Check if model exists
        if not os.path.exists(model_path):
            raise click.ClickException(f"Model file not found: {model_path}")
        
        # Check if server is already running
        pid = get_pid()
        if pid and is_process_running(pid):
            raise click.ClickException("Server is already running")
            
        if daemon and not foreground:
            # Start daemon process
            try:
                pid = start_daemon(model_path, host, port)
                click.echo(f"🚀 Model deployed at http://{host}:{port}")
                click.echo("Run 'mlship status' to check deployment")
                click.echo("Run 'mlship logs' to view logs")
                click.echo("Run 'mlship stop' to stop deployment")
            except Exception as e:
                logger.error(f"Failed to start daemon: {str(e)}")
                raise click.ClickException(str(e))
        else:
            # Run in foreground
            model = ModelLoader.load(model_path)
            server = ModelServer(model)
            
            # Save PID and initialize metrics before starting server
            save_pid(os.getpid())
            update_metrics()
            
            if not foreground:
                click.echo(f"\nStarting MLShip API server:")
                click.echo(f"→ API URL: http://{host}:{port}")
                click.echo(f"→ Health check: http://{host}:{port}/health")
                click.echo(f"→ Predictions: http://{host}:{port}/predict")
                click.echo("\nPress Ctrl+C to stop the server\n")
            
            try:
                server.serve(host=host, port=port)
            finally:
                # Clean up PID file when server stops
                if os.path.exists(PID_FILE):
                    os.remove(PID_FILE)
                if os.path.exists(METRICS_FILE):
                    os.remove(METRICS_FILE)
    except Exception as e:
        # Clean up files if startup fails
        if os.path.exists(PID_FILE):
            os.remove(PID_FILE)
        if os.path.exists(METRICS_FILE):
            os.remove(METRICS_FILE)
        logger.error(f"Deployment failed: {str(e)}")
        raise click.ClickException(str(e))

@cli.command()
@click.option('--output', default='test_model.pkl', help='Output path for test model')
def create_model(output):
    """Create a test model for deployment"""
    try:
        model_path = create_test_model(output)
        click.echo(f"✓ Test model created at: {model_path}")
        click.echo(f"Run 'mlship deploy {output}' to deploy the model")  # Use original output path
    except Exception as e:
        logger.error(f"Failed to create test model: {str(e)}")
        raise click.ClickException(str(e))

@cli.command()
def status():
    """Check deployment status"""
    pid = get_pid()
    if not pid:
        click.echo("✗ Server not running")
        return
        
    if not is_process_running(pid):
        os.remove(PID_FILE)
        if os.path.exists(METRICS_FILE):
            os.remove(METRICS_FILE)
        click.echo("✗ Server not running")
        return
    
    metrics = get_metrics()
    if metrics:
        uptime = int((time.time() - metrics['start_time']) / 60)
        click.echo("✓ Server running")
        click.echo(f"Uptime: {uptime}m")
        click.echo(f"Requests: {metrics['requests']}")
        click.echo(f"Avg latency: {metrics['avg_latency']}ms")
    else:
        click.echo("✓ Server running")
        click.echo("No metrics available")

@cli.command()
def logs():
    """View server logs"""
    try:
        if not os.path.exists(LOG_FILE):
            click.echo("No logs available - server hasn't been started")
            return
        
        with open(LOG_FILE) as f:
            content = f.read()
            if not content:
                click.echo("No logs available (file is empty)")
            else:
                click.echo(content)
    except Exception as e:
        logger.error(f"Failed to read logs: {str(e)}")
        raise click.ClickException(str(e))

@cli.command()
def stop():
    """Stop the server"""
    try:
        pid = get_pid()
        if not pid:
            click.echo("No server running")
            return
            
        if not is_process_running(pid):
            # Clean up stale files
            for file in [PID_FILE, METRICS_FILE]:
                if os.path.exists(file):
                    os.remove(file)
            click.echo("No server running")
            return
            
        # Stop the server
        stop_daemon()
    except Exception as e:
        logger.error(f"Failed to stop server: {str(e)}")
        raise click.ClickException(str(e))

@cli.command()
@click.option('--aws-key', prompt=True, help='AWS Access Key ID')
@click.option('--aws-secret', prompt=True, hide_input=True, 
              help='AWS Secret Access Key')
def configure(aws_key, aws_secret):
    """Configure AWS credentials"""
    try:
        config = {'aws_key': aws_key, 'aws_secret': aws_secret}
        save_config(config)
        click.echo("Configuration saved successfully!")
    except Exception as e:
        logger.error(f"Configuration failed: {str(e)}")
        raise click.ClickException(str(e))