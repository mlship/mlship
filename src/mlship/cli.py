import click
import os
import sys
import json
import time
import logging
import subprocess
import psutil
import signal
from pathlib import Path
from .utils.constants import PID_FILE, METRICS_FILE, LOG_FILE, CONFIG_FILE
from .utils.daemon import cleanup_files, daemonize
from .utils.create_test_model import create_test_model
from .server.cloud_app import deploy_from_cli

logger = logging.getLogger(__name__)

@click.group()
def cli():
    """MLship - Simple ML Model Deployment

    Deploy machine learning models with a single command.

    Commands:
    - deploy: Deploy a model as a REST API
    - status: Check server status
    - logs: View server logs
    - stop: Stop the server
    - ui: Open web interface
    - create-test-model: Create a test model
    """
    pass

@cli.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--port', default=8000, help='Port to bind to')
@click.option('--ui/--no-ui', default=True, help='Enable/disable web interface')
@click.option('--daemon', is_flag=True, help='Run in daemon mode')
def deploy(model_path, host, port, ui, daemon):
    """Deploy a model as a REST API.
    
    MODEL_PATH is the path to the model file to deploy.
    Supported formats: .joblib, .pkl, .pt, .pth, .h5, .keras, .onnx
    
    Examples:
    \b
    mlship deploy model.joblib                # Run in foreground with UI
    mlship deploy model.pkl --port 8080       # Custom port
    mlship deploy model.h5 --no-ui            # Disable UI
    mlship deploy model.pt --daemon           # Run in background
    """
    try:
        # Check if server is already running
        if os.path.exists(PID_FILE):
            with open(PID_FILE) as f:
                pid = int(f.read().strip())
                try:
                    process = psutil.Process(pid)
                    if process.is_running():
                        click.echo("Server is already running. Stop it first.")
                        return
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            # Clean up stale PID file
            cleanup_files()

        # Convert model path to absolute path if relative
        if not os.path.isabs(model_path):
            model_path = os.path.abspath(model_path)

        # Print URLs
        click.echo(f"🚀 API: http://{host if host != '0.0.0.0' else 'localhost'}:{port}")
        if ui:
            click.echo(f"🎨 UI: http://{host if host != '0.0.0.0' else 'localhost'}:{port}/ui")
        else:
            click.echo("UI disabled")

        if daemon:
            click.echo("Starting server in daemon mode...")
            daemonize(model_path, host, port, ui)
        else:
            click.echo("Starting server in foreground mode...")
            from .server.app import start_server
            start_server(model_path, host=host, port=port, ui=ui)
    except Exception as e:
        logger.error(f"Deployment failed: {str(e)}")
        click.echo(f"Failed to load model: {str(e)}")
        cleanup_files()
        sys.exit(1)

@cli.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--port', default=3000, help='Port to bind to')
def serve(model_path, host, port):
    """Serve a model with the cloud dashboard UI.
    
    MODEL_PATH is the path to the model file to serve.
    This will start the cloud dashboard UI on port 3000 (default).
    
    Examples:
    \b
    mlship serve model.joblib                # Start dashboard on port 3000
    mlship serve model.pkl --port 3001       # Custom port for dashboard
    """
    try:
        # Convert to absolute path
        abs_path = os.path.abspath(model_path)
        # Save model path for cloud deployment
        from .server.cloud_app import save_model_path
        save_model_path(abs_path)
        # Open the cloud dashboard
        import webbrowser
        webbrowser.open(f"http://{host if host != '0.0.0.0' else 'localhost'}:{port}")
        # Start the cloud server
        from .server.cloud_app import start_cloud_server
        start_cloud_server(host=host, port=8000)  # API server on 8000
    except Exception as e:
        logger.error(f"Failed to serve model: {str(e)}")
        click.echo(f"Failed to serve model: {str(e)}")
        sys.exit(1)

@cli.command()
def status():
    """Check if the server is running and show metrics.
    
    Shows:
    - Running status and PID
    - Memory and CPU usage
    - Request count and latency metrics
    """
    try:
        if not os.path.exists(PID_FILE):
            click.echo("Server is not running")
            return

        with open(PID_FILE) as f:
            pid = int(f.read().strip())

        try:
            process = psutil.Process(pid)
            if process.is_running():
                click.echo(f"Server is running (PID: {pid})")
                
                # Get process info
                memory = process.memory_info().rss / 1024 / 1024  # Convert to MB
                cpu = process.cpu_percent()
                click.echo(f"Memory usage: {memory:.1f} MB")
                click.echo(f"CPU usage: {cpu:.1f}%")
                
                # Get metrics if available
                if os.path.exists(METRICS_FILE):
                    try:
                        with open(METRICS_FILE) as f:
                            metrics = json.load(f)
                            click.echo(f"Total requests: {metrics.get('requests', 0)}")
                            click.echo(f"Average latency: {metrics.get('avg_latency', 0):.2f}ms")
                    except json.JSONDecodeError:
                        pass
            else:
                click.echo("Server is not running")
                cleanup_files()
        except psutil.NoSuchProcess:
            click.echo("Server is not running")
            cleanup_files()
    except Exception as e:
        click.echo(f"Error checking status: {str(e)}")
        sys.exit(1)

@cli.command()
@click.option('--tail', '-n', default=10, help='Number of lines to show')
@click.option('--follow', '-f', is_flag=True, help='Follow log output')
def logs(tail, follow):
    """View server logs.
    
    Options:
    -n, --tail: Number of lines to show (default: 10)
    -f, --follow: Follow log output in real-time
    """
    try:
        if not os.path.exists(LOG_FILE):
            click.echo("No logs found")
            return

        if follow:
            # Use tail -f equivalent
            subprocess.run(['tail', '-f', LOG_FILE])
        else:
            # Read last N lines
            with open(LOG_FILE) as f:
                lines = f.readlines()
                for line in lines[-tail:]:
                    click.echo(line.strip())
    except Exception as e:
        click.echo(f"Error reading logs: {str(e)}")
        sys.exit(1)

@cli.command()
@click.option('--force', '-f', is_flag=True, help='Force stop the server')
def stop(force):
    """Stop the server.
    
    Options:
    -f, --force: Force stop the server if it's not responding
    """
    try:
        if not os.path.exists(PID_FILE):
            click.echo("Server is not running")
            return

        with open(PID_FILE) as f:
            pid = int(f.read().strip())

        try:
            process = psutil.Process(pid)
            if process.is_running():
                if force:
                    process.kill()
                    click.echo("Server force stopped")
                else:
                    process.terminate()
                    try:
                        process.wait(timeout=5)  # Wait up to 5 seconds
                        click.echo("Server stopped")
                    except psutil.TimeoutExpired:
                        click.echo("Server not responding. Use --force to force stop.")
                        return
            else:
                click.echo("Server is not running")
            cleanup_files()
        except psutil.NoSuchProcess:
            click.echo("Server is not running")
            cleanup_files()
    except Exception as e:
        click.echo(f"Error stopping server: {str(e)}")
        sys.exit(1)

@cli.command()
@click.argument('output_path', type=click.Path(), default='test_model.joblib')
def create_test_model(output_path):
    """Create a test model for development.
    
    Creates a simple test model that predicts if the second number is greater than the first.
    
    Example:
    mlship create-test-model custom_model.joblib
    """
    try:
        model_path = create_test_model(output_path)
        click.echo(f"Created test model at: {model_path}")
    except Exception as e:
        click.echo(f"Error creating test model: {str(e)}")
        sys.exit(1)

@cli.command()
def ui():
    """Open the web UI in the default browser.
    
    Opens the MLship dashboard in your default web browser.
    The server must be running first.
    """
    try:
        if not os.path.exists(PID_FILE):
            click.echo("Server is not running. Deploy a model first.")
            return

        # Try to read config for port
        port = 8000  # Default port
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE) as f:
                    config = json.load(f)
                    port = config.get('port', 8000)
            except json.JSONDecodeError:
                pass

        url = f"http://localhost:{port}/ui"
        click.echo(f"Opening {url}")
        
        # Open URL in default browser
        import webbrowser
        webbrowser.open(url)
    except Exception as e:
        click.echo(f"Error opening UI: {str(e)}")
        sys.exit(1)

def main():
    cli()