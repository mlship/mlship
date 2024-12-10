import click
import os
import sys
import json
import time
import logging
import subprocess
from pathlib import Path
from .utils.constants import PID_FILE, METRICS_FILE, LOG_FILE, CONFIG_FILE
from .server.loader import ModelLoader
from .server.app import ModelServer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def write_pid_file(pid=None):
    """Write PID to file"""
    with open(PID_FILE, 'w') as f:
        f.write(str(pid or os.getpid()))

def read_pid_file():
    """Read PID from file"""
    try:
        with open(PID_FILE) as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError):
        return None

def cleanup_files():
    """Clean up state files"""
    for file in [PID_FILE, METRICS_FILE]:
        if os.path.exists(file):
            os.remove(file)

def start_daemon(model_path, port, ui):
    """Start server in daemon mode"""
    # Prepare command
    cmd = [
        sys.executable, "-m", "mlship", "deploy",
        model_path,
        "--port", str(port),
        "--foreground"  # Internal flag
    ]
    if ui:
        cmd.append("--ui")
        
    # Prepare environment
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Start process
    with open(LOG_FILE, 'w') as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,  # Daemonize
            preexec_fn=os.setpgrp  # Detach from terminal
        )
        
    # Wait briefly to ensure process starts
    time.sleep(1)
    if process.poll() is not None:
        raise click.ClickException("Failed to start daemon process")
        
    return process.pid

@click.group()
def cli():
    """MLShip - Simple ML Model Deployment"""
    pass

@cli.command()
@click.argument('model_path')
@click.option('--daemon', is_flag=True, help='Run in daemon mode')
@click.option('--port', default=8000, help='Port to run the server on')
@click.option('--ui', is_flag=True, help='Enable web interface')
@click.option('--foreground', is_flag=True, hidden=True, help='Internal flag for daemon mode')
def deploy(model_path, daemon, port, ui, foreground):
    """Deploy a model"""
    try:
        # Convert model path to absolute path if relative
        if not os.path.isabs(model_path):
            model_path = os.path.abspath(model_path)
        
        # Check if model exists
        if not os.path.exists(model_path):
            click.echo(f"Model file not found: {model_path}", err=True)
            sys.exit(1)
        
        # Check if server is already running
        if os.path.exists(PID_FILE):
            click.echo("Server is already running", err=True)
            sys.exit(1)
            
        if daemon and not foreground:
            # Start daemon process
            pid = start_daemon(model_path, port, ui)
            write_pid_file(pid)
            
            # Create initial metrics file
            with open(METRICS_FILE, 'w') as f:
                json.dump({
                    'start_time': time.time(),
                    'requests': 0,
                    'avg_latency': 0
                }, f)
                
            # Show URLs
            click.echo(f"🚀 API: http://localhost:{port}")
            if ui:
                click.echo(f"🎨 UI: http://localhost:{port}/ui")
        else:
            # Run in foreground
            try:
                model = ModelLoader.load(model_path)
                server = ModelServer(model)
                
                # Save PID and initialize metrics
                write_pid_file()
                with open(METRICS_FILE, 'w') as f:
                    json.dump({
                        'start_time': time.time(),
                        'requests': 0,
                        'avg_latency': 0
                    }, f)
                    
                if not foreground:
                    click.echo(f"🚀 API: http://localhost:{port}")
                    if ui:
                        click.echo(f"🎨 UI: http://localhost:{port}/ui")
                        
                server.serve(host="0.0.0.0", port=port)
            except Exception as e:
                cleanup_files()
                raise click.ClickException(str(e))
            
    except Exception as e:
        cleanup_files()
        logger.error(f"Deployment failed: {str(e)}")
        click.echo(str(e), err=True)
        sys.exit(1)

@cli.command()
def stop():
    """Stop the server"""
    try:
        if not os.path.exists(PID_FILE):
            click.echo("No server running")
            return
            
        cleanup_files()
        click.echo("Server stopped")
            
    except Exception as e:
        logger.error(f"Failed to stop server: {str(e)}")
        click.echo(str(e), err=True)
        sys.exit(1)

@cli.command()
def status():
    """Check deployment status"""
    try:
        if not os.path.exists(PID_FILE):
            click.echo("Server not running")
            return
            
        # Get metrics if available
        if os.path.exists(METRICS_FILE):
            with open(METRICS_FILE) as f:
                metrics = json.load(f)
                uptime = int((time.time() - metrics['start_time']) / 60)
                click.echo("Server running")
                click.echo(f"Uptime: {uptime}m")
                click.echo(f"Requests: {metrics['requests']}")
                click.echo(f"Avg latency: {metrics['avg_latency']}ms")
        else:
            click.echo("Server running")
            
    except Exception as e:
        logger.error(f"Failed to get status: {str(e)}")
        click.echo(str(e), err=True)
        sys.exit(1)

@cli.command()
def logs():
    """View server logs"""
    try:
        if not os.path.exists(LOG_FILE):
            click.echo("No logs available")
            return
            
        with open(LOG_FILE) as f:
            content = f.read()
            if not content:
                click.echo("No logs available")
            else:
                click.echo(content)
            
    except Exception as e:
        logger.error(f"Failed to read logs: {str(e)}")
        click.echo(str(e), err=True)
        sys.exit(1)

@cli.command()
@click.option('--aws-key', prompt=True, help='AWS Access Key ID')
@click.option('--aws-secret', prompt=True, hide_input=True, help='AWS Secret Access Key')
def configure(aws_key, aws_secret):
    """Configure AWS credentials"""
    try:
        config = {'aws_key': aws_key, 'aws_secret': aws_secret}
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f)
        click.echo("Configuration saved successfully")
            
    except Exception as e:
        logger.error(f"Configuration failed: {str(e)}")
        click.echo(str(e), err=True)
        sys.exit(1)

@cli.command()
@click.option('--output', default='test_model.pkl', help='Output path for test model')
def create_model(output):
    """Create a test model for deployment"""
    try:
        from .utils.create_test_model import create_test_model
        model_path = create_test_model(output)
        click.echo(f"✓ Test model created at: {model_path}")
        click.echo(f"Run 'mlship deploy {output}' to deploy the model")
    except Exception as e:
        logger.error(f"Failed to create test model: {str(e)}")
        click.echo(str(e), err=True)
        sys.exit(1)