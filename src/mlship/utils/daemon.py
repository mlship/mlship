import os
import sys
import json
import time
import signal
import subprocess
import click
import logging
import psutil
from pathlib import Path
from .constants import PID_FILE, METRICS_FILE, LOG_FILE, MLSHIP_DIR

logger = logging.getLogger(__name__)

def cleanup_files():
    """Clean up all temporary files"""
    for file in [PID_FILE, METRICS_FILE, LOG_FILE]:
        try:
            if os.path.exists(file):
                os.remove(file)
        except Exception as e:
            logger.warning(f"Failed to remove {file}: {str(e)}")

def save_pid(pid):
    """Save process ID to file"""
    with open(PID_FILE, 'w') as f:
        f.write(str(pid))

def get_pid():
    """Get saved process ID"""
    try:
        with open(PID_FILE) as f:
            return int(f.read().strip())
    except FileNotFoundError:
        return None

def is_process_running(pid):
    """Check if process is running"""
    try:
        process = psutil.Process(pid)
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return False

def update_metrics(requests=0, latency=0):
    """Update server metrics"""
    metrics = {
        'start_time': time.time(),
        'requests': requests,
        'avg_latency': latency
    }
    with open(METRICS_FILE, 'w') as f:
        json.dump(metrics, f)

def get_metrics():
    """Get server metrics"""
    try:
        with open(METRICS_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        return None

def verify_server_running(port):
    """Verify if server is actually running and responding"""
    import socket
    try:
        with socket.create_connection(("localhost", port), timeout=5):
            return True
    except (socket.timeout, ConnectionRefusedError):
        return False

def start_daemon(model_path, host, port):
    """Start server as a daemon"""
    # Clean up any existing files
    cleanup_files()
    
    # Get the path to the current Python executable and entry point
    python_path = sys.executable
    mlship_script = os.path.join(os.path.dirname(python_path), 'mlship')
    
    # Convert model path to absolute path
    model_path = os.path.abspath(model_path)
    
    # Prepare command to run the server
    cmd = [
        mlship_script,
        'deploy',
        model_path,
        '--host', host,
        '--port', str(port),
        '--foreground'
    ]
    
    # Start the process
    try:
        # Create log file and directory if they don't exist
        MLSHIP_DIR.mkdir(parents=True, exist_ok=True)
        log_file = open(LOG_FILE, 'w+')
        
        # Get current working directory
        cwd = os.getcwd()
        
        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            close_fds=True,
            start_new_session=True,  # This creates a new process group
            cwd=cwd  # Use current working directory
        )
        
        # Give the process time to start
        max_retries = 10
        for i in range(max_retries):
            time.sleep(1)  # Wait a bit longer between checks
            
            # Check if process is still running
            if not psutil.pid_exists(process.pid):
                # Process died, check logs for error
                log_file.seek(0)
                error_log = log_file.read()
                log_file.close()
                raise click.ClickException(f"Server failed to start. Logs:\n{error_log}")
            
            # Check if server is responding
            if verify_server_running(port):
                # Save PID and initialize metrics
                save_pid(process.pid)
                update_metrics()
                return process.pid
                
        # If we get here, server didn't start properly
        process.terminate()
        log_file.seek(0)
        error_log = log_file.read()
        log_file.close()
        raise click.ClickException(f"Server failed to respond after {max_retries} seconds. Logs:\n{error_log}")
        
    except Exception as e:
        # Clean up if anything goes wrong
        if 'process' in locals():
            try:
                process.terminate()
            except:
                pass
        if 'log_file' in locals():
            log_file.close()
        raise click.ClickException(f"Failed to start server: {str(e)}")

def stop_daemon():
    """Stop the daemon process"""
    pid = get_pid()
    if not pid:
        click.echo("No server running")
        return
        
    try:
        # Get the process and its children
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)
        
        # Send SIGTERM to parent and children
        for p in [parent] + children:
            try:
                p.terminate()
            except psutil.NoSuchProcess:
                pass
        
        # Wait for processes to terminate
        gone, alive = psutil.wait_procs([parent] + children, timeout=3)
        
        # If any processes are still alive, kill them
        for p in alive:
            try:
                p.kill()
            except psutil.NoSuchProcess:
                pass
        
        # Clean up files
        for file in [PID_FILE, METRICS_FILE]:
            if os.path.exists(file):
                os.remove(file)
                
        click.echo("Server stopped")
    except psutil.NoSuchProcess:
        # Clean up files if process not found
        for file in [PID_FILE, METRICS_FILE]:
            if os.path.exists(file):
                os.remove(file)
        click.echo("Server was not running")
    except Exception as e:
        logger.error(f"Failed to stop server: {str(e)}")
        raise click.ClickException(str(e))

def view_logs():
    """View server logs"""
    try:
        with open(LOG_FILE) as f:
            content = f.read()
            if not content:
                click.echo("No logs available (file is empty)")
            else:
                click.echo(content)
    except FileNotFoundError:
        click.echo("No logs available (file not found)") 