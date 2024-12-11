import os
import sys
import atexit
import signal
import logging
import json
import platform
from pathlib import Path
from .constants import PID_FILE, LOG_FILE, METRICS_FILE

logger = logging.getLogger(__name__)

def cleanup_files():
    """Clean up PID and metrics files."""
    try:
        if os.path.exists(PID_FILE):
            os.remove(PID_FILE)
        if os.path.exists(METRICS_FILE):
            os.remove(METRICS_FILE)
    except Exception as e:
        logger.error(f"Error cleaning up files: {str(e)}")

def daemonize(model_path: str, host: str, port: int, ui: bool):
    """Daemonize the server process."""
    try:
        # For macOS, use subprocess instead of fork
        if platform.system() == 'Darwin':
            import subprocess
            import psutil
            
            # Get the path to the current Python executable
            python_path = sys.executable
            
            # Prepare command to run the server
            cmd = [
                python_path,
                "-c",
                f"""
import os
import logging
from mlship.server.app import start_server
from mlship.utils.constants import PID_FILE, LOG_FILE

# Configure logging
logging.basicConfig(
    filename='{LOG_FILE}',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Write PID file
with open('{PID_FILE}', 'w') as f:
    f.write(str(os.getpid()))

# Start server
start_server('{model_path}', host='{host}', port={port}, ui={ui})
                """
            ]
            
            # Start process
            with open(LOG_FILE, 'a') as log_file:
                process = subprocess.Popen(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    start_new_session=True
                )
            
            # Give the process time to start
            try:
                psutil.Process(process.pid).wait(timeout=1)
            except psutil.TimeoutExpired:
                # Process is still running, which is good
                pass
            
            return
            
        # For other platforms, use traditional fork
        # First fork (detaches from parent)
        try:
            pid = os.fork()
            if pid > 0:
                # Parent process exits
                sys.exit(0)
        except OSError as err:
            logger.error(f'Fork #1 failed: {err}')
            sys.exit(1)

        # Decouple from parent environment
        os.chdir('/')  # Change working directory
        os.umask(0)    # Reset file creation mask
        os.setsid()    # Create new session

        # Second fork (relinquish session leadership)
        try:
            pid = os.fork()
            if pid > 0:
                # Parent process exits
                sys.exit(0)
        except OSError as err:
            logger.error(f'Fork #2 failed: {err}')
            sys.exit(1)

        # Flush I/O buffers and redirect standard file descriptors
        sys.stdout.flush()
        sys.stderr.flush()

        # Open log file
        log_file = open(LOG_FILE, 'a+')
        os.dup2(log_file.fileno(), sys.stdout.fileno())
        os.dup2(log_file.fileno(), sys.stderr.fileno())

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )

        # Write PID file
        with open(PID_FILE, 'w') as f:
            f.write(str(os.getpid()))

        # Register cleanup on exit
        atexit.register(cleanup_files)

        # Handle signals
        signal.signal(signal.SIGTERM, lambda signo, frame: cleanup_and_exit())
        signal.signal(signal.SIGINT, lambda signo, frame: cleanup_and_exit())

        # Import and start server
        from ..server.app import start_server
        logger.info(f"Starting server with model: {model_path}")
        start_server(model_path, host=host, port=port, ui=ui)

    except Exception as e:
        logger.error(f"Daemon failed: {str(e)}")
        cleanup_files()
        sys.exit(1)

def cleanup_and_exit(signo=None, frame=None):
    """Clean up and exit gracefully."""
    cleanup_files()
    sys.exit(0)

# For Windows compatibility
if os.name == 'nt':
    def daemonize(model_path: str, host: str, port: int, ui: bool):
        """Windows version of daemonize (runs in foreground)."""
        try:
            # Configure logging
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(LOG_FILE),
                    logging.StreamHandler()
                ]
            )

            # Write PID file
            with open(PID_FILE, 'w') as f:
                f.write(str(os.getpid()))

            # Register cleanup on exit
            atexit.register(cleanup_files)

            # Import and start server
            from ..server.app import start_server
            logger.info(f"Starting server with model: {model_path}")
            start_server(model_path, host=host, port=port, ui=ui)

        except Exception as e:
            logger.error(f"Server failed: {str(e)}")
            cleanup_files()
            sys.exit(1) 