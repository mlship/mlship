from pathlib import Path

# Constants for process management
MLSHIP_DIR = Path.home() / '.mlship'
PID_FILE = MLSHIP_DIR / 'server.pid'
METRICS_FILE = MLSHIP_DIR / 'metrics.json'
LOG_FILE = MLSHIP_DIR / 'server.log'

# Ensure directory exists
MLSHIP_DIR.mkdir(exist_ok=True) 