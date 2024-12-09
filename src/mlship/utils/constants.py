from pathlib import Path

# Create .mlship directory in user's home
MLSHIP_DIR = Path.home() / '.mlship'
MLSHIP_DIR.mkdir(parents=True, exist_ok=True)

# File paths
PID_FILE = MLSHIP_DIR / 'server.pid'
METRICS_FILE = MLSHIP_DIR / 'metrics.json'
LOG_FILE = MLSHIP_DIR / 'server.log'
CONFIG_FILE = MLSHIP_DIR / 'config.json' 