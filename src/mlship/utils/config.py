import yaml
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

CONFIG_DIR = Path.home() / '.mlship'
CONFIG_FILE = CONFIG_DIR / 'config.yaml'

def init_config():
    CONFIG_DIR.mkdir(exist_ok=True)
    if not CONFIG_FILE.exists():
        with open(CONFIG_FILE, 'w') as f:
            yaml.dump({}, f)

def load_config():
    init_config()
    try:
        with open(CONFIG_FILE) as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load config: {str(e)}")
        return {}

def save_config(config):
    init_config()
    try:
        with open(CONFIG_FILE, 'w') as f:
            yaml.dump(config, f)
    except Exception as e:
        logger.error(f"Failed to save config: {str(e)}")
        raise