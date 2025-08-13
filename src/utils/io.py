import yaml
from pathlib import Path
from typing import Any, Dict

class ConfigLoadError(Exception):
    """Raised when loading YAML config fails."""
    pass

def load_yaml(path: str | Path) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise ConfigLoadError(f"Config file not found: {path}")
    except yaml.YAMLError as e:
        raise ConfigLoadError(f"YAML parse error in {path}: {e}")
    except Exception as e:
        raise ConfigLoadError(f"Error loading config {path}: {e}")
