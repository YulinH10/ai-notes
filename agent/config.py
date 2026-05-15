import os
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "agent_config.yaml"


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["api_key"] = os.environ.get("ANTHROPIC_API_KEY", "")
    return config
