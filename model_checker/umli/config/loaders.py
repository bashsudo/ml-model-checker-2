"""
Configuration Loaders

Utilities for loading and saving configurations.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_config(path: str, format: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from file.

    Args:
        path: Path to configuration file
        format: Optional format specification ('json', 'yaml', 'yml')

    Returns:
        Configuration dictionary
    """
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    if format is None:
        format = _infer_format_from_path(path)

    with open(path, "r") as f:
        if format in ("yaml", "yml"):
            return yaml.safe_load(f)
        elif format == "json":
            return json.load(f)
        else:
            raise ValueError(f"Unsupported configuration format: {format}")


def save_config(
    config: Dict[str, Any], path: str, format: Optional[str] = None
) -> None:
    """
    Save configuration to file.

    Args:
        config: Configuration dictionary
        path: Path to save configuration file
        format: Optional format specification ('json', 'yaml', 'yml')
    """
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    if format is None:
        format = _infer_format_from_path(path)

    with open(path, "w") as f:
        if format in ("yaml", "yml"):
            yaml.dump(config, f, default_flow_style=False)
        elif format == "json":
            json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported configuration format: {format}")


def _infer_format_from_path(path: str) -> str:
    """Infer configuration format from file path extension."""
    path_lower = path.lower()
    if path_lower.endswith(".yaml") or path_lower.endswith(".yml"):
        return "yaml"
    elif path_lower.endswith(".json"):
        return "json"
    else:
        return "json"  # Default to JSON


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration against UnifiedConfig schema.

    Args:
        config: Configuration dictionary

    Returns:
        True if configuration is valid
    """
    required_keys = ["model", "data", "training"]

    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")

    # Validate model config
    if "type" not in config["model"]:
        raise ValueError("Model configuration must include 'type'")

    # Validate data config
    if "type" not in config["data"]:
        raise ValueError("Data configuration must include 'type'")

    # Validate training config
    if "epochs" not in config["training"]:
        config["training"]["epochs"] = 1
    if "batch_size" not in config["training"]:
        config["training"]["batch_size"] = 32

    return True
