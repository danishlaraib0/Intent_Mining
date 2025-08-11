"""Configuration manager and validator for TnT-LLM."""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


class ConfigManager:
    """Loads and validates pipeline configuration files."""

    DEFAULT_CONFIG_PATH = "config.yaml"

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load YAML configuration with environment variable support."""
        if config_path:
            path = Path(config_path)
        else:
            root = Path(__file__).resolve().parent.parent
            path = root / cls.DEFAULT_CONFIG_PATH

        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        cls._validate(cfg)
        return cfg

    @classmethod
    def _validate(cls, cfg: Dict[str, Any]) -> None:
        """Validate critical sections in configuration."""
        required_keys = ["project", "corpus", "llm", "phase1"]
        for key in required_keys:
            if key not in cfg:
                raise KeyError(f"Missing required section in config: '{key}'")

        # Validate corpus
        if "path" not in cfg["corpus"]:
            raise KeyError("Corpus path must be specified under corpus.path")

        # Validate LLM
        if "primary" not in cfg["llm"]:
            raise KeyError("Primary LLM configuration must be provided under llm.primary")


def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    """Top-level functional interface."""
    return ConfigManager.load(path)
