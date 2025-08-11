"""Configuration manager and validator for TnT-LLM.
Dynamically resolves all relative paths so the project is 100% portable across machines.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


class ConfigManager:
    """Loads, resolves relative paths, and validates pipeline configuration files."""

    DEFAULT_CONFIG_PATH = "config.yaml"

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load YAML configuration, resolve all relative paths to absolute paths dynamically."""
        if config_path:
            cfg_file = Path(config_path).resolve()
        else:
            root = Path(__file__).resolve().parent.parent
            cfg_file = (root / cls.DEFAULT_CONFIG_PATH).resolve()

        if not cfg_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {cfg_file}")

        with open(cfg_file, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        # Base directory is the project directory containing config.yaml
        project_root = cfg_file.parent

        # 1. Dynamically resolve project directories
        if "project" in cfg:
            cfg["project"]["root_dir"] = str(project_root)
            if "data_dir" in cfg["project"]:
                data_p = Path(cfg["project"]["data_dir"])
                cfg["project"]["data_dir"] = str(data_p if data_p.is_absolute() else (project_root / data_p).resolve())
            if "output_dir" in cfg["project"]:
                out_p = Path(cfg["project"]["output_dir"])
                cfg["project"]["output_dir"] = str(out_p if out_p.is_absolute() else (project_root / out_p).resolve())

        # 2. Dynamically resolve corpus path
        if "corpus" in cfg and "path" in cfg["corpus"]:
            corp_p = Path(cfg["corpus"]["path"])
            cfg["corpus"]["path"] = str(corp_p if corp_p.is_absolute() else (project_root / corp_p).resolve())

        # 3. Dynamically resolve prompts path
        if "prompts" in cfg and "path" in cfg["prompts"]:
            pmt_p = Path(cfg["prompts"]["path"])
            cfg["prompts"]["path"] = str(pmt_p if pmt_p.is_absolute() else (project_root / pmt_p).resolve())

        # 4. Dynamically resolve logging directory
        if "logging" in cfg and "dir" in cfg["logging"]:
            log_p = Path(cfg["logging"]["dir"])
            if not log_p.is_absolute():
                out_dir = Path(cfg.get("project", {}).get("output_dir", project_root / "output"))
                if str(log_p).startswith("output"):
                    cfg["logging"]["dir"] = str((project_root / log_p).resolve())
                else:
                    cfg["logging"]["dir"] = str((out_dir / log_p).resolve())

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
