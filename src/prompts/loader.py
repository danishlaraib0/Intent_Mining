"""Prompt template manager.
Loads externalized YAML prompt templates and renders them with parameters.
"""

from pathlib import Path
from typing import Any, Dict, Optional
import yaml


class PromptManager:
    """Loads prompt templates from YAML and formats them dynamically."""

    def __init__(self, prompt_file: Optional[str] = None):
        if prompt_file:
            path = Path(prompt_file)
        else:
            # Look in config/prompts.yaml or src/prompts/prompts.yaml
            root = Path(__file__).resolve().parent.parent.parent
            path = root / "config" / "prompts.yaml"
            if not path.exists():
                path = root / "src" / "prompts" / "prompts.yaml"

        if not path.exists():
            raise FileNotFoundError(f"Prompts template file not found at: {path}")

        with open(path, "r", encoding="utf-8") as f:
            self.templates = yaml.safe_load(f)

    def get(self, stage: str) -> Dict[str, str]:
        """Return the dictionary containing system and user prompt for a stage."""
        if stage not in self.templates:
            raise KeyError(f"Stage '{stage}' not defined in prompts YAML. Available: {list(self.templates.keys())}")
        return self.templates[stage]

    def render(self, stage: str, **kwargs: Any) -> Dict[str, str]:
        """Format both system and user prompts with kwargs."""
        stage_templates = self.get(stage)
        system_tmpl = stage_templates.get("system", "")
        user_tmpl = stage_templates.get("user", "")

        rendered_system = system_tmpl.format(**kwargs) if system_tmpl else None
        rendered_user = user_tmpl.format(**kwargs) if user_tmpl else ""

        return {
            "system": rendered_system,
            "user": rendered_user,
        }
