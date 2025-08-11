"""Stage 2: Initial Taxonomy Generator."""

import json
from typing import Any, Dict, List

from src.prompts.loader import PromptManager
from src.utils.parsers import OutputParser
from src.utils.rate_limiter import RateLimitedLLM


class TaxonomyGenerator:
    """Generates the initial candidate taxonomy from a minibatch of summaries."""

    def __init__(
        self,
        cfg: Dict[str, Any],
        llm: RateLimitedLLM,
        prompts: PromptManager,
    ):
        self.cfg = cfg
        self.llm = llm
        self.prompts = prompts

        self.tax_cfg = cfg["phase1"]["taxonomy"]
        self.num_labels = self.tax_cfg.get("num_labels", 10)
        self.max_label_words = self.tax_cfg.get("max_label_words", 5)
        self.max_desc_words = self.tax_cfg.get("max_desc_words", 40)
        self.use_case_instruction = cfg["phase1"]["use_case"]["instruction"].strip()
        self.temperature = cfg["llm"]["primary"].get("temperature_generate", 0.5)

    def generate(self, batch_summaries: List[str]) -> List[Dict[str, str]]:
        """Generate initial taxonomy from the first minibatch."""
        formatted_summaries = "\n".join(f"- {s}" for s in batch_summaries)

        rendered = self.prompts.render(
            "generate",
            batch_size=len(batch_summaries),
            num_labels=self.num_labels,
            use_case_instruction=self.use_case_instruction,
            max_label_words=self.max_label_words,
            max_desc_words=self.max_desc_words,
            summaries=formatted_summaries,
        )

        resp = self.llm.call(
            rendered["user"],
            temperature=self.temperature,
            system_prompt=rendered["system"],
            stage="taxonomy_generate",
        )

        taxonomy = OutputParser.parse_json(resp)
        print(f"  🌱 Generated initial taxonomy with {len(taxonomy)} labels")
        return taxonomy
