"""Stage 2: Taxonomy Reviewer and Auditor."""

import json
from typing import Any, Dict, List

from src.prompts.loader import PromptManager
from src.utils.parsers import OutputParser
from src.utils.rate_limiter import RateLimitedLLM


class TaxonomyReviewer:
    """Audits, cleans, deduplicates, and polishes the taxonomy."""

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
        self.temperature = cfg["llm"]["primary"].get("temperature_review", 0.0)

    def review(self, taxonomy: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Perform final quality audit and standardization."""
        tax_str = json.dumps(taxonomy, indent=2)

        rendered = self.prompts.render(
            "review",
            use_case_instruction=self.use_case_instruction,
            taxonomy=tax_str,
            num_labels=self.num_labels,
            max_label_words=self.max_label_words,
            max_desc_words=self.max_desc_words,
        )

        resp = self.llm.call(
            rendered["user"],
            temperature=self.temperature,
            system_prompt=rendered["system"],
            stage="taxonomy_review",
        )

        try:
            reviewed = OutputParser.parse_json(resp)
            print(f"  ✨ Taxonomy review complete ({len(reviewed)} labels)")
            return reviewed
        except Exception as e:
            print(f"  ⚠️  Review parsing failed ({e}), retaining unreviewed taxonomy")
            return taxonomy
