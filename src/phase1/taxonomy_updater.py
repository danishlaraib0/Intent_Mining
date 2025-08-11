"""Stage 2: Iterative Taxonomy Updater (SGD minibatches)."""

import json
from typing import Any, Dict, List

from src.prompts.loader import PromptManager
from src.utils.parsers import OutputParser
from src.utils.rate_limiter import RateLimitedLLM


class TaxonomyUpdater:
    """Iteratively updates and refines a candidate taxonomy over minibatches."""

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
        self.temperature = cfg["llm"]["primary"].get("temperature_update", 0.2)

    def update(
        self,
        current_taxonomy: List[Dict[str, str]],
        batch_summaries: List[str],
        step: int = 1,
    ) -> List[Dict[str, str]]:
        """Run an update step using a new batch of summaries."""
        formatted_summaries = "\n".join(f"- {s}" for s in batch_summaries)
        tax_str = json.dumps(current_taxonomy, indent=2)

        rendered = self.prompts.render(
            "update",
            use_case_instruction=self.use_case_instruction,
            current_taxonomy=tax_str,
            summaries=formatted_summaries,
            num_labels=self.num_labels,
            max_label_words=self.max_label_words,
            max_desc_words=self.max_desc_words,
        )

        resp = self.llm.call(
            rendered["user"],
            temperature=self.temperature,
            system_prompt=rendered["system"],
            stage="taxonomy_update",
        )

        try:
            updated = OutputParser.parse_json(resp)
            print(f"  🔄 Step {step}: Taxonomy updated successfully ({len(updated)} labels)")
            return updated
        except Exception as e:
            print(f"  ⚠️  Step {step}: Update parse failed ({e}), retaining previous version")
            return current_taxonomy
