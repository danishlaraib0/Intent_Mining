"""Stage 2: Validation-based Taxonomy Model Selection."""

import json
import re
from typing import Any, Dict, List, Tuple

from src.prompts.loader import PromptManager
from src.utils.rate_limiter import RateLimitedLLM


class TaxonomyEvaluator:
    """Evaluates candidate taxonomies on a validation set to select the optimal candidate."""

    def __init__(
        self,
        cfg: Dict[str, Any],
        llm: RateLimitedLLM,
        prompts: PromptManager,
    ):
        self.cfg = cfg
        self.llm = llm
        self.prompts = prompts
        self.use_case_instruction = cfg["phase1"]["use_case"]["instruction"].strip()

    def evaluate_and_select(
        self,
        candidate_taxonomies: List[List[Dict[str, str]]],
        val_summaries: List[str],
    ) -> Tuple[int, List[Dict[str, str]]]:
        """Judge which candidate taxonomy best covers the validation set."""
        if len(candidate_taxonomies) == 1:
            print("  ℹ️ Only 1 candidate taxonomy generated; auto-selecting Trial 1")
            return 0, candidate_taxonomies[0]

        formatted_val = "\n".join(f"- {s}" for s in val_summaries[:20])

        options_text = ""
        for idx, tax in enumerate(candidate_taxonomies, 1):
            options_text += f"\n--- TAXONOMY OPTION {idx} ---\n"
            options_text += json.dumps(tax, indent=2) + "\n"

        rendered = self.prompts.render(
            "evaluate",
            use_case_instruction=self.use_case_instruction,
            summaries=formatted_val,
            taxonomy_options=options_text,
        )

        resp = self.llm.call(
            rendered["user"],
            temperature=0.0,
            system_prompt=rendered["system"],
            stage="evaluate_selection",
        )

        match = re.search(r"\b([1-9])\b", resp)
        if match:
            choice = int(match.group(1)) - 1
            if 0 <= choice < len(candidate_taxonomies):
                print(f"  🏆 Evaluator selected Taxonomy #{choice + 1}")
                return choice, candidate_taxonomies[choice]

        print("  ⚠️  Evaluator output inconclusive, defaulting to Candidate #1")
        return 0, candidate_taxonomies[0]
