"""Phase 2: LLM-based Pseudo-Labeling against Taxonomy."""

import json
from pathlib import Path
from typing import Any, Dict, List

from src.prompts.loader import PromptManager
from src.utils.parsers import OutputParser
from src.utils.rate_limiter import RateLimitedLLM


class LabelAssigner:
    """Classifies questions into winning taxonomy categories using LLM with batching."""

    def __init__(
        self,
        cfg: Dict[str, Any],
        llm: RateLimitedLLM,
        prompts: PromptManager,
        output_dir: Path,
    ):
        self.cfg = cfg
        self.llm = llm
        self.prompts = prompts
        self.output_dir = output_dir

        self.p2_cfg = cfg.get("phase2", {}).get("label_assignment", {})
        self.output_file = output_dir / self.p2_cfg.get("output_file", "labeled_questions.jsonl")
        self.batch_size = cfg["llm"]["rate_limits"].get("batch_size", 10)
        self.temperature = 0.0  # Deterministic classification

    def run(
        self,
        records: List[Dict[str, Any]],
        taxonomy: List[Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        """Label questions, resuming from existing checkpoint if available."""
        tax_str = json.dumps(taxonomy, indent=2)
        text_field = self.cfg["corpus"].get("text_field", "question")

        # Resume support
        existing_labeled: Dict[str, Dict[str, Any]] = {}
        if self.output_file.exists():
            with open(self.output_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            item = json.loads(line)
                            existing_labeled[item["question"]] = item
                        except json.JSONDecodeError:
                            pass
            print(f"🔄 [LabelAssigner] Found {len(existing_labeled)} previously labeled questions")

        results: List[Dict[str, Any]] = []
        pending: List[Dict[str, Any]] = []

        for r in records:
            q = r.get(text_field, "").strip()
            if not q:
                continue
            if q in existing_labeled:
                results.append(existing_labeled[q])
            else:
                pending.append(r)

        if not pending:
            print("✅ [LabelAssigner] All questions already labeled.")
            return results

        print(f"🏷️  [LabelAssigner] Labeling {len(pending)} questions ({len(results)} already cached)...")

        for i in range(0, len(pending), self.batch_size):
            batch = pending[i : i + self.batch_size]
            batch_num = (i // self.batch_size) + 1
            total_batches = (len(pending) + self.batch_size - 1) // self.batch_size

            print(f"  ⚡ Labeling batch {batch_num}/{total_batches} ({len(batch)} items)...")

            for item in batch:
                q = item.get(text_field, "").strip()
                rendered = self.prompts.render(
                    "assign",
                    question=q,
                    taxonomy=tax_str,
                )

                try:
                    resp = self.llm.call(
                        rendered["user"],
                        temperature=self.temperature,
                        system_prompt=rendered["system"],
                        stage="label_assignment",
                    )
                    parsed = OutputParser.parse_json(resp)
                    entry = {
                        "question": q,
                        "label": parsed.get("primary_label", "Unclassified"),
                        "confidence": parsed.get("confidence", "unknown"),
                        "reasoning": parsed.get("reasoning", ""),
                    }
                except Exception as e:
                    print(f"  ❌ Error labeling '{q[:40]}': {e}")
                    entry = {
                        "question": q,
                        "label": "Error",
                        "confidence": "none",
                        "reasoning": str(e)[:100],
                    }

                results.append(entry)

                with open(self.output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            if i + self.batch_size < len(pending):
                self.llm.batch_delay()

        print(f"✅ [LabelAssigner] Completed labeling {len(results)} questions → {self.output_file}")
        return results
