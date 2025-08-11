"""Stage 1: Document/Question Summarizer with Resume Support and Batching."""

import json
from pathlib import Path
from typing import Any, Dict, List

from src.prompts.loader import PromptManager
from src.utils.logger import PipelineLogger
from src.utils.rate_limiter import RateLimitedLLM


class DocumentSummarizer:
    """Summarizes text records using LLM with checkpointing and rate limiting."""

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

        self.sum_cfg = cfg["phase1"]["summarization"]
        self.target_length = self.sum_cfg.get("target_length_words", 15)
        self.output_file = output_dir / self.sum_cfg.get("output_file", "summaries.jsonl")
        self.batch_size = cfg["llm"]["rate_limits"].get("batch_size", 10)
        self.temperature = cfg["llm"]["primary"].get("temperature_summarize", 0.2)

    def run(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Summarize records, resuming from existing checkpoint if present."""
        text_field = self.cfg["corpus"].get("text_field", "question")

        # Load existing summaries for resume support
        existing_summaries: Dict[str, str] = {}
        if self.output_file.exists():
            with open(self.output_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            item = json.loads(line)
                            txt = item.get("text") or item.get(text_field) or item.get("question")
                            if txt and "summary" in item:
                                existing_summaries[txt] = item["summary"]
                        except json.JSONDecodeError:
                            pass
            print(f"🔄 [Summarizer] Found {len(existing_summaries)} existing summaries (resuming)")

        results: List[Dict[str, Any]] = []
        pending_items: List[Dict[str, Any]] = []

        for r in records:
            txt = r.get(text_field, "").strip()
            if not txt:
                continue
            if txt in existing_summaries:
                results.append({text_field: txt, "question": txt, "text": txt, "summary": existing_summaries[txt]})
            else:
                pending_items.append({text_field: txt, "question": txt, "text": txt})

        print(f"🚀 [Summarizer] Total to summarize: {len(pending_items)} (Already done: {len(results)})")

        if not pending_items:
            print("✅ [Summarizer] All summaries already exist on disk.")
            return results

        # Process in batches
        for i in range(0, len(pending_items), self.batch_size):
            batch = pending_items[i : i + self.batch_size]
            batch_num = (i // self.batch_size) + 1
            total_batches = (len(pending_items) + self.batch_size - 1) // self.batch_size

            print(f"  ⚡ Summarizing batch {batch_num}/{total_batches} ({len(batch)} items)...")

            for item in batch:
                txt = item["text"]
                rendered = self.prompts.render(
                    "summarize",
                    target_length=self.target_length,
                    question=txt,
                )

                try:
                    summary = self.llm.call(
                        rendered["user"],
                        temperature=self.temperature,
                        system_prompt=rendered["system"],
                        stage="summarize",
                    )
                    summary = summary.strip().strip('"')
                except Exception as e:
                    print(f"  ❌ Error summarizing '{txt[:40]}': {e}")
                    summary = txt[:60]  # Fallback

                entry = {"text": txt, "question": txt, text_field: txt, "summary": summary}
                results.append(entry)

                # Append immediately to disk (checkpointing)
                with open(self.output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            # Cooldown between batches if there are more
            if i + self.batch_size < len(pending_items):
                self.llm.batch_delay()

        print(f"✅ [Summarizer] Completed all {len(results)} summaries → {self.output_file}")
        return results
