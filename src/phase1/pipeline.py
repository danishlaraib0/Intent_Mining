"""Phase 1 Orchestrator: End-to-End Taxonomy Generation Pipeline."""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.phase1.summarizer import DocumentSummarizer
from src.phase1.taxonomy_evaluator import TaxonomyEvaluator
from src.phase1.taxonomy_generator import TaxonomyGenerator
from src.phase1.taxonomy_reviewer import TaxonomyReviewer
from src.phase1.taxonomy_updater import TaxonomyUpdater
from src.prompts.loader import PromptManager
from src.utils.logger import PipelineLogger
from src.utils.rate_limiter import RateLimitedLLM


class Phase1Pipeline:
    """Coordinates Stage 1 (Summarization) and Stage 2 (Taxonomy Generation & Selection)."""

    def __init__(
        self,
        cfg: Dict[str, Any],
        llm: RateLimitedLLM,
        prompts: PromptManager,
        logger: PipelineLogger,
        output_dir: Path,
    ):
        self.cfg = cfg
        self.llm = llm
        self.prompts = prompts
        self.logger = logger
        self.output_dir = output_dir

        self.summarizer = DocumentSummarizer(cfg, llm, prompts, output_dir)
        self.generator = TaxonomyGenerator(cfg, llm, prompts)
        self.updater = TaxonomyUpdater(cfg, llm, prompts)
        self.reviewer = TaxonomyReviewer(cfg, llm, prompts)
        self.evaluator = TaxonomyEvaluator(cfg, llm, prompts)

    def run(
        self,
        train_records: List[Dict[str, Any]],
        val_records: List[Dict[str, Any]],
        fresh_taxonomy: bool = False,
    ) -> List[Dict[str, str]]:
        """Execute Phase 1 end-to-end."""
        tax_dir = self.output_dir / self.cfg["phase1"]["taxonomy"].get("output_dir", "taxonomies")
        tax_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "═" * 60)
        print("STAGE 1: DOCUMENT SUMMARIZATION")
        print("═" * 60)
        all_to_summarize = train_records + val_records
        summaries_data = self.summarizer.run(all_to_summarize)

        text_field = self.cfg["corpus"].get("text_field", "question")
        text_to_summary = {}
        for item in summaries_data:
            txt = item.get("text") or item.get("question") or item.get(text_field)
            if txt and "summary" in item:
                text_to_summary[txt] = item["summary"]

        train_summaries = [text_to_summary.get(r[text_field], "") for r in train_records if r.get(text_field) in text_to_summary]
        val_summaries = [text_to_summary.get(r[text_field], "") for r in val_records if r.get(text_field) in text_to_summary]

        print(f"\n📊 Prepared {len(train_summaries)} train summaries and {len(val_summaries)} validation summaries.")

        print("\n" + "═" * 60)
        print("STAGE 2: ITERATIVE TAXONOMY GENERATION (SGD-STYLE)")
        print("═" * 60)

        tax_cfg = self.cfg["phase1"]["taxonomy"]
        minibatch_size = tax_cfg.get("minibatch_size", 40)
        num_trials = tax_cfg.get("num_trials", 2)

        candidate_taxonomies: List[List[Dict[str, str]]] = []

        for trial in range(1, num_trials + 1):
            print(f"\n🧪 --- Starting Trial {trial}/{num_trials} ---")

            # Check if trial already exists on disk
            trial_path = tax_dir / f"trial_{trial}.json"
            if not fresh_taxonomy and trial_path.exists():
                print(f"  📂 Loaded existing Trial {trial} from {trial_path}")
                with open(trial_path, "r", encoding="utf-8") as f:
                    candidate_taxonomies.append(json.load(f))
                continue

            # Minibatch partition
            minibatches = [
                train_summaries[i : i + minibatch_size]
                for i in range(0, len(train_summaries), minibatch_size)
            ]
            if not minibatches:
                raise ValueError("No minibatches available from train summaries!")

            # 1. Initial Generation
            print(f"  🌱 Generating initial taxonomy from Batch 1 ({len(minibatches[0])} summaries)...")
            current_taxonomy = self.generator.generate(minibatches[0])

            # 2. Iterative Updates
            for step, mb in enumerate(minibatches[1:], 2):
                print(f"  🔄 SGD Step {step}/{len(minibatches)}: Updating with Batch {step} ({len(mb)} summaries)...")
                current_taxonomy = self.updater.update(current_taxonomy, mb, step=step)

            # 3. Review & Polish
            print("  ✨ Reviewing and polishing taxonomy...")
            final_taxonomy = self.reviewer.review(current_taxonomy)

            # Save trial
            with open(trial_path, "w", encoding="utf-8") as f:
                json.dump(final_taxonomy, f, indent=2, ensure_ascii=False)
            print(f"  💾 Saved Trial {trial} → {trial_path}")
            candidate_taxonomies.append(final_taxonomy)

        # 4. Model Selection using Validation Set
        print("\n" + "═" * 60)
        print("STAGE 2: VALIDATION-BASED BEST TAXONOMY SELECTION")
        print("═" * 60)
        best_idx, best_taxonomy = self.evaluator.evaluate_and_select(candidate_taxonomies, val_summaries)

        best_path = tax_dir / "best_taxonomy.json"
        with open(best_path, "w", encoding="utf-8") as f:
            json.dump(best_taxonomy, f, indent=2, ensure_ascii=False)
        print(f"\n🏆 Best Taxonomy Saved (Selected Trial {best_idx + 1}) → {best_path}")

        print("\n📋 Winning Taxonomy Labels:")
        for idx, item in enumerate(best_taxonomy, 1):
            print(f"  {idx}. {item.get('name')}: {item.get('description')}")

        return best_taxonomy
