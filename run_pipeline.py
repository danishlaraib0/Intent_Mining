#!/usr/bin/env python3
"""
Production Orchestrator for TnT-LLM Taxonomy Generation.
========================================================
Modular, production-ready pipeline conforming to the paper:
"Text Mining at Scale with Large Language Models" (Wan et al.)

Key Features:
- Fully modular: loaders, samplers, prompt managers, stage handlers, evaluators
- Externalized prompts via config/prompts.yaml (customizable without code changes)
- High-resolution granular taxonomy generation (no broad generic buckets)
- Full per-call and per-run telemetry: latency_ms, tokens, model fallback, stage breakdowns
- Crash-safe checkpointing and resumable execution
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import ConfigManager
from src.data.loader import CorpusLoader
from src.data.sampler import CorpusSampler
from src.phase1.pipeline import Phase1Pipeline
from src.phase2.label_assigner import LabelAssigner
from src.prompts.loader import PromptManager
from src.utils.logger import PipelineLogger
from src.utils.rate_limiter import RateLimitedLLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TnT-LLM Production Taxonomy Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Override sample size from config (e.g. 100 for testing)",
    )
    parser.add_argument(
        "--skip-labeling",
        action="store_true",
        help="Skip Phase 2 label assignment",
    )
    parser.add_argument(
        "--fresh-taxonomy",
        action="store_true",
        help="Regenerate taxonomy trials from scratch (ignore cached trial_*.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 1. Load Configuration
    cfg = ConfigManager.load(args.config)
    if args.sample_size:
        cfg["corpus"]["sample_size"] = args.sample_size
        print(f"⚙️ Overriding sample_size → {args.sample_size}")

    output_dir = Path(cfg["project"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_log_dir = cfg.get("logging", {}).get("dir", "logs")
    log_dir_path = Path(raw_log_dir)
    if log_dir_path.is_absolute():
        log_dir = log_dir_path
    elif str(raw_log_dir).startswith("output/"):
        log_dir = ROOT / raw_log_dir
    else:
        log_dir = output_dir / raw_log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    # 2. Initialize Telemetry Logger
    logger = PipelineLogger(log_dir=log_dir)

    print("\n" + "═" * 70)
    print("🚀 STARTING TNT-LLM TAXONOMY PIPELINE")
    print("═" * 70)
    print(f"  Project:       {cfg['project']['name']}")
    print(f"  Corpus:        {cfg['corpus']['path']}")
    print(f"  Sample Size:   {cfg['corpus'].get('sample_size', 'All')}")
    print(f"  Primary Model: {cfg['llm']['primary']['model']}")
    print(f"  Fallback:      {cfg['llm']['fallback']['model']}")
    print(f"  Log File:      {logger.log_path}")
    print("═" * 70 + "\n")

    try:
        # 3. Initialize Prompt Manager from YAML
        prompts_path = cfg.get("prompts", {}).get("path", "config/prompts.yaml")
        prompts = PromptManager(prompts_path)

        # 4. Initialize Rate-Limited LLM with Logger
        llm = RateLimitedLLM(cfg, logger)

        # 5. Load Corpus & Sample
        all_records = CorpusLoader.load(
            cfg["corpus"]["path"],
            text_field=cfg["corpus"].get("text_field", "question"),
        )
        train_records, val_records = CorpusSampler.sample_and_split(all_records, cfg)

        # 6. Execute Phase 1: Taxonomy Generation
        p1 = Phase1Pipeline(
            cfg=cfg,
            llm=llm,
            prompts=prompts,
            logger=logger,
            output_dir=output_dir,
        )
        best_taxonomy = p1.run(
            train_records,
            val_records,
            fresh_taxonomy=args.fresh_taxonomy,
        )

        # 7. Execute Phase 2: Label Assignment
        if not args.skip_labeling:
            print("\n" + "═" * 60)
            print("PHASE 2: PSEUDO-LABEL ASSIGNMENT")
            print("═" * 60)
            target_records = train_records + val_records
            assigner = LabelAssigner(cfg, llm, prompts, output_dir)
            labeled_records = assigner.run(target_records, best_taxonomy)
        else:
            print("\n⏭️ Skipping Phase 2 label assignment per flag.")

        print("\n🎉 Pipeline Execution Completed Successfully!")

    except KeyboardInterrupt:
        print("\n⚠️  Pipeline interrupted by user. Finalizing telemetry logs...")
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 8. Print and persist complete telemetry summary
        logger.print_summary()


if __name__ == "__main__":
    main()
