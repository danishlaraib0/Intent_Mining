"""Corpus sampling and train/val/test splitting logic."""

import random
from typing import Any, Dict, List, Tuple


class CorpusSampler:
    """Handles deterministic sampling and splitting for taxonomy learning."""

    @staticmethod
    def sample_and_split(
        records: List[Dict[str, Any]],
        cfg: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Sub-sample dataset and partition into train and validation sets.
        
        Train set is used for Stage 1 summarization and Stage 2 SGD taxonomy updates.
        Validation set is used for Stage 2 validation-based taxonomy selection.
        """
        corpus_cfg = cfg.get("corpus", {})
        seed = corpus_cfg.get("random_seed", 42)
        requested_sample = corpus_cfg.get("sample_size", len(records))
        sample_size = min(requested_sample, len(records))
        val_ratio = corpus_cfg.get("val_ratio", 0.2)

        rng = random.Random(seed)
        sampled = rng.sample(records, sample_size)
        rng.shuffle(sampled)

        val_size = max(1, int(len(sampled) * val_ratio))
        val_set = sampled[:val_size]
        train_set = sampled[val_size:]

        print(f"📊 [CorpusSampler] Sampled {len(sampled)} records (Seed: {seed}) → Train: {len(train_set)}, Val: {len(val_set)}")
        return train_set, val_set


def sample_corpus(records: List[Dict[str, Any]], cfg: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Convenience functional interface."""
    return CorpusSampler.sample_and_split(records, cfg)
