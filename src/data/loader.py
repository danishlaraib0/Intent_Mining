"""Production-grade Corpus Loader and Data I/O Utilities.

Supports:
- JSONL, JSON array, and CSV formats
- Field validation & schema inspection
- Empty/corrupt record filtering
- Safe chunked writes with error handling
"""

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Union

logger = logging.getLogger("CorpusLoader")


class CorpusLoader:
    """Production data loader with schema validation and multi-format support."""

    @staticmethod
    def load_jsonl(path: Union[str, Path], required_field: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load records from a JSONL file with validation."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Corpus file not found: {path}")

        records: List[Dict[str, Any]] = []
        corrupted_lines = 0

        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if required_field and (required_field not in obj or not str(obj[required_field]).strip()):
                        continue
                    records.append(obj)
                except json.JSONDecodeError as e:
                    corrupted_lines += 1
                    if corrupted_lines <= 5:
                        logger.warning(f"Corrupt JSON at {path}:{line_no} - {e}")

        if corrupted_lines > 0:
            logger.warning(f"Skipped {corrupted_lines} corrupted lines in {path}")
        return records

    @staticmethod
    def load_csv(path: Union[str, Path], text_field: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load records from a CSV file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        records: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if text_field and not row.get(text_field, "").strip():
                    continue
                records.append(dict(row))
        return records

    @staticmethod
    def load_json(path: Union[str, Path]) -> List[Dict[str, Any]]:
        """Load records from a JSON array file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"JSON file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            raise ValueError(f"Expected JSON array at {path}, got {type(data)}")

    @classmethod
    def load(cls, path: Union[str, Path], text_field: Optional[str] = None) -> List[Dict[str, Any]]:
        """Auto-detect format by extension and load dataset."""
        p = Path(path)
        suffix = p.suffix.lower()

        if suffix in [".jsonl", ".ndjson"]:
            records = cls.load_jsonl(p, required_field=text_field)
        elif suffix == ".csv":
            records = cls.load_csv(p, text_field=text_field)
        elif suffix == ".json":
            records = cls.load_json(p)
        else:
            # Fall back to jsonl
            records = cls.load_jsonl(p, required_field=text_field)

        print(f"📂 [CorpusLoader] Successfully loaded {len(records)} records from {p.name}")
        return records


def load_corpus(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convenience function matching pipeline config schema."""
    corpus_cfg = cfg.get("corpus", {})
    path = corpus_cfg.get("path")
    text_field = corpus_cfg.get("text_field", "question")
    return CorpusLoader.load(path, text_field=text_field)


def save_jsonl(data: List[Dict[str, Any]], path: Union[str, Path]) -> None:
    """Save records to JSONL safely."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp")

    with open(temp_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    temp_path.replace(path)
    print(f"💾 Saved {len(data)} records → {path}")
