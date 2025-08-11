"""Output parsing utilities for taxonomy and classification JSON outputs."""

import json
import re
from typing import Any, Dict, List, Optional


class OutputParser:
    """Robust parser for LLM JSON generation."""

    @staticmethod
    def parse_json(raw_text: str) -> Any:
        """Strip markdown fences, find JSON block, and parse."""
        cleaned = raw_text.strip()

        # Remove ```json ... ``` or ``` ... ```
        fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
        if fence_match:
            cleaned = fence_match.group(1).strip()

        # Try direct parse
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Try finding the outermost [ ... ] or { ... }
        array_match = re.search(r"\[\s*\{[\s\S]*\}\s*\]", cleaned)
        if array_match:
            try:
                return json.loads(array_match.group(0))
            except json.JSONDecodeError:
                pass

        obj_match = re.search(r"\{[\s\S]*\}", cleaned)
        if obj_match:
            try:
                return json.loads(obj_match.group(0))
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not parse valid JSON from LLM response:\n{raw_text[:300]}...")


def parse_json_response(raw: str) -> Any:
    """Functional wrapper for parse_json."""
    return OutputParser.parse_json(raw)
