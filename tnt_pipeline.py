#!/usr/bin/env python3
"""
TnT-LLM Pipeline: Modular Interface.
Delegates to the production architecture in src/ and run_pipeline.py.
Maintained for backward compatibility.
"""

import sys
from pathlib import Path

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from run_pipeline import main

if __name__ == "__main__":
    main()
