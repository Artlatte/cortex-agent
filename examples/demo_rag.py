"""Run the offline RAG demo: python examples/demo_rag.py"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cortex.demo import run_rag_demo  # noqa: E402

if __name__ == "__main__":
    asyncio.run(run_rag_demo())
