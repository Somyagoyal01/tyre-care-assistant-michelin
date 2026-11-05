"""Root conftest so plain `pytest` from a fresh clone can import the `assistant` package."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
