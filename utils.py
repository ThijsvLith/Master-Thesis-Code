"""Utility helpers for repository-wide paths."""
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

PROCESSED_DATA_DIR = PROJECT_DIR / "processed_data"
RESULTS_DIR = PROJECT_DIR / "results"


def ensure_directory(path: Path) -> None:
    """Create directory if it does not exist already."""
    path.mkdir(parents=True, exist_ok=True)
