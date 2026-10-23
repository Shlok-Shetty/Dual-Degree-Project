"""Paths and hyperparams for the lite app. No model configs — no LLMs here."""
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = APP_ROOT.parent  # ddp-llm/ or comparison-search-llm/

SCENERY_ROOT = PROJECT_ROOT / "scenery-search"
SCENERY_DATA = SCENERY_ROOT / "data"
EMBEDDING_NPZ = SCENERY_DATA / "scenery_embedding.npz"
IMAGES_DIR = SCENERY_DATA / "intel_images"

SIGMA_EPS = 0.05
MAX_QUERIES = 50
DEFAULT_SEED = 42

DRIVE_GALLERY_URL = "https://drive.google.com/drive/folders/PLACEHOLDER"

CLASS_COLORS = {
    "buildings": "#e74c3c",
    "forest":    "#27ae60",
    "glacier":   "#3498db",
    "mountain":  "#8b4513",
    "sea":       "#1abc9c",
    "street":    "#f39c12",
}