"""Loads scenery_embedding.npz, applies the scale fix, exposes helpers.

Images are auto-downloaded from HuggingFace Hub on first run if they aren't
present locally (Nightshade2304/scenery-search-images).
"""
from pathlib import Path
import numpy as np

from . import config


HF_IMAGES_REPO = "Nightshade2304/scenery-search-images"


def _ensure_images_available() -> Path:
    """Make sure images are on disk. If IMAGES_DIR has them, use it.
    Otherwise download the HF dataset once and use its cache path."""
    local_dir = config.IMAGES_DIR
    if local_dir.exists() and any(local_dir.rglob("*.jpg")):
        return local_dir

    # Fall back to HF Hub download (cached under ~/.cache/huggingface/)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise RuntimeError(
            "images not found locally and huggingface_hub is not installed. "
            "Either place images at "
            f"{local_dir} or run: pip install huggingface_hub"
        ) from e

    cached = Path(snapshot_download(
        repo_id=HF_IMAGES_REPO,
        repo_type="dataset",
    ))
    return cached


class Embeddings:
    def __init__(self, npz_path: Path = config.EMBEDDING_NPZ):
        if not npz_path.exists():
            raise FileNotFoundError(
                f"scenery_embedding.npz not found at {npz_path}. "
                f"Expected sibling folder scenery-search/data/ — check config.py."
            )
        data = np.load(npz_path, allow_pickle=True)
        E_work = data["E_work"]
        # scale-calibration fix from session 4 (see session 5 handover §5)
        self.X_scale = float(E_work.std(axis=0).mean())
        self.X = E_work / self.X_scale
        self.paths = np.array([str(p) for p in data["paths"]])
        self.labels = np.array([str(l) for l in data["labels"]])
        self.n, self.d = self.X.shape

        # Resolve where images actually live (local dir or HF cache).
        # Done once at load time so we can rewrite paths cheaply.
        self._images_root = _ensure_images_available()

    def image_path(self, idx: int) -> Path:
        """Resolve to a file on disk. Tries the absolute path from the .npz
        first (works on the machine that generated it), then falls back to
        <images_root>/<class>/<filename>."""
        p = Path(self.paths[idx])
        if p.exists():
            return p
        # HF dataset is organized as <class>/<filename>.jpg — same as our
        # staging layout — so class + filename is enough.
        label = self.labels[idx]
        candidate = self._images_root / str(label) / p.name
        if candidate.exists():
            return candidate
        # Last-resort: legacy intel_images layout
        try:
            rel = Path(*p.parts[p.parts.index("intel_images") + 1:])
            legacy = config.IMAGES_DIR / rel
            if legacy.exists():
                return legacy
        except (ValueError, IndexError):
            pass
        return candidate  # return the expected path even if missing, so
                          # streamlit's warning shows where it was looking

    def label(self, idx: int) -> str:
        return str(self.labels[idx])

    def indices_by_class(self, cls: str) -> np.ndarray:
        return np.where(self.labels == cls)[0]

    def classes(self) -> list[str]:
        return sorted(set(self.labels.tolist()))