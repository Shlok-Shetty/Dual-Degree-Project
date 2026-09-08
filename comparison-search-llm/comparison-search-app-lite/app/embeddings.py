"""Loads scenery_embedding.npz, applies the scale fix, exposes helpers."""
from pathlib import Path
import numpy as np

from . import config


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

    def image_path(self, idx: int) -> Path:
        """Resolve to a file on disk. Falls back to local IMAGES_DIR if the
        absolute path stored in the npz doesn't exist on this machine."""
        p = Path(self.paths[idx])
        if p.exists():
            return p
        try:
            rel = Path(*p.parts[p.parts.index("intel_images") + 1:])
            return config.IMAGES_DIR / rel
        except (ValueError, IndexError):
            return p

    def label(self, idx: int) -> str:
        return str(self.labels[idx])

    def indices_by_class(self, cls: str) -> np.ndarray:
        return np.where(self.labels == cls)[0]

    def classes(self) -> list[str]:
        return sorted(set(self.labels.tolist()))
