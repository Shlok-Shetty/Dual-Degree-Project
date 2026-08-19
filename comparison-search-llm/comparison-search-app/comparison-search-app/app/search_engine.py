"""GAUSSSEARCH primitives + stateful SearchEngine.

Math is copy-pasted verbatim from gamma-ckl/notebooks/01_gauss_search.ipynb and
scenery-search/notebooks/03_gauss_search_3d_viz.ipynb so behavior matches your
existing experiments.
"""
import numpy as np
from scipy.stats import norm

from . import config


def symmetrize(A):
    return 0.5 * (A + A.T)


def bisecting_hyperplane(x_i, x_j):
    diff = x_i - x_j
    nrm = np.linalg.norm(diff)
    if nrm < 1e-12:
        w = np.zeros_like(diff); w[0] = 1.0
        return w, 0.0
    w = diff / nrm
    b = (np.dot(x_j, x_j) - np.dot(x_i, x_i)) / (2 * nrm)
    return w, b


def probit_prob(x_i, x_j, x_t, sigma_eps):
    w, b = bisecting_hyperplane(x_i, x_j)
    return norm.cdf((np.dot(x_t, w) + b) / sigma_eps)


def query_oracle(x_i, x_j, x_t, sigma_eps, rng):
    p = probit_prob(x_i, x_j, x_t, sigma_eps)
    return 0 if rng.random() < p else 1


def sample_mirror(mu, Sigma, X, used, rng):
    Sigma = symmetrize(Sigma)
    _, eigvecs = np.linalg.eigh(Sigma)
    w_star = eigvecs[:, -1]
    b_star = -np.dot(w_star, mu)
    z1 = rng.multivariate_normal(mu, Sigma)
    z2 = z1 - 2 * (np.dot(w_star, z1) + b_star) * w_star

    def nearest(z, exclude):
        diffs = X - z
        d2 = np.einsum("nd,nd->n", diffs, diffs)
        for k in exclude:
            d2[k] = np.inf
        return int(np.argmin(d2))

    i = nearest(z1, used)
    j = nearest(z2, used | {i})
    return i, j


def adf_update(mu, Sigma, x_i, x_j, y, sigma_eps):
    w, b = bisecting_hyperplane(x_i, x_j)
    if y == 1:
        w = -w; b = -b
    Sw = Sigma @ w
    wSw = float(w @ Sw)
    denom = np.sqrt(wSw + sigma_eps ** 2)
    mu_w = float(mu @ w)
    g = (mu_w + b) / denom
    phi_g = norm.pdf(g)
    Phi_g = max(norm.cdf(g), 1e-12)
    alpha = phi_g / (Phi_g * denom)
    beta = -alpha * (g / denom + alpha)
    tau = -beta / (1 + beta * wSw)
    nu = (alpha - beta * (mu_w + b)) / (1 + beta * wSw)
    Sigma_new = Sigma - (tau / (1 + tau * wSw)) * np.outer(Sw, Sw)
    Sigma_new = symmetrize(Sigma_new)
    mu_new = mu + (nu - b * tau - tau * mu_w) * (Sigma_new @ w)
    return mu_new, Sigma_new


class SearchEngine:
    """Stateful GAUSSSEARCH — one step at a time so the UI can drive it.

    Usage:
        engine = SearchEngine(X, sigma_eps=0.05, target_idx=42, seed=1)
        while not engine.done:
            i, j = engine.propose_query()
            # ... show images, obtain answer y ∈ {0, 1} (or None to skip) ...
            engine.apply_answer(y)

    Swap this out for γ-CKL later by mirroring the same surface:
    propose_query, apply_answer, done, step, step_records, belief_stats.
    """

    def __init__(self, X: np.ndarray, sigma_eps: float = config.SIGMA_EPS,
                 target_idx: int | None = None, seed: int = config.DEFAULT_SEED,
                 max_queries: int = config.MAX_QUERIES):
        self.X = X
        self.n, self.d = X.shape
        self.sigma_eps = sigma_eps
        self.target_idx = target_idx
        self.max_queries = max_queries
        self.rng = np.random.default_rng(seed)

        self.mu = np.zeros(self.d)
        self.Sigma = np.eye(self.d)
        self.used: set[int] = set()
        self.step = 0
        self.step_records: list[dict] = []

        self.done = False
        self.stop_reason: str | None = None  # in_query | max_queries | user_found_target
        self._pending_query: tuple[int, int] | None = None

    def propose_query(self) -> tuple[int, int]:
        """Return (i, j) for the next query. Idempotent within a step."""
        if self.done:
            raise RuntimeError("search already finished")
        if self._pending_query is None:
            i, j = sample_mirror(self.mu, self.Sigma, self.X, self.used, self.rng)
            self._pending_query = (i, j)
        return self._pending_query

    def oracle_answer(self, i: int, j: int) -> int:
        """Ground-truth Probit oracle answer. Requires known target (LLM mode)."""
        if self.target_idx is None:
            raise RuntimeError("oracle_answer requires target_idx")
        return query_oracle(self.X[i], self.X[j], self.X[self.target_idx],
                             self.sigma_eps, self.rng)

    def apply_answer(self, y: int | None, *,
                     utterance: str | None = None,
                     style: str | None = None,
                     parsed=None,
                     status: str = "clean") -> dict:
        """Advance by one step. y=None skips the belief update but still counts
        the step and marks (i, j) as used."""
        if self.done:
            raise RuntimeError("search already finished")
        if self._pending_query is None:
            raise RuntimeError("call propose_query() before apply_answer()")

        i, j = self._pending_query
        self.used |= {i, j}
        target_in_query = (self.target_idx is not None) and (self.target_idx in (i, j))

        record = {
            "step": self.step,
            "i": i, "j": j,
            "y": y,
            "utterance": utterance,
            "style": style,
            "parsed": parsed,
            "status": status,
            "target_in_query": target_in_query,
            "applied_update": False,
        }

        # in-query stop happens BEFORE update (paper §6.3, matches user-study UX)
        if target_in_query:
            self.done = True
            self.stop_reason = "in_query"
            self.step_records.append(record)
            self._pending_query = None
            self.step += 1
            return record

        if y is not None:
            self.mu, self.Sigma = adf_update(
                self.mu, self.Sigma, self.X[i], self.X[j], y, self.sigma_eps,
            )
            record["applied_update"] = True

        self.step_records.append(record)
        self._pending_query = None
        self.step += 1

        if self.step >= self.max_queries:
            self.done = True
            self.stop_reason = "max_queries"

        return record

    def stop_manual(self, reason: str = "user_found_target"):
        self.done = True
        self.stop_reason = reason

    def belief_stats(self) -> dict:
        return {
            "step": self.step,
            "trace_sigma": float(np.trace(self.Sigma)),
            "dist_to_target": (float(np.linalg.norm(self.mu - self.X[self.target_idx]))
                                if self.target_idx is not None else None),
        }
