import numpy as np


class GammaCKLEngine:
    """γ-CKLSearch engine (Chumbalov et al. 2024, Algorithm 3).

    Public interface matches SearchEngine so streamlit_app.py stays unchanged:
        step (int)          — number of queries answered
        done (bool)         — search complete
        stop_reason (str)   — "in_query", "max_queries", or a manual reason
        propose_query()     — returns (i, j)
        oracle_answer(i, j) — returns 0 (picked i) or 1 (picked j)
        apply_answer(y, status="clean") — advances state, returns history record
        stop_manual(reason) — for user_found_target / user_quit
    """

    def __init__(self, X, gamma, target_idx=None, seed=0, max_queries=50,
                 r=2.0, score_mode="paper_textual"):
        self.X = X
        self.n, self.d = X.shape
        self.gamma = float(gamma)
        self.r = float(r)
        self.score_mode = score_mode
        self.target_idx = target_idx
        self.max_queries = max_queries

        self.rng = np.random.default_rng(seed)
        self.P = np.ones(self.n) / self.n
        self.used = set()
        self.step = 0
        self.done = False
        self.stop_reason = None
        self._pending_query = None

    def propose_query(self):
        if self._pending_query is not None:
            return self._pending_query
        eps = 1e-12
        mu = (self.P[:, None] * self.X).sum(axis=0)
        diff = self.X - mu
        Sigma = (self.P[:, None, None] * diff[:, :, None] * diff[:, None, :]).sum(axis=0)
        eigvals, eigvecs = np.linalg.eigh(Sigma)
        lam_max = max(eigvals[-1], eps)
        v_max = eigvecs[:, -1]
        z1 = mu + self.r * np.sqrt(lam_max) * v_max
        z2 = mu - self.r * np.sqrt(lam_max) * v_max

        def score(z, excl):
            d2 = np.sum((self.X - z) ** 2, axis=1)
            if self.score_mode == "paper_literal":
                s = self.P * d2
            else:
                s = d2 / (self.P + eps)
            for u in excl:
                s[u] = np.inf
            return int(np.argmin(s))

        i = score(z1, self.used)
        j = score(z2, self.used | {i})
        self._pending_query = (i, j)
        return i, j

    def oracle_answer(self, i, j):
        """Simulate a γ-CKL oracle answering the query. Returns 0 (picked i) or 1 (picked j).
        Only used in auto-run; for human search the UI supplies y directly."""
        if self.target_idx is None:
            raise RuntimeError("oracle_answer called without a target_idx")
        x_t = self.X[self.target_idx]
        d_i = np.linalg.norm(self.X[i] - x_t)
        d_j = np.linalg.norm(self.X[j] - x_t)
        if d_i == 0:
            return 0
        if d_j == 0:
            return 1
        p_i = (d_j ** self.gamma) / (d_i ** self.gamma + d_j ** self.gamma)
        return 0 if self.rng.random() < p_i else 1

    def apply_answer(self, y, status="clean"):
        """y is 0 (picked i) or 1 (picked j), or None for skip (treated as no-op)."""
        i, j = self.propose_query()
        record = {
            "step": self.step, "i": int(i), "j": int(j),
            "y": None if y is None else int(y), "status": status,
        }
        self.used |= {i, j}
        self._pending_query = None

        if self.target_idx is not None and self.target_idx in (i, j) and y is not None:
            self.step += 1
            self.done = True
            self.stop_reason = "in_query"
            return record

        if y is not None:
            picked_idx = i if y == 0 else j
            other_idx = j if y == 0 else i
            eps = 1e-12
            d_pick = np.linalg.norm(self.X - self.X[picked_idx], axis=1)
            d_other = np.linalg.norm(self.X - self.X[other_idx], axis=1)
            d_pick_g = d_pick ** self.gamma
            d_other_g = d_other ** self.gamma
            denom = d_pick_g + d_other_g + eps
            likelihood = d_other_g / denom
            P_new = self.P * likelihood
            Z = P_new.sum()
            if Z > eps:
                self.P = P_new / Z

        self.step += 1
        if self.step >= self.max_queries:
            self.done = True
            self.stop_reason = "max_queries"
        return record

    def stop_manual(self, reason: str):
        self.done = True
        self.stop_reason = reason
