"""γ-CKLSearch engine (Chumbalov et al. 2024, Algorithm 3).

Public interface matches SearchEngine so streamlit_app.py can swap between
them: propose_query, oracle_answer, apply_answer, stop_manual, step, done,
stop_reason, belief_stats.
"""
import numpy as np

from . import config


class GammaCKLEngine:
    algo_name = "gamma-CKL"

    def __init__(self, X, gamma: float = config.GAMMA,
                 target_idx: int | None = None, seed: int = config.DEFAULT_SEED,
                 max_queries: int = config.MAX_QUERIES,
                 r: float = 2.0, score_mode: str = "paper_textual"):
        self.X = X
        self.n, self.d = X.shape
        self.gamma = float(gamma)
        self.r = float(r)
        self.score_mode = score_mode
        self.target_idx = target_idx
        self.max_queries = max_queries

        self.rng = np.random.default_rng(seed)
        self.P = np.ones(self.n) / self.n
        self.used: set[int] = set()
        self.step = 0
        self.step_records: list[dict] = []
        self.done = False
        self.stop_reason: str | None = None
        self._pending_query: tuple[int, int] | None = None

    def propose_query(self) -> tuple[int, int]:
        if self.done:
            raise RuntimeError("search already finished")
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

        def pick(z, excl):
            d2 = np.sum((self.X - z) ** 2, axis=1)
            if self.score_mode == "paper_literal":
                s = self.P * d2
            else:
                s = d2 / (self.P + eps)
            for u in excl:
                s[u] = np.inf
            return int(np.argmin(s))

        i = pick(z1, self.used)
        j = pick(z2, self.used | {i})
        self._pending_query = (i, j)
        return i, j

    def oracle_answer(self, i: int, j: int) -> int:
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

    def apply_answer(self, y: int | None, *,
                     utterance: str | None = None,
                     style: str | None = None,
                     parsed=None,
                     status: str = "clean") -> dict:
        if self.done:
            raise RuntimeError("search already finished")
        if self._pending_query is None:
            raise RuntimeError("call propose_query() before apply_answer()")

        i, j = self._pending_query
        self.used |= {i, j}
        target_in_query = (self.target_idx is not None) and (self.target_idx in (i, j))

        record = {
            "step": self.step,
            "i": int(i), "j": int(j),
            "y": None if y is None else int(y),
            "utterance": utterance,
            "style": style,
            "parsed": parsed,
            "status": status,
            "target_in_query": target_in_query,
            "applied_update": False,
        }

        if target_in_query:
            self.done = True
            self.stop_reason = "in_query"
            self.step_records.append(record)
            self._pending_query = None
            self.step += 1
            return record

        if y is not None:
            eps = 1e-12
            picked_idx = i if y == 0 else j
            other_idx = j if y == 0 else i
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
            "algo": self.algo_name,
            "step": self.step,
            "argmax_idx": int(np.argmax(self.P)),
            "p_target": (float(self.P[self.target_idx])
                          if self.target_idx is not None else None),
            "max_p": float(self.P.max()),
        }
