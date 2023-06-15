from dataclasses import dataclass
from typing import Optional


LIST_OF_ENVIRONMENTS = [
    "cartpole",
    "inverted_pendulum",
    "mountain_car",
    "lunar_lander",
]


@dataclass(kw_only=True)
class HOOTRunConfigs:
    algorithm: str = "hoot"
    environment: str
    n_actions: int
    search_depth: int
    algorithm_iter: int

    gamma: float = 0.99
    v1: Optional[float] = None
    ce: float = 1.

    seed: Optional[int] = None

    def __post_init__(self):
        if self.environment not in LIST_OF_ENVIRONMENTS:
            raise ValueError(
                f"Environment should be in {LIST_OF_ENVIRONMENTS}"
            )

    def to_dict(self):
        return {
            "algorithm": self.algorithm,
            "environment": self.environment,
            "search_depth": self.search_depth,
            "n_actions": self.n_actions,
            "algorithm_iter": self.algorithm_iter,
            "gamma": self.gamma,
            "v1": self.v1,
            "ce": self.ce,
            "seed": self.seed,
        }


@dataclass(kw_only=True)
class tHOOTRunConfigs(HOOTRunConfigs):
    algorithm: str = "tuncated_hoot"


@dataclass(kw_only=True)
class LDHOOTRunConfigs(HOOTRunConfigs):
    algorithm: str = "ld_hoot"
    hoo_max_depth: int

    def to_dict(self):
        return {
                **super().to_dict(),
                "hoo_max_depth": self.hoo_max_depth,
            }


@dataclass(kw_only=True)
class PolyHOOTRunConfigs(LDHOOTRunConfigs):
    algorithm: str = "poly_hoot"

    alpha: float = 5.
    eta: float = 20.
    xi: float = 0.5

    def to_dict(self):
        return {
                **super().to_dict(),
                "alpha": self.alpha,
                "eta": self.eta,
                "xi": self.xi,
            }
