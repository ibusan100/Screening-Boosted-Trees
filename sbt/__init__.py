"""
Screening Boosted Trees (sbt) — M4 release.

A GBDT framework where each candidate split is scored by an *absolute*
relevance value derived from the screening transform in "Screening Is Enough"
(Nakanishi 2026, arXiv:2604.01178). Splits below a learnable threshold are
rejected exactly, enabling the tree to represent "no relevant split" without
an external min_gain_to_split heuristic.

NumPy reference (always available):

    from sbt import ScreeningParams, ScreeningBooster, ScreeningTree
    from sbt import screening_split_numpy, build_histogram_numpy

Triton-accelerated path (requires [triton] extra):

    from sbt import screening_split_triton, build_histogram_triton
"""

from .binning import Binner
from .booster import ScreeningBooster
from .diagnostics import NodeDiagnostics, ScreeningDiagnostics
from .oblivious_tree import ObliviousTree
from .param_search import ScreeningParamSearch
from .screening_split import (
    ScreeningParams,
    build_histogram_numpy,
    screening_split_numpy,
)
from .tree import ScreeningTree

__all__ = [
    "Binner",
    "NodeDiagnostics",
    "ScreeningDiagnostics",
    "ScreeningParams",
    "ScreeningBooster",
    "ObliviousTree",
    "ScreeningParamSearch",
    "ScreeningTree",
    "build_histogram_numpy",
    "screening_split_numpy",
    "screening_split_triton",
    "build_histogram_triton",
]

__version__ = "0.0.1"


def __getattr__(name: str):
    if name in {"screening_split_triton", "build_histogram_triton"}:
        try:
            from .kernels import screening_split_triton as _mod
        except ImportError as e:
            raise ImportError(
                f"sbt.{name} requires Triton. Install with: "
                "pip install screening-boosted-trees[triton]"
            ) from e
        return getattr(_mod, name)
    raise AttributeError(f"module 'sbt' has no attribute {name!r}")
