"""
Screening diagnostics — per-node and aggregate statistics collected during
tree fitting.

These are the runtime observability layer required by D0 in CLAUDE.md:
early detection of over-rejection (all splits killed) or under-rejection
(screening is a no-op, behaving identically to standard GBDT).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NodeDiagnostics:
    """Per-node screening statistics collected during a single split evaluation."""

    node_id: int
    depth: int
    n_samples: int
    n_candidates: int       # total (feat × bin) pairs evaluated
    n_accepted: int         # pairs where rho > 0
    accept_rate: float      # n_accepted / n_candidates
    rho_max: float
    rho_mean: float         # mean over accepted candidates only (nan if none)
    rho_median: float       # median over accepted candidates only (nan if none)
    split_found: bool       # True if a split was actually emitted


@dataclass
class ScreeningDiagnostics:
    """Aggregate diagnostics over all nodes evaluated during a tree fit."""

    nodes: list[NodeDiagnostics] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    # Aggregate properties                                                 #
    # ------------------------------------------------------------------ #

    @property
    def n_total_candidates(self) -> int:
        return sum(n.n_candidates for n in self.nodes)

    @property
    def n_total_accepted(self) -> int:
        return sum(n.n_accepted for n in self.nodes)

    @property
    def overall_accept_rate(self) -> float:
        total = self.n_total_candidates
        return self.n_total_accepted / total if total > 0 else float("nan")

    @property
    def root_accept_rate(self) -> Optional[float]:
        """accept_rate of the root node (depth 0), or None if not yet recorded."""
        for n in self.nodes:
            if n.depth == 0:
                return n.accept_rate
        return None

    @property
    def rejected_at_root(self) -> bool:
        """True if the root node found no valid split (Inv-2 / D0 critical alert)."""
        for n in self.nodes:
            if n.depth == 0:
                return not n.split_found
        return False

    @property
    def n_nodes_evaluated(self) -> int:
        return len(self.nodes)

    @property
    def n_splits_found(self) -> int:
        return sum(1 for n in self.nodes if n.split_found)

    @property
    def n_leaves_by_screening(self) -> int:
        """Nodes that became leaves specifically because screening rejected all splits."""
        return sum(1 for n in self.nodes if not n.split_found and n.n_candidates > 0)

    def summary(self) -> str:
        lines = [
            f"Nodes evaluated : {self.n_nodes_evaluated}",
            f"Splits found    : {self.n_splits_found}",
            f"Leaves (screening rejected): {self.n_leaves_by_screening}",
            f"Root accept_rate: {self.root_accept_rate:.1%}" if self.root_accept_rate is not None else "Root accept_rate: n/a",
            f"Overall accept_rate: {self.overall_accept_rate:.1%}",
        ]
        if self.rejected_at_root:
            lines.append("WARNING: root node was rejected — tree has no splits. "
                         "Try reducing s_r or increasing s_w.")
        return "\n".join(lines)
