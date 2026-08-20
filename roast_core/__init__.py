"""Shared, paper-faithful primitives used by ROAST dataset pipelines."""

from .clustering import ClusterResult, cluster_risk_profiles

__all__ = [
    "ClusterResult",
    "cluster_risk_profiles",
]
