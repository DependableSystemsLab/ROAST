"""DTW complete-linkage clustering with deterministic vulnerability labels."""

from dataclasses import dataclass

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


def _dtw_distance(left, right):
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    previous = np.full(len(right) + 1, np.inf)
    previous[0] = 0.0
    for value in left:
        current = np.full(len(right) + 1, np.inf)
        for index, other in enumerate(right, start=1):
            cost = (value - other) ** 2
            current[index] = cost + min(current[index - 1], previous[index], previous[index - 1])
        previous = current
    return float(np.sqrt(previous[-1]))


def _distance_matrix(profiles):
    try:
        from dtaidistance import dtw

        matrix = np.asarray(dtw.distance_matrix_fast(profiles, compact=False), dtype=np.float64)
    except ImportError:
        count = len(profiles)
        matrix = np.zeros((count, count), dtype=np.float64)
        for row in range(count):
            for column in range(row + 1, count):
                matrix[row, column] = matrix[column, row] = _dtw_distance(
                    profiles[row], profiles[column]
                )
    np.fill_diagonal(matrix, 0.0)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("DTW produced non-finite distances")
    return matrix


@dataclass(frozen=True)
class ClusterResult:
    labels: tuple
    less_vulnerable: tuple
    more_vulnerable: tuple
    assignments: np.ndarray
    distance_matrix: np.ndarray
    linkage_matrix: np.ndarray
    cluster_medians: dict
    subject_impacts: dict


def cluster_risk_profiles(records):
    """Cluster profiles into two groups and label the lower-impact group as less vulnerable."""
    labels = tuple(str(record[0]) for record in records)
    profiles = [np.asarray(record[1], dtype=np.float64).reshape(-1) for record in records]
    if len(profiles) < 2 or any(len(profile) == 0 for profile in profiles):
        raise ValueError("at least two non-empty risk profiles are required")

    distance_matrix = _distance_matrix(profiles)
    condensed = squareform(distance_matrix, checks=True)
    linkage_matrix = linkage(condensed, method="complete")
    assignments = fcluster(linkage_matrix, t=2, criterion="maxclust")
    cluster_ids = sorted(np.unique(assignments))
    if len(cluster_ids) != 2:
        raise ValueError("risk profiles did not form two non-empty clusters")

    impacts = {label: float(np.mean(profile)) for label, profile in zip(labels, profiles)}
    medians = {
        cluster_id: float(np.median([impacts[label] for label, value in zip(labels, assignments) if value == cluster_id]))
        for cluster_id in cluster_ids
    }
    less_cluster = min(cluster_ids, key=lambda value: (medians[value], value))
    less = tuple(label for label, value in zip(labels, assignments) if value == less_cluster)
    more = tuple(label for label, value in zip(labels, assignments) if value != less_cluster)
    return ClusterResult(
        labels=labels,
        less_vulnerable=less,
        more_vulnerable=more,
        assignments=assignments,
        distance_matrix=distance_matrix,
        linkage_matrix=linkage_matrix,
        cluster_medians=medians,
        subject_impacts=impacts,
    )
