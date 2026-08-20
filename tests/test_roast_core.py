import unittest

import numpy as np

from roast_core import (
    cluster_risk_profiles,
)


class ClusteringTests(unittest.TestCase):
    def test_lower_median_cluster_is_labeled_less(self):
        result = cluster_risk_profiles([
            ("low-a", [0.0, 0.1, 0.0]),
            ("low-b", [0.1, 0.0, 0.1]),
            ("high-a", [10.0, 10.1, 9.9]),
            ("high-b", [9.8, 10.2, 10.0]),
        ])
        self.assertEqual(set(result.less_vulnerable), {"low-a", "low-b"})
        self.assertEqual(set(result.more_vulnerable), {"high-a", "high-b"})
        self.assertEqual(result.distance_matrix.shape, (4, 4))

if __name__ == "__main__":
    unittest.main()
