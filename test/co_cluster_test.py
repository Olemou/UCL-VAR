import torch
import unittest
from spatialcl.uncertainty import co_cluster_uncertainty

class TestCoCluster(unittest.TestCase):
    def test_basic(self):
        z = torch.randn(4, 8)

        img_id = torch.tensor([0, 1, 2, 3])
        label = torch.tensor([0, 1, 0, 1])

        prior_weight = 2

        output = co_cluster_uncertainty(z, label, img_id, prior_weight)

        self.assertIsNotNone(output)
        self.assertIsInstance(output, torch.Tensor)
        self.assertTrue(output.dim() == 2)
        self.assertFalse(torch.isnan(output).any(), "Output contains NaN values")
        self.assertFalse(torch.isinf(output).any(), "Output contains Inf values")


if __name__ == "__main__":
    unittest.main()
