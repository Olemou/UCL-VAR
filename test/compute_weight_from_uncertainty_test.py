import torch
import unittest
from spatialcl.clearn import compute_weights_from_uncertainty

class TestComputeWeight(unittest.TestCase):
    def test_basic(self):
       uncertainty_matrix=torch.randn(8, 64) 
       output = compute_weights_from_uncertainty(
           uncertainty=uncertainty_matrix,
           epoch= 0
        )
       self.assertIsNotNone(output)
       self.assertIsInstance(output, torch.Tensor)
       self.assertTrue(output.max() <= torch.exp(torch.tensor(1)))

if __name__ == "__main__":
    unittest.main()
