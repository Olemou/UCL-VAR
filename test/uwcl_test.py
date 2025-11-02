import torch
import unittest
from spatialcl.uwcl import build_uwcl

class TestUwcl(unittest.TestCase):
    def test_basic(self):
        img_id=torch.tensor([0, 1, 2, 3])
        label=torch.tensor([0, 1, 0, 1])
        z = torch.randn(4, 8)
        output = build_uwcl(
            z=z,
            img_ids= img_id,
            labels= label,
            epoch = 0
        )
       
        self.assertIsNotNone(output)
        self.assertIsInstance(output, torch.Tensor)
        
if __name__ == "__main__":
    unittest.main()
