import torch
import unittest
from spatialcl.uncertainty import co_cluster_uncertainty
from spatialcl._dto.image_label_dto import ImageLabelDTO

class TestCoCluster(unittest.TestCase):
    def test_basic(self):
        z = torch.randn(4, 8)
        image_label_dto = ImageLabelDTO(
            img_id=torch.tensor([0, 1, 2, 3]),
            label=torch.tensor([0, 1, 0, 1]),

        )
        prior_weight = 2

        output = co_cluster_uncertainty(z, image_label_dto,prior_weight)
       
        self.assertIsNotNone(output)
        self.assertIsInstance(output, torch.Tensor)
        self.assertTrue(output.dim() == 2)

if __name__ == "__main__":
    unittest.main()
