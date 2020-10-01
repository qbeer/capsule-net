import unittest

import numpy as np
import torch
from src.capsule_net.loss import margin_loss


class TestLoss(unittest.TestCase):
    def setUp(self):
        self.labels = torch.tensor([[0, 1, 1], [1, 1, 1]], dtype=torch.float32)

        outputs = np.array([[[3.4, 4.3], [1.2, 3.2], [2.1, 2.1]],
                            [[3.2, 6.5], [3.3, 1.2], [2.1, 2.5]]],
                           dtype=np.float32)
        outputs = torch.from_numpy(outputs)

        self.outputs = torch.norm(outputs, p=2, dim=-1, keepdim=False)

    def test_loss_with_normed_outputs(self):
        loss = margin_loss(
            outputs=self.outputs,
            one_hot_labels=self.labels,
        )
        self.assertIsNotNone(loss)
        self.assertEqual(loss.shape, ())
