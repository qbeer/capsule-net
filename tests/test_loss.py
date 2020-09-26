import unittest
from src.capsule_net.loss import margin_loss
import torch
import numpy as np


class TestLoss(unittest.TestCase):
    def setUp(self):
        self.labels = [[1, 2], [0, 1, 2]]
        self.n_classes = 3

        outputs = np.array([
            [[3.4, 4.3], [1.2, 3.2], [2.1, 2.1]],
            [[3.2, 6.5], [3.3, 1.2], [2.1, 2.5]]], dtype=np.float32)
        outputs = torch.from_numpy(outputs)

        self.outputs = torch.norm(outputs, p=2, dim=-1, keepdim=False)

    def test_loss_with_normed_outputs(self):
        loss = margin_loss(self.outputs, self.labels, n_classes=self.n_classes)
        self.assertIsNotNone(loss)
        self.assertEqual(loss.shape, ())
