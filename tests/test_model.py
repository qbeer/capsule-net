import unittest

import numpy as np
import torch
from src.capsule_net.model import CapsuleNetwork


class CapsuleNetTester(unittest.TestCase):
    def get_random_inputs(self):
        x = np.random.randn(self.batch_size, 3, 28, 28)
        x = torch.from_numpy(x.astype(np.float32)).to(self.device)
        y = np.random.randint(
            low=0, high=self.n_caps[-1], size=(self.batch_size,))
        y = torch.from_numpy(y.astype(np.int32)).to(self.device)

        return x, y

    def setUp(self):
        self.capsule_dims = [8, 16]
        self.n_caps = [1152, 10]
        self.batch_size = 32
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.capsule_net = CapsuleNetwork(
            batch_size=self.batch_size,
            capsule_dims=self.capsule_dims, n_caps=self.n_caps)
        self.capsule_net = self.capsule_net.to(device=self.device)

    def test_ouput_prediction_shape_for_test_input(self):
        inputs = self.get_random_inputs()
        preds, capsules, capsules_normed = self.capsule_net(inputs)
        self.assertEqual(preds.shape, inputs[1].shape)
        self.assertEqual(capsules.shape, (self.batch_size,
                                          self.n_caps[-1],
                                          self.capsule_dims[-1]))
        self.assertEqual(capsules_normed.shape, (self.batch_size,
                                                 self.n_caps[-1]))

    def test_capsule_transformation_matrices(self):
        weight_matrices = self.capsule_net.transformation_weights.detach()\
            .squeeze().cpu().numpy()
        self.assertEqual(weight_matrices.shape, (
            self.n_caps[0], self.n_caps[1], self.capsule_dims[1],
            self.capsule_dims[0]))

    def test_norm(self):
        x, _ = self.get_random_inputs()
        x_normed = self.capsule_net._norm(x, axis=1, keepdims=False)
        self.assertEqual(x_normed.shape, (self.batch_size, 28, 28))
        x_normed = self.capsule_net._norm(x_normed, axis=0, keepdims=True)
        self.assertEqual(x_normed.shape, (1, 28, 28))

    def test_squash(self):
        x, _ = self.get_random_inputs()
        x_squashed = self.capsule_net._squash(x, axis=1)
        self.assertEqual(x_squashed.shape, (self.batch_size, 3, 28, 28))
        norm_x = torch.norm(x, p=2, dim=1, keepdim=True).detach().cpu().numpy()
        self.assertTrue((norm_x >= x_squashed.detach().cpu().numpy()).all())
