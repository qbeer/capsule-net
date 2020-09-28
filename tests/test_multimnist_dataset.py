import torchvision
import matplotlib.pyplot as plt
import numpy as np
from src.capsule_net.multimnist_dataset import MultiMnist
import unittest
import torch


class MultiMnistDatasetTests(unittest.TestCase):
    def setUp(self):
        self.dataset = MultiMnist(deterministic=True)

    def test_output_shape_after_overlay(self):
        image, label = self.dataset[0]
        self.assertTrue(hasattr(image, "im"))
        self.assertEqual(image.size, (28, 28))
        self.assertEqual(label.shape, (1, 10))

    def test_transforms(self):
        transforms = torchvision.transforms.RandomResizedCrop(size=(28, 28))
        dataset = MultiMnist(transforms=transforms, deterministic=True)
        image_tr, label_tr = dataset[0]
        image, label = self.dataset[0]
        self.assertListEqual(label_tr.tolist(), label.tolist())
        self.assertListEqual(np.array(image_tr).tolist(),
                             np.array(transforms(image)).tolist())

    def test_to_tensor_image_and_target_transforms(self):
        transform = torchvision.transforms.ToTensor()
        target_transform = torchvision.transforms.ToTensor()
        dataset = MultiMnist(transforms=transform,
                             target_transform=target_transform)
        image, label = dataset[0]
        self.assertTrue(isinstance(image, torch.Tensor))
        self.assertTrue(isinstance(label, torch.Tensor))

    def test_image_visualization(self):
        fig, axes = plt.subplots(
            3, 3, sharex=True, sharey=True, figsize=(10, 10))
        for data, ax in zip(self.dataset, axes.flatten()):
            image, label = data
            ax.imshow(image)
            ax.set_title(np.where(label)[0].tolist())
            ax.set_xticks([])
            ax.set_yticks([])
        fig.tight_layout()
        plt.savefig('multimnist_example.png')
