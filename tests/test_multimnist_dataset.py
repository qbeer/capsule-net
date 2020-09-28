import unittest
from src.capsule_net.multimnist_dataset import MultiMnist
import numpy as np
import matplotlib.pyplot as plt


class MultiMnistDatasetTests(unittest.TestCase):
    def setUp(self):
        self.dataset = MultiMnist()

    def test_output_shape_after_overlay(self):
        image, label = self.dataset[0]
        self.assertTrue(hasattr(image, "im"))
        self.assertEqual(image.size, (28, 28))
        self.assertEqual(label.shape, (10,))

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
