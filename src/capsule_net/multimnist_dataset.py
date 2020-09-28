import torch
import torchvision
import random
import numpy as np
from PIL import Image


class MultiMnist(torch.utils.data.Dataset):
    def __init__(self, transforms=None, transfrom_target=None,
                 image_transforms=torchvision.transforms.Compose(transforms=[
                     torchvision.transforms.RandomAffine(degrees=(-5, 5),
                                                         translate=(.2, .2),
                                                         scale=(.5, .7)),
                     torchvision.transforms.RandomCrop(size=(26, 26)),
                     torchvision.transforms.Resize(size=(28, 28))
                 ]), train=True):
        self.transforms = transforms
        self.transform_target = transfrom_target
        self.image_transforms = image_transforms
        self.train = train
        self.mnist = torchvision.datasets.MNIST(
            root='/tmp/', download=True, train=self.train)

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, index):
        index1 = index
        index2 = random.randint(0, len(self))

        image1, label1 = self.mnist[index1]
        image2, label2 = self.mnist[index2]

        while label1 == label2:
            index2 = random.randint(0, len(self))
            image2, label2 = self.mnist[index2]

        if self.image_transforms:
            image1 = self.image_transforms(image1)
            image2 = self.image_transforms(image2)

        x = np.array(image1)
        y = np.array(image2)
        blend = np.where(x > y, x, y)
        image = Image.fromarray(blend)

        if self.transforms:
            image = self.transforms(image)

        label = np.zeros(shape=(10, ))
        label[label1] = 1
        label[label2] = 1

        if self.transform_target:
            label = self.transform_target(label)

        return image, label
