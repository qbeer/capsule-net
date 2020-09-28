from PIL import Image
import numpy as np
import torch
import torchvision
import random


class MultiMnist(torch.utils.data.Dataset):
    def __init__(self, transforms=None, target_transform=None,
                 image_transforms=None, train=True, deterministic=False):
        self.deterministic = deterministic
        if self.deterministic:
            self.seed = 42
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            np.random.seed(self.seed)
            random.seed(self.seed)

        self.transforms = transforms
        self.target_transform = target_transform
        self.image_transforms = torchvision.transforms.Compose(transforms=[
            torchvision.transforms.RandomAffine(degrees=(-5, 5),
                                                translate=(.2, .2),
                                                scale=(.5, .7)),
            torchvision.transforms.RandomCrop(size=(26, 26)),
            torchvision.transforms.Resize(size=(28, 28))
        ])
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

        if self.deterministic:
            random.seed(self.seed)
        while label1 == label2:
            index2 = random.randint(0, len(self))
            image2, label2 = self.mnist[index2]

        if self.deterministic:
            random.seed(self.seed)
        if self.image_transforms:
            image1 = self.image_transforms(image1)
            image2 = self.image_transforms(image2)

        x = np.array(image1)
        y = np.array(image2)
        blend = np.where(x > y, x, y)
        image = Image.fromarray(blend)

        if self.deterministic:
            random.seed(self.seed)
        if self.transforms:
            image = self.transforms(image)

        label = np.zeros(shape=(1, 10), dtype=np.uint8)
        label[:, label1] = 1
        label[:, label2] = 1

        if self.deterministic:
            random.seed(self.seed)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label
