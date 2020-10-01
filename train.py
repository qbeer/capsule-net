import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision
from src.capsule_net.model import CapsuleNetwork
from src.capsule_net.multimnist_dataset import MultiMnist
from src.capsule_net.loss import margin_loss

BATCH_SIZE = 32
EPOCH = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = MultiMnist(transforms=torchvision.transforms.ToTensor(
), target_transform=torchvision.transforms.ToTensor())

dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                         drop_last=True)

model = CapsuleNetwork(batch_size=BATCH_SIZE)
model = model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

writer = SummaryWriter(log_dir='./runs/')

model.train()
n_iter = 0
for _ in range(EPOCH):
    for samples in dataloader:
        images, labels = samples
        images = images.to(DEVICE)
        labels = labels.to(DEVICE).squeeze()
        labels_pred, capsules, capsules_normed = model((images, labels))
        loss = margin_loss(capsules_normed, labels)
        writer.add_scalar('loss', loss, n_iter)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
