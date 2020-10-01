import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision
from src.capsule_net.model import CapsuleNetwork
from src.capsule_net.multimnist_dataset import MultiMnist
from src.capsule_net.loss import margin_loss
from pathlib import Path
import yaml

config = yaml.safe_load(open('./config.yml'))

BATCH_SIZE = config['common']['batch_size']
DEVICE = torch.device(config['train']['device'])
CHKPT_PATH = config['common']['chkpt_path']
EPOCHS = config['train']['epochs']
LR = config['train']['lr']

dataset = MultiMnist(transforms=torchvision.transforms.ToTensor(),
                     target_transform=torchvision.transforms.ToTensor())

dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=BATCH_SIZE,
                                         drop_last=True)

model = CapsuleNetwork(batch_size=BATCH_SIZE)
model = model.to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

writer = SummaryWriter(log_dir='./runs/')

if Path(CHKPT_PATH).exists():
    print('Found checkpoint, loading it...')
    model.load_state_dict(torch.load(CHKPT_PATH,
                                     map_location=DEVICE))

model.train()
n_iter = 0
for _ in range(EPOCHS):
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
        n_iter += 1
    torch.save(model.state_dict(), CHKPT_PATH)
