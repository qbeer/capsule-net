from src.capsule_net.multimnist_dataset import MultiMnist
from src.capsule_net.model import CapsuleNetwork
import torchvision
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml

config = yaml.safe_load(open('./config.yml'))

BATCH_SIZE = config['common']['batch_size']
DEVICE = torch.device(config['predict']['device'])
CHKPT_PATH = config['common']['chkpt_path']
GRID_SIZE = config['predict']['grid_size']
MARGIN_POS = config['predict']['margin_pos']


dataset = MultiMnist(transforms=torchvision.transforms.ToTensor(),
                     target_transform=torchvision.transforms.ToTensor())

dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=BATCH_SIZE)

model = CapsuleNetwork(batch_size=BATCH_SIZE)
model.load_state_dict(torch.load(CHKPT_PATH, map_location=DEVICE))
model = model.to(DEVICE)

model.eval()

for sample in dataloader:
    images, labels = sample
    images = images.to(DEVICE)
    labels = labels.to(DEVICE).squeeze()

    single_digit_preds, capsules, capsules_normed = model((images, labels))
    break

capsules_normed = capsules_normed.detach().cpu().numpy()
images = images.detach().cpu().numpy().reshape(-1, 28, 28)
labels = labels.detach().cpu().numpy()


fig, axes = plt.subplots(GRID_SIZE, GRID_SIZE, sharex=True, sharey=True,
                         figsize=(16, 16))

for image, label, capsule, ax in zip(images, labels, capsules_normed,
                                     axes.flatten()):
    ax.imshow(image)
    labels_pred = np.where(capsule > MARGIN_POS)[0]
    labels_true = np.where(label > MARGIN_POS)[0]
    ax.set_title('T : %s | P : %s' %
                 (labels_true.tolist(),
                  labels_pred.tolist()))
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
plt.savefig('multimnist_predictions.png', dpi=50)
