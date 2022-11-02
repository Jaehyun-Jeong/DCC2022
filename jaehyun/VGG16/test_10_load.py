import sys
sys.path.append("../")

from train import Trainer
from model import VGG16
from load import Load
from utils import FocalLoss

import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torchvision import transforms

# Use Gpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = VGG16((3, 224, 224), 20)

criterion = FocalLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-6)

# Init Trainer
trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        )

# Train

trainer.load("./saved_models/test_10.obj")

trainer.loss_graph()
trainer.f1_graph()
