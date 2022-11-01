import sys
sys.path.append("../")

from train import Trainer
from model import VGG16
from load import Load

import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torchvision import transforms

if __name__ == "__main__":

# Use Gpu
    device = torch.device('cpu')
    model = VGG16((3, 224, 224), 20)

    transformer = [
        transforms.Resize(256), 
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]

# Validation loader
    valid_dataloader = Load(
            transformer,
            num_workers=1,
            batch_size=128)
    _, valloader = valid_dataloader("../random_augmented_dataset/valid")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-6)

# Init Trainer
    trainer = Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            )

    trainer.load("./saved_models/test_06.obj")

    trainer.loss_graph()
    trainer.f1_graph()
    trainer.confusion_matrix(valloader)
