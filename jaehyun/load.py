from PIL import Image

import torch
import torchvision
import torchvision.transforms as transforms


def Load(
        directory: str):

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize((32, 32)),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 4

    trainset = torchvision.datasets.ImageFolder(
            root=directory,
            transform=transform)
    trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2)

    testset = torchvision.datasets.ImageFolder(
            root=directory,
            transform=transform)
    testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2)

    return trainset, trainloader, testset, testloader
