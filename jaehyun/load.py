from PIL import Image

import torch
import torchvision
import torchvision.transforms as transforms


class Load():
    
    def __init__(
            self,
            batch_size: int = 4):

        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize((32, 32)),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.batch_size =  batch_size

    def tensor(
            directory: str):

        trainset = torchvision.datasets.ImageFolder(
                root=directory,
                transform=self.transform)
        trainloader = torch.utils.data.DataLoader(
                trainset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=2)

        testset = torchvision.datasets.ImageFolder(
                root=directory,
                transform=self.transform)
        testloader = torch.utils.data.DataLoader(
                testset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=2)

        return trainset, trainloader, testset, testloader

    def numpy(
            directory: str):

        trainset, _, testset, _ = self.tensor(directory)

        return trainset.data.numpy(), testset.data.numpy()
