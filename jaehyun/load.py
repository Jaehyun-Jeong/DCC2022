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
            self,
            directory: str):

        dataset = torchvision.datasets.ImageFolder(
                root=directory,
                transform=self.transform)
        dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=2)

        return dataset, dataloader

    def numpy(
            self,
            directory: str):

        dataset, dataloader = self.tensor(directory)
        train_dataset_array = next(iter(dataloader))[0].numpy()

        return train_dataset_array
