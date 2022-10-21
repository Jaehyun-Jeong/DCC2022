<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
import numpy as np

=======
<<<<<<< HEAD
>>>>>>> 81582bb (made file and methods')
=======
<<<<<<< HEAD
>>>>>>> 04afce7 (test)
import torch
import torchvision
import torchvision.transforms as transforms
=======
<<<<<<< HEAD
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
=======
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
>>>>>>> e2f22c7 (fixed merge conflicts)
>>>>>>> 9f37040 (fixed merge conflicts)


class Load():

    def __init__(
            self,
            batch_size: int = 4,
            flatten: bool = False):

        transformer = [
             transforms.ToTensor(),
             transforms.Resize((128, 128)),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        if flatten:
            transformer.append(transforms.Lambda(torch.flatten))

        self.transform = transforms.Compose(transformer)
        self.batch_size = batch_size

    def __call__(
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

    def tensor_label(
            self,
            directory: str,
            label_name: str,
            ) -> torch.Tensor:

        dataset, dataloader = self(directory)

        label_idx = dataset.class_to_idx[label_name]
        label_idx_list = [
                idx for idx, target_idx in enumerate(dataset.targets)
                if target_idx == label_idx]

        label_dataset = dataset[label_idx_list[0]][0].unsqueeze(0)
        for idx in label_idx_list[1:]:
            label_dataset = torch.cat(
                    (label_dataset, dataset[idx][0].unsqueeze(0)),
                    dim=0)

        return label_dataset

    def tensor(
            self,
            directory: str):

        dataset, dataloader = self(directory)

        total_targets = dataset.targets

        total_dataset = dataset[0][0].unsqueeze(0)
        for data_idx in range(1, len(dataset)):
            total_dataset = torch.cat(
                    (total_dataset, dataset[data_idx][0].unsqueeze(0)),
                    dim=0)

        return total_dataset, total_targets

    def numpy_label(
            self,
            directory: str,
            label_name: str,
            ) -> np.ndarray:

        label_dataset = self.tensor_label(
                directory=directory,
                label_name=label_name)

        return label_dataset.numpy()


if __name__ == "__main__":

    Loader = Load()
    dataset, targetset = Loader.tensor('./dataset')

    print(dataset.shape)
    print(len(targetset))

    pass
<<<<<<< HEAD
=======
=======
import torch
import torchvision
import torchvision.transforms as transforms
>>>>>>> d3d50fb (test)


def Load(
        directory: str):

<<<<<<< HEAD
    def __init__(self):
        pass
>>>>>>> 36ae001 (made file and methods')
=======
    transform = transforms.Compose(
        [transforms.ToTensor(),
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
            download=True,
            transform=transform)
    testloader = torch.utils.data.DataLoader(
            testset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2)

    return trainloader, testloader
>>>>>>> d3d50fb (test)
=======
>>>>>>> e2f22c7 (fixed merge conflicts)
