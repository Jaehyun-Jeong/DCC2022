<<<<<<< HEAD
<<<<<<< HEAD
=======
# PyTorch
>>>>>>> 36ae001 (made file and methods')
=======
>>>>>>> d3d50fb (test)
import torch
import torch.nn as nn
import torch.nn.functional as F


<<<<<<< HEAD
<<<<<<< HEAD
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 20)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
=======
class CNN_V2(nn.Module):
    def __init__(self, input_dim, output_dim):
=======
class Model(nn.Module):
    def __init__(self):
>>>>>>> d3d50fb (test)
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
<<<<<<< HEAD
        img = x

        img = F.relu(self.conv1(img))
        img = F.relu(self.conv2(img))
        img = F.relu(self.conv3(img))
        img = torch.flatten(img, 1)
        img = F.relu(self.linear1(img))
        img = self.linear2(img)

        return img
>>>>>>> 36ae001 (made file and methods')
=======
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
>>>>>>> d3d50fb (test)
