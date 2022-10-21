<<<<<<< HEAD
=======
# PyTorch
>>>>>>> 36ae001 (made file and methods')
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        super().__init__()

        c, h, w = input_dim

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        self.conv1 = nn.Conv2d(
                in_channels=c,
                out_channels=32,
                kernel_size=8,
                stride=4)
        self.conv2 = nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2)
        self.conv3 = nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1)
        self.linear1 = nn.Linear(3136, 512)
        self.linear2 = nn.Linear(512, output_dim)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        img = x

        img = F.relu(self.conv1(img))
        img = F.relu(self.conv2(img))
        img = F.relu(self.conv3(img))
        img = torch.flatten(img, 1)
        img = F.relu(self.linear1(img))
        img = self.linear2(img)

        return img
>>>>>>> 36ae001 (made file and methods')
