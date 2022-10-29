import torch
import torch.nn as nn
import torch.nn.functional as F


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


class VGG16(nn.Module):

    def __init__(
            self,
            inputs: tuple = (3, 224, 224),
            outputs: int = 20):

        super().__init__()

        self.feature = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        )

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))

        self.flatten = nn.Flatten(1, 3)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=outputs, bias=True),
        )

    # Input as 224x224x3 image
    def forward(self, x):
        x = self.feature(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)

        return x


class ResidualBlock(nn.Module):

    def __init__(
            self,
            input_size: int,
            output_size: int):

        super().__init__()

        if input_size == output_size:
            self.identity_block1 = nn.Sequential()
        else:
            self.identity_block1 = self.identity_block(input_size, output_size)

        self.identity_block2 = nn.Sequential()

        self.relu = nn.ReLU(inplace=True)
        self.basic_block1 = self.basic_block(input_size, output_size)
        self.basic_block2 = self.basic_block(output_size, output_size)

    def identity_block(
            self,
            input_size: int,
            output_size: int):

        return nn.Sequential(
            nn.Conv2d(input_size, output_size, kernel_size=(1, 1), stride=(2, 2), bias=False),
            nn.BatchNorm2d(output_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))

    def basic_block(
            self,
            input_size: int,
            output_size: int):

        stride_size = int(output_size / input_size)

        block = nn.Sequential(
            nn.Conv2d(input_size, output_size, kernel_size=(3, 3), stride=(stride_size, stride_size), padding=(1, 1), bias=False),
            nn.BatchNorm2d(output_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_size, output_size, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(output_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),)

        return block

    def forward(
            self,
            x: torch.Tensor):

        identity1 = self.identity_block1(x)
        x = self.basic_block1(x)
        x += identity1
        x = self.relu(x)

        identity2 = self.identity_block2(x)
        x = self.basic_block2(x)
        x += identity2
        x = self.relu(x)

        return x


class ResNet18(nn.Module):

    def __init__(
            self,
            inputs: tuple = (3, 224, 224),
            outputs: int = 20):

        super().__init__()

        self._first_block = self.input_block()
        self._residual_block1 = ResidualBlock(64, 64)
        self._residual_block2 = ResidualBlock(64, 128)
        self._residual_block3 = ResidualBlock(128, 256)
        self._residual_block4 = ResidualBlock(256, 512)
        self._last_block = self.output_block(512, outputs)

    def input_block(self):

        block = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
                    nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
                )

        return block


    def output_block(
            self,
            input_feature_size: int,
            output_feature_size: int):

        block = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(1, 3),
            nn.Linear(in_features=input_feature_size, out_features=output_feature_size, bias=True),)

        return block

    def forward(
            self,
            x: torch.Tensor):

        x = self._first_block(x)
        x = self._residual_block1(x)
        x = self._residual_block2(x)
        x = self._residual_block3(x)
        x = self._residual_block4(x)
        x = self._last_block(x)

        return x


if __name__ == "__main__":
    model = ResNet18((3, 224, 224), 20)
    print(model)
