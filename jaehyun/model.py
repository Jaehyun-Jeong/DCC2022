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


class ResNet18(nn.Module):

    def __init__(
            self,
            inputs: tuple = (3, 224, 224),
            outputs: int = 20):

        super().__init__()

        self.flatten = nn.Flatten(1, 3)

    def input_block(
            self,
            x: torch.Tensor):

        block = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
                    nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                    ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
                )

        return block(x)

    def basic_block(
            self,
            x: torch.Tensor,
            input_size: int,
            output_size: int):

        if input_size != output_size:
            down_conv1 = Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
            down_bn1 = BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        block = nn.Sequential(
            Conv2d(size, output_size, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            atchNorm2d(output_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            ReLU(inplace=True)
            Conv2d(output_size, output_size, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            BatchNorm2d(output_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))

        x = F.relu(x)

        if input_size == output_size:
            out = block(x) + x
        else:
            downsampled_x = down_bn1(down_conv1(x))
            out = block(x) + downsampled_x

        out = F.relu(out)

        return out

    def output_block(
            self,
            x: torch.Tensor,
            input_feature_size: int,
            output_feature_size: int):

            avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7)),
            linear = nn.Linear(in_features=input_feature_size, out_features=output_feature_size, bias=True)

            x = avgpool(x)
            x = self.flatten(x)
            x = linear(x)

            return x

    def forward(
            self,
            x: torch.Tensor):

        x = self.input_block(x)
        x = self.basic_block(x, 64, 64)
        x = self.basic_block(x, 64, 64)
        x = self.basic_block(x, 64, 128)
        x = self.basic_block(x, 128, 128)
        x = self.basic_block(x, 128, 256)
        x = self.basic_block(x, 256, 256)
        x = self.basic_block(x, 256, 512)
        x = self.basic_block(x, 512, 512)
        x = self.output_block(512, outputs)

        return x


if __name__ == "__main__":
    model = VGG16()
    print(model)
