import torch
import torch.nn as nn
import torch.nn.functional as F


class PrepBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(PrepBlock, self).__init__()
        self.conv_input1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.activation1 = nn.ReLU(inplace=False)
        self.maxpooling = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        residual = self.conv_input1(x)
        residual = self.batch_norm1(residual)
        residual = self.activation1(residual)
        x = self.maxpooling(residual)
        return x


class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out += self.shortcut(residual)
        out = F.relu(out)
        return out


class Layer(nn.Module):
    def __init__(self):
        super(Layer, self).__init__()

        self.layer1 = PrepBlock(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        
        self.layer2 = nn.Sequential(
            BasicBlock(in_channels=64, out_channels=64, stride=1),
            BasicBlock(in_channels=64, out_channels=64, stride=1)
        )
        self.layer3 = nn.Sequential(
            BasicBlock(in_channels=64, out_channels=128, stride=2),
            BasicBlock(in_channels=128, out_channels=128, stride=1)
        )
        self.layer4 = nn.Sequential(
            BasicBlock(in_channels=128, out_channels=256, stride=2),
            BasicBlock(in_channels=256, out_channels=256, stride=1)
        )
        self.layer5 = nn.Sequential(
            BasicBlock(in_channels=256, out_channels=512, stride=2),
            BasicBlock(in_channels=512, out_channels=512, stride=1)
        )

        self.maxpool = nn.AdaptiveAvgPool2d((4,4))  # Adjusted to (4, 4) for spatial dimensions
        self.fc = nn.Linear(512*4*4, 10)  # Adjusted input size for fully connected layer

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

