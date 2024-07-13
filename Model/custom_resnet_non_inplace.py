import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary

class PrepBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(PrepBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        return x

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)

        residual = self.shortcut(residual)
        x += residual
        x = self.relu2(x)

        return x

class Layer(nn.Module):
    def __init__(self):
        super(Layer, self).__init__()
        self.preplayer = PrepBlock(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.conv1_layer1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn1_layer1 = nn.BatchNorm2d(128)
        self.relu1_layer1 = nn.ReLU()
        self.maxpool_layer1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.resblock_layer1 = ResBlock(128, 128)

        self.conv1_layer2 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn1_layer2 = nn.BatchNorm2d(256)
        self.relu1_layer2 = nn.ReLU()
        self.maxpool_layer2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1_layer3 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn1_layer3 = nn.BatchNorm2d(512)
        self.relu1_layer3 = nn.ReLU()
        self.maxpool_layer3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.resblock_layer3 = ResBlock(512, 512)

        self.maxpool_layer4 = nn.MaxPool2d(kernel_size=4)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 10)  # 10 classes for CIFAR10

    def forward(self, x):
        x = self.preplayer(x)

        x = self.conv1_layer1(x)
        x = self.bn1_layer1(x)
        x = self.relu1_layer1(x)
        x = self.maxpool_layer1(x)
        
        residual = x
        x = self.resblock_layer1(x)
        x += residual

        x = self.conv1_layer2(x)
        x = self.bn1_layer2(x)
        x = self.relu1_layer2(x)
        x = self.maxpool_layer2(x)

        x = self.conv1_layer3(x)
        x = self.bn1_layer3(x)
        x = self.relu1_layer3(x)
        x = self.maxpool_layer3(x)
        
        residual = x
        x = self.resblock_layer3(x)
        x += residual

        x = self.maxpool_layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return F.log_softmax(x, dim=1)

# Instantiate the network
# model = Layer()
# print(model)
