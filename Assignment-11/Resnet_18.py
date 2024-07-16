import torch
import torch.nn as nn
import torch.optim as optim


class PrepBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(PrepBlock, self).__init__()
        self.conv_input1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.activation1 = nn.ReLU()
        self.maxpooling = nn.MaxPool2d(3, stride=2, padding=0)

    def forward(self, x):
        residual = self.conv_input1(x)  # Convolve input directly for downsampling
        residual = self.batch_norm1(residual)
        residual = self.activation1(residual)
        x = self.maxpooling(residual)  # Ensure to use residual for maxpooling
        return x

class BasicCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(BasicCNNBlock, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_input1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.activation1 = nn.ReLU(inplace=False)
        self.conv_input2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.activation2 = nn.ReLU(inplace=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x

        x = self.conv_input1(x)
        x = self.batch_norm1(x)
        x = self.activation1(x)

        x = self.conv_input2(x)
        x = self.batch_norm2(x)

        if self.stride != 1 or self.in_channels != self.out_channels:
            residual = self.shortcut(residual)

        x = x+residual
        x = self.activation2(x)

        return x

class Layer(nn.Module):
    def __init__(self):
        super(Layer, self).__init__()
        self.layer1 = PrepBlock(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.layer2 = BasicCNNBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.layer3 = BasicCNNBlock(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.layer4 = BasicCNNBlock(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.layer5 = BasicCNNBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)

        self.maxpool_layer = nn.MaxPool2d(kernel_size=7)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = self.maxpool_layer(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

