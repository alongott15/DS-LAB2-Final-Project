import torch
from torch import nn

class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        inner_channel = 4 * growth_rate
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, inner_channel, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inner_channel)
        self.conv2 = nn.Conv2d(inner_channel, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        return torch.cat((x, out), 1)

class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downsample = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.downsample(x)

class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=1000):
        super().__init__()
        self.growth_rate = growth_rate
        inner_channels = 2 * growth_rate
        self.conv1 = nn.Conv2d(3, inner_channels, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(inner_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.features = nn.Sequential()

        for i in range(len(nblocks) - 1):
            self.features.add_module(f'desne_block_layer_{i}', self.make_layer(block, inner_channels, nblocks[i]))
            inner_channels += growth_rate * nblocks[i]

            out_channels = int(reduction * inner_channels)
            self.features.add_module(f'transition_layer_{i}', Transition(inner_channels, out_channels))
            inner_channels = out_channels

        self.features.add_module(f'dense_block_{len(nblocks) - 1}', self.make_layer(block, inner_channels, nblocks[len(nblocks) - 1]))
        inner_channels += growth_rate * nblocks[len(nblocks) - 1]
        self.features.add_module('bn', nn.BatchNorm2d(inner_channels))
        self.features.add_module('relu', nn.ReLU(inplace=True))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(inner_channels, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.features(out)
        out = self.avgpool(out)
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        probas = F.softmax(out, dim=1)
        return out, probas

    def make_layer(self, block, in_channels, nblocks):
        dense_block = nn.Sequential()
        for i in range(nblocks):
            dense_block.add_module(f'bottle_neck_layer_{i}', block(in_channels, self.growth_rate))
            in_channels += self.growth_rate
        return dense_block

def densenet121():
    return DenseNet(Bottleneck, [6, 12, 24, 16], growth_rate=32)

def densenet169():
    return DenseNet(Bottleneck, [6, 12, 32, 32], growth_rate=32)

def densenet201():
    return DenseNet(Bottleneck, [6, 12, 48, 32], growth_rate=32)

def densenet161():
    return DenseNet(Bottleneck, [6, 12, 36, 24], growth_rate=48)