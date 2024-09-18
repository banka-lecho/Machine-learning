import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batch_norm=False, stride=1):
        super(BasicBlock, self).__init__()
        self.use_batch_norm = use_batch_norm
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        if use_batch_norm:
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            if use_batch_norm:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
                )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out) if self.use_batch_norm else out
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out) if self.use_batch_norm else out

        out += self.shortcut(x)
        out = self.relu(out)

        return out


class ResNet18(nn.Module):
    def __init__(self, block, num_blocks=2, num_classes=1000, use_batch_norm=False):
        super(ResNet18, self).__init__()
        self.use_batch_norm = use_batch_norm
        self.in_channels = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if self.use_batch_norm:
            self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # далее идут 4 слоя с двумя остаточными блоками
        self.layer1 = self.make_layer(block, 64, num_blocks, use_batch_norm, stride=1)
        self.layer2 = self.make_layer(block, 128, num_blocks, use_batch_norm, stride=2)
        self.layer3 = self.make_layer(block, 256, num_blocks, use_batch_norm, stride=2)
        self.layer4 = self.make_layer(block, 512, num_blocks, use_batch_norm, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

        # Инициализация Xavier
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)  # инициализация смещения нулями

    def make_layer(self, block, out_channels, num_blocks, use_batch_norm, stride):
        """Реализуем слоя с n остаточными блоками"""
        layers = []
        # кладем в каждый слой блок
        layers.append(block(self.in_channels, out_channels, use_batch_norm, stride))
        self.in_channels = out_channels
        for i in range(1, num_blocks):
            layers.append(block(out_channels, out_channels, use_batch_norm, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        """Прямой проход по сети"""
        out = self.conv1(x)
        out = self.bn1(out) if self.use_batch_norm else out
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out
