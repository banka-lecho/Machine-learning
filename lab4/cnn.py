import torch
import warnings

warnings.filterwarnings("ignore")


class CNN(torch.nn.Module):
    @property
    def device(self):
        for p in self.parameters():
            return p.device

    def __init__(self):
        super(CNN, self).__init__()

        self.layers = torch.nn.Sequential()
        # 1 x 28 x 28 -> 6 x 26 x 26
        self.layers.add_module('conv1', torch.nn.Conv2d(1, 1 * 6, kernel_size=3))
        self.layers.add_module('relu1', torch.nn.ReLU())
        # 6 на 26 на 26 -> 6 на 13 на 13
        # self.layers.add_module('pool1', torch.nn.MaxPool2d(kernel_size=2))

        # 6 на 26 на 26 ->  16 на 24 на 24
        self.layers.add_module('conv2', torch.nn.Conv2d(1 * 6, 1 * 16, kernel_size=3))
        self.layers.add_module('relu2', torch.nn.ReLU())
        # 16 на 24 на 24 -> 16 на 12 на 12
        # self.layers.add_module('pool1', torch.nn.MaxPool2d(kernel_size=2))

        # 16 на 24 на 24 -> -> 16 x 22 x 22
        self.layers.add_module('conv3', torch.nn.Conv2d(16, 16, kernel_size=3))
        self.layers.add_module('relu3', torch.nn.ReLU())
        # 16 на 22 на 22 -> 16 на 11 на 11
        # self.layers.add_module('pool1', torch.nn.MaxPool2d(kernel_size=2))

        # 16 на 22 на 22 -> -> 16 x 20 x 20
        self.layers.add_module('conv4', torch.nn.Conv2d(16, 16, kernel_size=3))
        self.layers.add_module('relu4', torch.nn.ReLU())
        # 16 на 20 на 20 -> 16 на 10 на 10
        self.layers.add_module('pool1', torch.nn.MaxPool2d(kernel_size=2))

        self.layers.add_module('flatten', torch.nn.Flatten())

        # полносвязный слой, который осуществляет операцию умножения входных данных
        # на матрицу весов и добавляет смещение
        self.layers.add_module('linear1', torch.nn.Linear(16 * 10 * 10, 120))
        self.layers.add_module('relu3', torch.nn.ReLU())
        self.layers.add_module('linear2', torch.nn.Linear(120, 84))
        self.layers.add_module('relu4', torch.nn.ReLU())
        self.layers.add_module('linear3', torch.nn.Linear(84, 10))

    def forward(self, input):
        return self.layers(input)
