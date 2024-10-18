import torch


class Perceptron(torch.nn.Module):
    @property
    def device(self):
        for p in self.parameters():
            return p.device

    def __init__(self, input_dim=784, num_layers=0,
                 hidden_dim=64, output_dim=10, p=0.0, init_weights=False, batch_norm=False):
        super(Perceptron, self).__init__()

        self.layers = torch.nn.Sequential()
        prev_size = input_dim
        for i in range(num_layers):
            linear_layer = torch.nn.Linear(prev_size, hidden_dim)
            if init_weights:
                torch.nn.init.xavier_uniform_(linear_layer.weight)

            self.layers.add_module('layer{}'.format(i), linear_layer)
            if batch_norm:
                self.layers.add_module('batch_norm{}'.format(i),
                                       torch.nn.BatchNorm1d(hidden_dim))

            self.layers.add_module('relu{}'.format(i), torch.nn.ReLU())
            self.layers.add_module('dropout{}'.format(i), torch.nn.Dropout(p=p))
            prev_size = hidden_dim

        self.layers.add_module('classifier',
                               torch.nn.Linear(prev_size, output_dim))
        if init_weights:
            torch.nn.init.xavier_uniform_(self.layers[-1].weight)

    def forward(self, input):
        return self.layers(input)
