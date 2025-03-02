from typing import List
from torch.nn import Module, Linear, ReLU, Dropout
from torch.nn import ModuleList


class MLP(Module):

    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, dropout=0.0):
        super(MLP, self).__init__()
        self.layers = ModuleList()
        # Input layer
        self.layers.append(Linear(input_dim, hidden_dims[0]))
        self.layers.append(ReLU())
        self.layers.append(Dropout(dropout))
        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            self.layers.append(Linear(hidden_dims[i], hidden_dims[i + 1]))
            self.layers.append(ReLU())
            self.layers.append(Dropout(dropout))
        # Output layer
        self.layers.append(Linear(hidden_dims[-1], output_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x