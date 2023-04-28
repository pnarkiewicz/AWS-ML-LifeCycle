import torch
from torch import nn
import torchtext


class LSTM(nn.Module):
    """Class implementing and inheriting from torch.nn.Module
    Simple LSTM followed by 2 linear layers (1st one is activated with relu)

    :param batch_size: size of batch used during training or default inference (use forward_ for inference used out of training and different batch_size)
    :param num_classes: number of distinct classes
    :param kwargs: dictionary defining hyperparameters for torch.nn.LSTM (batch_first must be True always)
    """

    def __init__(self, batch_size, num_classes, kwargs=None):
        super().__init__()

        if kwargs is not None:
            self.kwargs = kwargs
        else:
            self.kwargs = {
                "input_size": 300,
                "hidden_size": 100,
                "num_layers": 4,
                "batch_first": True,
                "dropout": 0.05,
            }

        assert self.kwargs["batch_first"], "batch_first must be true!"

        self.lstm = nn.LSTM(**self.kwargs)
        self.linear1 = nn.Linear(
            self.kwargs["hidden_size"], self.kwargs["hidden_size"] // 2
        )
        self.linear2 = nn.Linear(self.kwargs["hidden_size"] // 2, num_classes)
        self.relu = nn.ReLU()

        self.batch_size = batch_size
        self.h0 = torch.zeros(
            size=(self.kwargs["num_layers"], batch_size, self.kwargs["hidden_size"])
        )
        self.c0 = torch.zeros(
            size=(self.kwargs["num_layers"], batch_size, self.kwargs["hidden_size"])
        )

        self.hc = (self.h0, self.c0)

    def forward(self, x):
        x, _ = self.lstm(x, self.hc)
        x = self.relu(self.linear1(x[:, -1, :]))
        x = self.linear2(x)

        return x

    def forward_(self, x):
        """Same function as forward but x's batch_size can be different"""

        h0 = torch.zeros(
            size=(self.kwargs["num_layers"], x.size()[0], self.kwargs["hidden_size"])
        )
        c0 = torch.zeros(
            size=(self.kwargs["num_layers"], x.size()[0], self.kwargs["hidden_size"])
        )
        x, _ = self.lstm(x, (h0, c0))
        x = self.relu(self.linear1(x[:, -1, :]))
        x = self.linear2(x)

        return x
