from PIL import Image
from torch.utils import data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import random

from torchsummary import summary

class GRU(nn.Module):
    """GRU-based RNN
    Attributes:
        cnn_embed_dim (int, optional): Number of output nodes from CNN.
        num_of_layers (int, optional): Number of hidden RNN layers.
        hidden_size (int, optional): Number of hidden nodes per layer.
        fc_dim (int, optional): Number of nodes in fully connected layer.
        bidirectional (bool, optional): Flag to use bidirectional GRU.
        drop_p (float, optional): Dropout rate.
        num_classes (int, optional): Number of classes.
    """

    def __init__(self, input_size=300, num_of_layers=1, hidden_size=3600, fc_dim=128, bidirectional=False, drop_p=0.5,
                 num_classes=50):
        super(GRU, self).__init__()

        self.input_size = input_size
        self.num_of_layers = num_of_layers  # RNN hidden layers
        self.hidden_size = hidden_size  # RNN hidden nodes
        self.fc_dim = fc_dim
        self.drop_p = drop_p
        self.num_classes = num_classes

        self.GRU = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=num_of_layers,
            bidirectional=bidirectional,
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        if bidirectional:
            self.fc1 = nn.Linear(self.hidden_size * 2, self.fc_dim)
        else:
            self.fc1 = nn.Linear(self.hidden_size, self.fc_dim)

        self.fc2 = nn.Linear(self.fc_dim, self.num_classes)

    def forward(self, input):
        # Shape of input: (batch_size, time_steps, input_size)
        self.GRU.flatten_parameters()
        output, h_n = self.GRU(input)

        # FC layers
        fc_seq = list()
        num_of_time_steps = output.shape[1]
        for t in range(num_of_time_steps): # iterate through time-steps
            output_at_time_step = output[:, t, :] # Shape: [batch_size, input_size]
            x = self.fc1(output_at_time_step)  
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_p, training=self.training)
            x = self.fc2(x) # Shape: [batch_size, num_classes]
            fc_seq.append(x)

        fc_seq = torch.stack(fc_seq, dim=0).transpose_(0, 1)
        return fc_seq # fc_seq.shape = [batch_size, time_steps, num_classes]

if __name__ == "__main__":
    gru = GRU()
    print(summary(gru.cuda(), (5, 300)))