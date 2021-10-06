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

class ResFeatureExtractor(nn.Module):
    """2D CNN encoder using pretrained ResNet.
    Attributes:
        backbone (string): Name of resnet architecture.
        fc_hidden1 (int, optional): Number of nodes in first hidden layer.
        fc_hidden2 (int, optional): Number of nodes in second hidden layer.
        drop_p (float, optional): Dropout rate.
        cnn_embed_dim (int, optional): Number of output nodes from CNN.
    """

    def __init__(self, backbone='resnet50', fc_hidden1=512, fc_hidden2=512, drop_p=0.3, cnn_embed_dim=300):
        """Load the pretrained ResNet, make Conv5_x layer learnable, and replace last fc layer."""
        super(ResFeatureExtractor, self).__init__()

        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p

        if backbone == 'resnet18':
            resnet = models.resnet18(pretrained=True)
        elif backbone == 'resnet34':
            resnet = models.resnet34(pretrained=True)
        elif backbone == 'resnet50':
            resnet = models.resnet50(pretrained=True)
        elif backbone == 'resnet101':
            resnet = models.resnet101(pretrained=True)

        # for name, child in resnet.named_children():
        #     for name2, params in child.named_parameters():
        #         print(name, name2)

        # Unfreeze layer >=8 (i.e., conv5_x layer)
        # ct = 0
        # for child in resnet.children():
        #     ct += 1
        #     if ct < 8:
        #         for param in child.parameters():
        #             param.requires_grad = False
        # print("Number of layers: {}".format(ct))

        # first_conv_layer = [nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)]
        # first_conv_layer.extend(list(resnet.features))  
        # resnet.features= nn.Sequential(*first_conv_layer)

        for param in resnet.parameters():
            param.requires_grad = True

        modules = list(resnet.children())[:-1]  # delete the last fc layer.

        # self.c1 = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=True)
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(resnet.fc.in_features, fc_hidden1)
        self.bn1 = nn.BatchNorm1d(fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(fc_hidden1, fc_hidden2)
        self.bn2 = nn.BatchNorm1d(fc_hidden2, momentum=0.01)
        self.fc3 = nn.Linear(fc_hidden2, cnn_embed_dim)

    def forward(self, input):
        num_of_strokes = input.shape[-1]
        # print(f'Initial shape of full input: {input.shape}')
        # print(f'Number of strokes: {num_of_strokes}')
        cnn_embed_seq = []
        for t in range(num_of_strokes):
            # ResNet CNN
            #with torch.no_grad():
            x = input[:,:,:,:,t] # Shape: [batch_size, channel (always 3), height, width, time_steps]
            # x = self.c1(x)
            transform = transforms.Compose([
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
            x = transform(x)
            x = self.resnet(x)  # ResNet
            x = x.view(x.size(0), -1)  # flatten output of conv

            # FC layers
            x = self.bn1(self.fc1(x))
            x = F.relu(x)
            x = self.bn2(self.fc2(x))
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_p, training=self.training)
            x = self.fc3(x) # Shape: [batch_size, cnn_embed_dim]

            cnn_embed_seq.append(x)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        # cnn_embed_seq: shape=(batch_size, time_steps, input_size)
        # print(cnn_embed_seq.shape)
        return cnn_embed_seq

if __name__ == "__main__":
    resnet = ResFeatureExtractor()
    print(summary(resnet.cuda(), (3,224,224,4)))
