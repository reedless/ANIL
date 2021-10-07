#!/usr/bin/env python3

"""
"""

import os
import random

import numpy as np
import torch
from torch import nn
import torchvision as tv

import learn2learn as l2l
from learn2learn.data.transforms import FusedNWaysKShots, LoadData, RemapLabels, ConsecutiveLabels

from statistics import mean
from copy import deepcopy
from quickdraw import Quickdraw
from models.cnn import ConvBase
from models.gru import GRU
from models.resnet import ResFeatureExtractor

# 70/15/15 split
SPLITS = {
    'train': [
        'hedgehog',
        'swan',
        'police car',
        'castle',
        'horse',
        'stairs',
        'van',
        'screwdriver',
        'marker',
        'duck',
        'oven',
        'dolphin',
        'stove',
        'ambulance',
        'basket',
        'popsicle',
        'whale',
        'crown',
        'teapot',
        'school bus',
        'potato',
        'eyeglasses',
        'diamond',
        'bowtie',
        'cat',
        'hurricane',
        'square',
        'river',
        'door',
        'triangle',
        'pear',
        'cup',
        'elephant',
        'compass',
        'tractor',
        'ladder',
        'pineapple',
        'bathtub',
        'tiger',
        'drums',
        'cake',
        'ceiling fan',
        'zigzag',
        'light bulb',
        'sheep',
        'flip flops',
        'sailboat',
        'sink',
        'necklace',
        'toothbrush',
        'snorkel',
        'trombone',
        'watermelon',
        'pliers',
        'cruise ship',
        'string bean',
        'raccoon',
        'rainbow',
        'fork',
        'fan',
        'fence',
        'microphone',
        'motorbike',
        'pool',
        'line',
        'bandage',
        'bracelet',
        'syringe',
        'lollipop',
        'grass',
    ],
    'validation': [
        'ant',
        'anvil',
        'backpack',
        'baseball',
        'basketball',
        'binoculars',
        'bread',
        'calendar',
        'camel',
        'candle',
        'carrot',
        'church',
        'clarinet',
        'coffee cup',
        'eye',
    ],
    'test': [        
        'The Eiffel Tower',
        'alarm clock',
        'bulldozer',
        'bus',
        'camera',
        'flower',
        'helicopter',
        'hexagon',
        'laptop',
        'lighthouse',
        'octagon',
        'radio',
        'snowman',
        'tennis racquet',
        'windmill',
    ]
}

class Lambda(nn.Module):

    def __init__(self, fn):
        super(Lambda, self).__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)


def fast_adapt(batch,
               learner,
               features,
               loss,
               adaptation_steps,
               shots,
               ways,
               device=None):

    data, labels = batch
    data, labels = data.to(device), labels.to(device)
    data = features(data)

    # Separate data into adaptation/evaluation sets
    adaptation_indices = np.zeros(data.size(0), dtype=bool)
    adaptation_indices[np.arange(shots*ways) * 2] = True
    evaluation_indices = torch.from_numpy(~adaptation_indices)
    adaptation_indices = torch.from_numpy(adaptation_indices)
    adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
    evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

    for step in range(adaptation_steps):
        train_error = loss(learner(adaptation_data), adaptation_labels)
        learner.adapt(train_error)

    predictions = learner(evaluation_data)
    valid_error = loss(predictions, evaluation_labels)
    valid_accuracy = accuracy(predictions, evaluation_labels)
    return valid_error, valid_accuracy


def main(
        ways=5,
        shots=5,
        meta_lr=0.001,
        fast_lr=0.1,
        adapt_steps=5,
        meta_bsz=32,
        iters=1000,
        cuda=1,
        seed=42,
):

    cuda = bool(cuda)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device('cpu')
    if cuda and torch.cuda.device_count():
        torch.cuda.manual_seed(seed)
        device = torch.device('cuda')

    # Create Datasets (too large to use entire quickdraw)
    train_dataset = Quickdraw(root='~/data',
                                transform=tv.transforms.ToTensor(),
                                mode='custom-train',
                                labels_list=SPLITS['train'],
                                download=True)
    valid_dataset = Quickdraw(root='~/data',
                                transform=tv.transforms.ToTensor(),
                                mode='custom-validation',
                                labels_list=SPLITS['validation'])
    test_dataset = Quickdraw(root='~/data',
                                transform=tv.transforms.ToTensor(),
                                mode='custom-test',
                                labels_list=SPLITS['test'])

    print(len(train_dataset)) # 2.5k * 70 = 175000
    print(len(valid_dataset)) # 2.5k * 15 = 37500
    print(len(test_dataset))  # 2.5k * 15 = 37500

    print(len(train_dataset.data[0])) # a class w 2.5k samples

    # get max number of strokes, need to pad all stroke_seqs to be the same
    max_strokes = 0
    all_classes = np.concatenate((train_dataset.data, valid_dataset.data, test_dataset.data))
    for class_set in all_classes:
        for stroke_seq in class_set:
            ml = stroke_seq.shape[0]
            if ml > max_strokes:
                max_strokes = ml

    print(max_strokes) # 227

    print(train_dataset[0][0])
    
    curr = train_dataset[0][0].size()[1]
    diff = max_strokes - curr
    print(diff)
    padding = torch.zeros([1, diff, 3], dtype=torch.int16)
    temp = torch.cat((train_dataset[0][0], padding), 1)

    print(temp.shape)

    # for stroke_seq, label in test_dataset:
    #     print(stroke_seq.shape)

    raise ValueError()

    train_dataset = l2l.data.MetaDataset(train_dataset)
    valid_dataset = l2l.data.MetaDataset(valid_dataset)
    test_dataset = l2l.data.MetaDataset(test_dataset)

    train_transforms = [
        FusedNWaysKShots(train_dataset, n=ways, k=2*shots),
        LoadData(train_dataset),
        RemapLabels(train_dataset),
        ConsecutiveLabels(train_dataset),
    ]
    train_tasks = l2l.data.TaskDataset(train_dataset,
                                       task_transforms=train_transforms,
                                       num_tasks=2000) # or add replacement=True

    valid_transforms = [
        FusedNWaysKShots(valid_dataset, n=ways, k=2*shots),
        LoadData(valid_dataset),
        ConsecutiveLabels(valid_dataset),
        RemapLabels(valid_dataset),
    ]
    valid_tasks = l2l.data.TaskDataset(valid_dataset,
                                       task_transforms=valid_transforms,
                                       num_tasks=60) # or add replacement=True

    test_transforms = [
        FusedNWaysKShots(test_dataset, n=ways, k=2*shots),
        LoadData(test_dataset),
        RemapLabels(test_dataset),
        ConsecutiveLabels(test_dataset),
    ]
    test_tasks = l2l.data.TaskDataset(test_dataset,
                                      task_transforms=test_transforms,
                                      num_tasks=60) # or add replacement=True

    # Create model 
    # body
    # features = ConvBase(hidden=50, channels=1, max_pool=True)
    resnet = ResFeatureExtractor(backbone='resnet18', fc_hidden1=512, fc_hidden2=512, drop_p=0.3, cnn_embed_dim=512).to(device)
    gru = GRU(input_size=512).to(device)
    features = torch.nn.Sequential(resnet, gru, Lambda(lambda x: x.view(-1, 256)))
    features.to(device)

    # head
    head = torch.nn.Linear(256, ways)
    head = l2l.algorithms.MAML(head, lr=fast_lr)
    head.to(device)

    # Setup optimization
    all_parameters = list(features.parameters()) + list(head.parameters())
    optimizer = torch.optim.Adam(all_parameters, lr=meta_lr)
    loss = nn.CrossEntropyLoss(reduction='mean')

    # Training loop
    print('Training')
    for iteration in range(iters):
        optimizer.zero_grad()
        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        meta_test_error = 0.0
        meta_test_accuracy = 0.0
        for task in range(meta_bsz):
            # Compute meta-training loss
            learner = head.clone()
            batch = train_tasks.sample()

            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               features,
                                                               loss,
                                                               adapt_steps,
                                                               shots,
                                                               ways,
                                                               device)
            evaluation_error.backward()
            meta_train_error += evaluation_error.item()
            meta_train_accuracy += evaluation_accuracy.item()

            # Compute meta-validation loss
            learner = head.clone()
            batch = valid_tasks.sample()
            # TODO: process images into strokes
            
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               features,
                                                               loss,
                                                               adapt_steps,
                                                               shots,
                                                               ways,
                                                               device)
            meta_valid_error += evaluation_error.item()
            meta_valid_accuracy += evaluation_accuracy.item()

            # Compute meta-testing loss
            learner = head.clone()
            batch = test_tasks.sample()
            # TODO: process images into strokes
            
            evaluation_error, evaluation_accuracy = fast_adapt(batch,
                                                               learner,
                                                               features,
                                                               loss,
                                                               adapt_steps,
                                                               shots,
                                                               ways,
                                                               device)
            meta_test_error += evaluation_error.item()
            meta_test_accuracy += evaluation_accuracy.item()

        # Print some metrics
        print('\n')
        print('Iteration', iteration)
        print('Meta Train Error', meta_train_error / meta_bsz)
        print('Meta Train Accuracy', meta_train_accuracy / meta_bsz)
        print('Meta Valid Error', meta_valid_error / meta_bsz)
        print('Meta Valid Accuracy', meta_valid_accuracy / meta_bsz)
        print('Meta Test Error', meta_test_error / meta_bsz)
        print('Meta Test Accuracy', meta_test_accuracy / meta_bsz)

        # Average the accumulated gradients and optimize
        for p in all_parameters:
            p.grad.data.mul_(1.0 / meta_bsz)
        optimizer.step()


if __name__ == '__main__':
    # sanity check
    for label in SPLITS['train']:
        if label in SPLITS['test']:
            print('test: ' + label)
    for label in SPLITS['validation']:
        if label in SPLITS['test']:
            print('test: ' + label)
    main()
