#!/usr/bin/env python3

import os
import tarfile
import subprocess
import numpy as np
import pickle

from PIL import Image
from torch.utils.data import Dataset

from learn2learn.data.utils import download_file
from torchvision.datasets.folder import default_loader

DATA_DIR = 'quickdraw'
GCLOUD_BUCKET = 'gs://quickdraw_dataset/sketchrnn/'

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
        'alarm clock',
        'crown',
        'teapot',
        'octagon',
        'school bus',
        'potato',
        'eyeglasses',
        'diamond',
        'bowtie',
        'cat',
        'The Eiffel Tower',
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
        'camera',
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
        'windmill',
        'lollipop',
        'grass',
        'airplane',
        'sword',
        'boomerang',
        'telephone',
        'guitar',
        'bed',
        'paint can',
        'sandwich',
        'sock',
        'tent',
        'stop sign',
        'scorpion',
        'toothpaste',
        'squiggle',
        'The Great Wall of China',
        'hot tub',
        'bottlecap',
        'mug',
        'baseball bat',
        'belt',
        'sun',
        'rake',
        'pillow',
        'parachute',
        'foot',
        'pencil',
        'traffic light',
        'underwear',
        'calculator',
        'blackberry',
        'shark',
        'nail',
        'ear',
        'cloud',
        'lighthouse',
        'lightning',
        'rain',
        'The Mona Lisa',
        'apple',
        'shorts',
        'star',
        'clock',
        'sea turtle',
        'bicycle',
        'fireplace',
        'lighter',
        'squirrel',
        'chandelier',
        'cannon',
        'paintbrush',
        'tree',
        'jail',
        'pants',
        'envelope',
        'onion',
        'pizza',
        'sleeping bag',
        'lipstick',
        'dragon',
        't-shirt',
        'snowflake',
        'hot air balloon',
        'cooler',
        'peas',
        'skull',
        'dresser',
        'harp',
        'garden',
        'leaf',
        'camouflage',
        'house plant',
        'see saw',
        'megaphone',
        'map',
        'penguin',
        'dog',
        'peanut',
        'keyboard',
        'strawberry',
        'truck',
        'car',
        'pig',
        'crayon',
        'headphones',
        'floor lamp',
        'hamburger',
        'wine glass',
        'beach',
        'ocean',
        'circle',
        'asparagus',
        'remote control',
        'moon',
        'rifle',
        'shovel',
        'hospital',
        'barn',
        'picture frame',
        'scissors',
        'crab',
        'moustache',
        'brain',
        'lion',
        'banana',
        'chair',
        'skateboard',
        'book',
        'mushroom',
        'shoe',
        'key',
        'passport',
        'broccoli',
        'elbow',
        'leg',
        'dumbbell',
        'bird',
        'cello',
        'hockey puck',
        'submarine',
        'canoe',
        'rhinoceros',
        'bush',
        'flying saucer',
        'arm',
        'frog',
        'train',
        'dishwasher',
        'washing machine',
        'swing set',
        'aircraft carrier',
        'vase',
        'crocodile',
        'monkey',
        'blueberry',
        'cell phone',
        'toe',
        'garden hose',
        'zebra',
        'hexagon',
        'owl',
        'postcard',
        'speedboat',
        'mosquito',
        'birthday cake',
        'pickup truck',
        'hand',
        'computer',
        'piano',
        'fish',
        'soccer ball',
        'bear',
        'mouth',
        'face',
        'violin',
        'bench',
        'stereo',
        'jacket',
        'spreadsheet',
        'power outlet',
        'knee',
        'mountain',
        'octopus',
        'laptop',
        'snail',
        'flamingo',
        'spoon',
    ],
    'test': [
        'angel',
        'animal migration',
        'axe',
        'bat',
        'beard',
        'bee',
        'bridge',
        'broom',
        'bucket',
        'butterfly',
        'cactus',
        'campfire',
        'cookie',
        'couch',
        'cow',
        'diving board',
        'donut',
        'drill',
        'eraser',
        'feather',
        'finger',
        'fire hydrant',
        'flashlight',
        'frying pan',
        'giraffe',
        'goatee',
        'hammer',
        'helicopter',
        'helmet',
        'hockey stick',
        'hourglass',
        'house',
        'kangaroo',
        'lantern',
        'mailbox',
        'mouse',
        'palm tree',
        'parrot',
        'rabbit',
        'rollerskates',
        'saw',
        'saxophone',
        'snowman',
        'spider',
        'stitches',
        'streetlight',
        'sweater',
        'table',
        'umbrella',
        'wheel',
        'wine bottle',
        'yoga',
    ],
    'validation': [
        'ant',
        'anvil',
        'backpack',
        'baseball',
        'basketball',
        'binoculars',
        'bread',
        'bulldozer',
        'bus',
        'calendar',
        'camel',
        'candle',
        'carrot',
        'church',
        'clarinet',
        'coffee cup',
        'eye',
        'firetruck',
        'flower',
        'golf club',
        'grapes',
        'hat',
        'hot dog',
        'ice cream',
        'knife',
        'lobster',
        'matches',
        'mermaid',
        'microwave',
        'nose',
        'panda',
        'paper clip',
        'pond',
        'purse',
        'radio',
        'roller coaster',
        'skyscraper',
        'smiley face',
        'snake',
        'steak',
        'stethoscope',
        'suitcase',
        'teddy-bear',
        'television',
        'tennis racquet',
        'toaster',
        'toilet',
        'tooth',
        'tornado',
        'trumpet',
        'waterslide',
        'wristwatch',
    ]
}


class Quickdraw(Dataset):

    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/vision/datasets/quickdraw.py)

    **Description**

    The Quickdraw dataset was originally introduced by Google Creative Lab in 2017 and then re-purposed for few-shot learning in Triantafillou et al., 2020.
    See Ha and Heck, 2017 for more information.

    The dataset consists of roughly 50M drawing images of 345 objects.
    Each image was hand-drawn by human annotators and is represented as black-and-white 28x28 pixel array.
    We follow the train-validation-test splits of Triantafillou et al., 2020.
    (241 classes for train, 52 for validation, and 52 for test.)


    **References**

    1. [https://github.com/googlecreativelab/quickdraw-dataset](https://github.com/googlecreativelab/quickdraw-dataset)
    2. Ha, David, and Douglas Eck. 2017. "A Neural Representation of Sketch Drawings." ArXiv '17.
    3. Triantafillou et al. 2020. "Meta-Dataset: A Dataset of Datasets for Learning to Learn from Few Examples." ICLR '20.

    **Arguments**

    * **root** (str) - Path to download the data.
    * **mode** (str, *optional*, default='train') - Which split to use.
        Must be 'train', 'validation', 'test', or 'custom'.
    * **transform** (Transform, *optional*, default=None) - Input pre-processing.
    * **target_transform** (Transform, *optional*, default=None) - Target pre-processing.
    * **download** (bool, *optional*, default=False) - Whether to download the dataset.

    **Example**

    ~~~python
    train_dataset = l2l.vision.datasets.Quickdraw(root='./data', mode='train')
    train_dataset = l2l.data.MetaDataset(train_dataset)
    train_generator = l2l.data.TaskDataset(dataset=train_dataset, num_tasks=1000)
    ~~~

    """

    def __init__(
        self,
        root,
        mode='all',
        transform=None,
        target_transform=None,
        download=False,
        labels_list=None
            ):
        root = os.path.expanduser(root)
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self._bookkeeping_path = os.path.join(self.root, 'quickdraw-' + mode + '-bookkeeping.pkl')

        if not self._check_exists() and download:
            self.download()

        self.load_bookkeeping(mode, labels_list)
        self.load_data(mode, labels_list)

    def _check_exists(self):
        if not os.path.exists(self.root):
            return False
        data_path = os.path.join(self.root, DATA_DIR)
        if not os.path.exists(data_path):
            return False
        all_classes = sum(SPLITS.values(), [])
        for cls_name in all_classes:
            cls_path = os.path.join(data_path, cls_name + '.npz')
            if not os.path.exists(cls_path):
                return False
        return True

    def download(self):
        if not os.path.exists(self.root):
            os.mkdir(self.root)
        data_path = os.path.join(self.root, DATA_DIR)
        if not os.path.exists(data_path):
            os.mkdir(data_path)
        print('Downloading Quickdraw dataset (10Gb)')
        all_classes = sum(SPLITS.values(), [])
        gcloud_url = GCLOUD_BUCKET + '*.npz'
        cmd = ['gsutil', '-m', 'cp', gcloud_url, data_path]
        subprocess.call(cmd)

    def load_bookkeeping(self, mode='all', labels_list=None):
        # We do manual bookkeeping, because the size of the dataset.
        if not os.path.exists(self._bookkeeping_path):
            # create bookkeeping
            data_path = os.path.join(self.root, DATA_DIR)
            splits = []
            if mode == 'all':
                splits = sum(SPLITS.values(), []) 
            elif mode[:6] == 'custom':
                if not labels_list:
                    raise ValueError("labels_list must be defined if mode is custom")
                print(labels_list)
                splits = labels_list
            else:
                splits = SPLITS[mode]
            labels = list(range(len(splits)))
            indices_to_labels = {}
            labels_to_indices = {}
            offsets = []
            index_counter = 0
            for cls_idx, cls_name in enumerate(splits):
                cls_path = os.path.join(data_path, cls_name + '.npz')
                cls_data = np.load(cls_path, encoding='latin1', allow_pickle=True)
                # use test from npz dataset for all? train/valid/test = 70k/2.5k/2.5k
                num_samples = len(cls_data['test'])
                labels_to_indices[cls_idx] = list(range(index_counter, index_counter + num_samples))
                for i in range(num_samples):
                    indices_to_labels[index_counter + i] = cls_idx
                offsets.append(index_counter)
                index_counter += num_samples
            bookkeeping = {
                'labels_to_indices': labels_to_indices,
                'indices_to_labels': indices_to_labels,
                'labels': labels,
                'offsets': offsets,
            }
            # Save bookkeeping to disk
            with open(self._bookkeeping_path, 'wb') as f:
                pickle.dump(bookkeeping, f, protocol=-1)
        else:
            with open(self._bookkeeping_path, 'rb') as f:
                bookkeeping = pickle.load(f)
        self._bookkeeping = bookkeeping
        self.labels_to_indices = bookkeeping['labels_to_indices']
        self.indices_to_labels = bookkeeping['indices_to_labels']
        self.labels = bookkeeping['labels']
        self.offsets = bookkeeping['offsets']

    def load_data(self, mode='train', labels_list=None):
        data_path = os.path.join(self.root, DATA_DIR)
        splits = []
        if mode == 'all':
            splits = sum(SPLITS.values(), []) 
        elif mode[:6] == 'custom':
            if not labels_list:
                raise ValueError("labels_list must be defined if mode is custom")
            print(labels_list)
            splits = labels_list
        else:
            splits = SPLITS[mode]
        self.data = []
        for cls_name in splits:
            cls_path = os.path.join(data_path, cls_name + '.npz')
            # self.data.append(np.load(cls_path, mmap_mode='r'))
            self.data.append(np.load(cls_path, encoding='latin1', allow_pickle=True)['test'])

    def __getitem__(self, i):
        label = self.indices_to_labels[i]
        cls_data = self.data[label]
        offset = self.offsets[label]
        image = cls_data[i - offset]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def __len__(self):
        return len(self.indices_to_labels)


if __name__ == '__main__':
    # import torchvision as tv

    # train_dataset = Quickdraw(root='~/data',
    #                                           transform=tv.transforms.ToTensor(),
    #                                           mode='train') # add download=True for first run
    # valid_dataset = Quickdraw(root='~/data',
    #                                           transform=tv.transforms.ToTensor(),
    #                                           mode='validation')
    # test_dataset = Quickdraw(root='~/data',
    #                                           transform=tv.transforms.ToTensor(),
    #                                          mode='test')
    # print("Done")
    qd = Quickdraw(root='~/data', download=True)
    img, lab = qd[len(qd) - 1]
    print(img)
    print(lab)
