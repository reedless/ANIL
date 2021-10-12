from utils import create_image
import numpy as np
from quickdraw import Quickdraw
import os 

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

output_dataset_dir = "./dataset"

def etl_samples_and_save(dataset, dirname, skip=False):
    prev_label = 0
    counter = 0
    print(prev_label)
    for sample, label in dataset:
        if prev_label != label:
            print(label)
            prev_label = label
            counter = 0

        # check if already saved and if skip is True
        outfile_name = os.path.join(output_dataset_dir, dirname, f'{label}_{counter}.npz')
        counter += 1
        if os.path.exists(outfile_name) and skip:
            continue

        image = create_image(sample,
                            accumulate_strokes=True, # default
                            output_dims=(224, 224))  # default
        outfile = open(outfile_name, 'wb')
        np.savez_compressed(outfile, image=image, label=label)
        outfile.close()

    return True


def main():
    # Create Datasets (too large to use entire quickdraw)
    train_dataset = Quickdraw(root='~/data',
                                # transform=tv.transforms.ToTensor(),
                                mode='custom-train',
                                labels_list=SPLITS['train'],
                                download=True)
    valid_dataset = Quickdraw(root='~/data',
                                # transform=tv.transforms.ToTensor(),
                                mode='custom-validation',
                                labels_list=SPLITS['validation'])
    test_dataset = Quickdraw(root='~/data',
                                # transform=tv.transforms.ToTensor(),
                                mode='custom-test',
                                labels_list=SPLITS['test'])

    print(len(train_dataset)) # 2.5k * 70 = 175000
    print(len(valid_dataset)) # 2.5k * 15 = 37500
    print(len(test_dataset))  # 2.5k * 15 = 37500
    print(len(train_dataset.data[0])) # a class w 2.5k samples
    print(create_image(train_dataset.data[0][0]).shape) # a sample

    # create directories
    if not os.path.exists(output_dataset_dir):
        os.mkdir(output_dataset_dir)
    for partition in ['train', 'valid', 'test']:
        directory = os.path.join(output_dataset_dir, partition)
        if not os.path.exists(directory):
            os.mkdir(directory)

    """
    Ignore KeyErrors, will occur when finish processing all samples
    Example:
    File "process_dataset.py", line 222, in <module>
        main()
    File "process_dataset.py", line 169, in main
        etl_samples_and_save(valid_dataset, 'valid')
    File "process_dataset.py", line 121, in etl_samples_and_save
        for sample, label in dataset:
    File "/home/reedless/anil/quickdraw.py", line 527, in __getitem__
        label = self.indices_to_labels[i]
    KeyError: 37500
    """
    etl_samples_and_save(train_dataset, 'train')
    etl_samples_and_save(valid_dataset, 'valid')
    etl_samples_and_save(test_dataset, 'test')

    # # get max number of strokes, need to pad all stroke_seqs to be the same
    # max_strokes = 0
    # all_classes = np.concatenate((train_dataset.data, valid_dataset.data, test_dataset.data))
    # for class_set in all_classes:
    #     for stroke_seq in class_set:
    #         ml = stroke_seq.shape[0]
    #         if ml > max_strokes:
    #             max_strokes = ml

    # print(max_strokes) # 227

    # padded_train_dataset = pad_dataset(train_dataset, max_strokes)
    # padded_valid_dataset = pad_dataset(valid_dataset, max_strokes)
    # padded_test_dataset  = pad_dataset(test_dataset, max_strokes)

    # print(len(padded_train_dataset))
    # print(len(padded_valid_dataset))
    # print(len(padded_test_dataset))
    # print(padded_train_dataset[0][0].size())
    # print(padded_valid_dataset[0][0].size())
    # print(padded_test_dataset[0][0].size())




if __name__ == '__main__':
    # sanity check
    for label in SPLITS['train']:
        if label in SPLITS['test']:
            print('test: ' + label)
    for label in SPLITS['validation']:
        if label in SPLITS['test']:
            print('test: ' + label)
    main()