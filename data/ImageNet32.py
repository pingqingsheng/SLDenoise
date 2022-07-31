import os
from typing import Any

import torch
from torchvision import datasets, transforms
import torch.utils.data as data_utils
import numpy as np
from PIL import Image
import pickle 


# For OOD set only
TRAIN_BATCH_SIZE : int = 128
TEST_BATCH_SIZE  : int = 128
IMAGENET32_PATH : str = '/scr/songzhu'


def ImageNet32(train=True, dataset='cifar10', batch_size=None, augm_flag=True, val_size=None, meanstd=None):
    assert dataset in ['mnist', 'fmnist', 'cifar10', 'svhn', 'cifar100'], 'Invalid dataset.'

    if batch_size==None:
        if train:
            batch_size=TRAIN_BATCH_SIZE
        else:
            batch_size=TEST_BATCH_SIZE

    if dataset in ['mnist', 'fmnist']:
        img_size = 28
        transform_base = [
            transforms.Resize(img_size),
            transforms.Grayscale(1),  # Single-channel grayscale image
            transforms.ToTensor()
        ]

        if meanstd is not None:
            transform_base.append(transforms.Normalize(*meanstd))

        transform_train = transforms.Compose(
            [transforms.RandomCrop(28, padding=2)] + transform_base
        )
    else:
        img_size = 32
        transform_base = [transforms.ToTensor()]

        if meanstd is not None:
            transform_base.append(transforms.Normalize(*meanstd))

        padding_mode = 'edge' if dataset == 'svh' else 'reflect'
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(img_size, padding=4, padding_mode=padding_mode),
            ] + transform_base
        )

    transform_test = transforms.Compose(transform_base)
    transform_train = transforms.RandomChoice([transform_train, transform_test])
    transform = transform_train if (augm_flag and train) else transform_test

    dataset = ImageNet32Dataset(IMAGENET32_PATH, train=train, transform=transform)

    if train or val_size is None:
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=train, num_workers=4, pin_memory=True)
        return loader
    else:
        # Split into val and test sets
        test_size = len(dataset) - val_size
        dataset_val, dataset_test = data_utils.random_split(dataset, (val_size, test_size))
        val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size,
                                                 shuffle=train, num_workers=4, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size,
                                                 shuffle=train, num_workers=4, pin_memory=True)
        return val_loader, test_loader


class ImageNet32Dataset(datasets.VisionDataset):
    """ https://arxiv.org/abs/1707.08819 """

    base_folder = 'imagenet32'
    train_list = [f'train_data_batch_{i}' for i in range(1, 11)]
    test_list = ['val_data']

    def __init__(self, root, train=True, transform=None, target_transform=None):

        super(ImageNet32Dataset, self).__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set
        self.offset = 0  # offset index---for inducing randomness

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                self.targets.extend(entry['labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index: int):
        # Shift the index by an offset, which can be chosen randomly
        index = (index + self.offset) % len(self)

        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")
    
    
if __name__ == '__main__':
    
    ood_train_loader = ImageNet32(dataset='CIFAR10', train=True)
    ood_val_loader = ImageNet32(dataset='CIFAR10', train=False)