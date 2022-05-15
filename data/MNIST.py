from __future__ import print_function
import os
import sys
sys.path.append( os.path.abspath("../"))
import torch.utils.data as data
from PIL import Image
import os.path
import gzip
import numpy as np
import torch
import codecs

from utils.utils import download_url, makedir_exist_ok


class MNIST(data.Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.
    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    urls = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    ]
    training_file = 'training.pt'
    test_file = 'test.pt'
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    def __init__(self, root, split='train', train_ratio=0.9, transform=None, target_transform=None, download=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.split = split  # training set or test set
        self.train_ratio = train_ratio

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +' You can use download=True to download it')

        if self.split == 'test':
            data_file = self.test_file
        else:
            data_file = self.training_file
        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))
        self.targets = self.targets.numpy().tolist()
        self.num_class = len(np.unique(self.targets))
        self.num_data = len(self.data)

        # split the original train set into train & validation set
        if self.split != 'test':
            num_data = len(self.data)
            train_num = int(num_data * self.train_ratio)
            if self.split == 'train':
                self.data = self.data[:train_num]
                self.targets = self.targets[:train_num]
                self.num_class = len(np.unique(self.targets))
                self.num_data = len(self.data)
            else:
                self.data = self.data[train_num:]
                self.targets = self.targets[train_num:]
                self.num_class = len(np.unique(self.targets))
                self.num_data = len(self.data)
        self.delta_eta = torch.zeros(len(self.targets), 10)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, delta_eta = self.data[index], int(self.targets[index]), self.delta_eta[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return index, img, target, delta_eta

    def __len__(self):
        return len(self.data)

    @property
    def raw_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self):
        return os.path.exists(os.path.join(self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.processed_folder, self.test_file))

    @staticmethod
    def extract_gzip(gzip_path, remove_finished=False):
        print('Extracting {}'.format(gzip_path))
        with open(gzip_path.replace('.gz', ''), 'wb') as out_f, \
                gzip.GzipFile(gzip_path) as zip_f:
            out_f.write(zip_f.read())
        if remove_finished:
            os.unlink(gzip_path)

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""

        if self._check_exists():
            return

        makedir_exist_ok(self.raw_folder)
        makedir_exist_ok(self.processed_folder)

        # download files
        for url in self.urls:
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.raw_folder, filename)
            download_url(url, root=self.raw_folder, filename=filename, md5=None)
            self.extract_gzip(gzip_path=file_path, remove_finished=True)

        # process and save as torch files
        print('Processing...')

        training_set = (
            read_image_file(os.path.join(self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = self.split
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def update_labels(self, new_label):
        self.targets[:] = new_label[:]

    def set_delta_eta(self, delta_eta):
        self.delta_eta = delta_eta


def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        return torch.from_numpy(parsed).view(length).long()


def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        return torch.from_numpy(parsed).view(length, num_rows, num_cols)


class MNIST_Combo(MNIST):

    def __init__(self, root, exogeneous_var, split='train', train_ratio=0.9, transform=None, target_transform=None, download=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.split = split  # training set or test set
        self.train_ratio = train_ratio

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +' You can use download=True to download it')

        if self.split == 'test':
            data_file = self.test_file
        else:
            data_file = self.training_file
        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))
        self.targets = self.targets.numpy().tolist()
        self.num_class = len(np.unique(self.targets))
        self.num_data = len(self.data)
        self.exogeneous_var = exogeneous_var

        # split the original train set into train & validation set
        if self.split != 'test':
            num_data = len(self.data)
            train_num = int(num_data * self.train_ratio)
            if self.split == 'train':
                self.data = self.data[:train_num]
                self.targets = self.targets[:train_num]
                self.num_class = len(np.unique(self.targets))
                self.num_data = len(self.data)
            else:
                self.data = self.data[train_num:]
                self.targets = self.targets[train_num:]
                self.num_class = len(np.unique(self.targets))
                self.num_data = len(self.data)
        self.delta_eta = torch.zeros(len(self.targets), 10)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, delta_eta, exogeneous_var = self.data[index], int(self.targets[index]), self.delta_eta[index], self.exogeneous_var[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return index, img, target, delta_eta, exogeneous_var


# Train model with clean examples to generate IDL synthetic dataset
if __name__ == "__main__":

    from typing import List

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torch.nn import DataParallel
    from torchvision import transforms
    import numpy as np
    import random
    from tqdm import tqdm
    import copy
    from termcolor import cprint

    from data.MNIST import MNIST
    from network.network import resnet18
    from utils.utils import _init_fn

    # Experiment Setting Control Panel
    SEED: int = 123
    N_EPOCH: int = 8
    LR: float = 1e-6
    WEIGHT_DECAY: float = 1e-2
    BATCH_SIZE: int = 128
    SCHEDULER_DECAY_MILESTONE: List = [20, 40, 60]
    TRAIN_VALIDATION_RATIO: float = 0.8
    MONITOR_WINDOW: int = 2
    GPU_IND: str = "0"

    seed = SEED
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # need to set to True as well

    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_IND
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    # Data Loading and Processing
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    trainset = MNIST(root="./data", split="train", train_ratio=TRAIN_VALIDATION_RATIO, download=True, transform=transform_train)
    validset = MNIST(root="./data", split="valid", train_ratio=TRAIN_VALIDATION_RATIO, download=True, transform=transform_train)
    testset  = MNIST(root='./data', split="test", download=True, transform=transform_test)
    # model_cls_clean = torch.load("./data/CIFAR10_resnet18_clean.pth")

    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, worker_init_fn=_init_fn(worker_id=seed))
    valid_loader = DataLoader(validset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, worker_init_fn=_init_fn(worker_id=seed))
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, worker_init_fn=_init_fn(worker_id=seed))

    model_cls = resnet18(num_classes=10, in_channels=1)
    model_cls = DataParallel(model_cls)
    model_cls = model_cls.to(device)

    optimizer_cls = torch.optim.Adam(model_cls.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler_cls = torch.optim.lr_scheduler.MultiStepLR(optimizer_cls, gamma=0.5, milestones=SCHEDULER_DECAY_MILESTONE)

    criterion_cls = torch.nn.CrossEntropyLoss()

    for inner_epoch in range(N_EPOCH):
        train_correct = 0
        train_total = 0
        train_loss = 0
        for _, (indices, images, labels, _) in enumerate(tqdm(train_loader, ascii=True, ncols=100)):
            if images.shape[0] == 1:
                continue
            optimizer_cls.zero_grad()
            images, labels = images.to(device), labels.to(device)
            outs = model_cls(images)
            conf = torch.softmax(outs, 1)
            loss = criterion_cls(outs, labels)
            loss.backward()
            optimizer_cls.step()

            train_loss += loss.detach().cpu().item()
            _, predict = outs.max(1)
            train_correct += predict.eq(labels).sum().item()
            train_total += len(labels)

        train_acc = train_correct / train_total

        if not (inner_epoch + 1) % MONITOR_WINDOW:

            valid_correct = 0
            valid_total = 0
            model_cls.eval()
            for _, (indices, images, labels, _) in enumerate(tqdm(valid_loader, ascii=True, ncols=100)):
                if images.shape[0] == 1:
                    continue
                images, labels = images.to(device), labels.to(device)
                outs = model_cls(images)

                _, predict = outs.max(1)
                valid_correct += predict.eq(labels).sum().item()
                valid_total += len(labels)

            valid_acc = valid_correct / valid_total
            print(f"Step [{inner_epoch + 1}|{N_EPOCH}] - Train Loss: {train_loss / train_total:7.3f} - Train Acc: {train_acc:7.3f} - Valid Acc: {valid_acc:7.3f}")
            model_cls.train()  # switch back to train mode
        scheduler_cls.step()

    # Classification Final Test
    test_correct = 0
    test_total = 0
    model_cls.eval()
    for _, (indices, images, labels, _) in enumerate(test_loader):
        if images.shape[0] == 1:
            continue
        images, labels = images.to(device), labels.to(device)
        outs = model_cls(images)
        _, predict = outs.max(1)
        test_correct += predict.eq(labels).sum().item()
        test_total += len(labels)
    cprint(f"Classification Test Acc: {test_correct / test_total:7.3f}", "cyan")

    # Save clean model
    model_file_name = "MNIST_resnet18_clean.pth"
    torch.save(model_cls, model_file_name)