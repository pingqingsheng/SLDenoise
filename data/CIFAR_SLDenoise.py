from PIL import Image
import os
import os.path
import numpy as np
import sys
sys.path.append("../")
sys.path.append("../../")

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch
import torch.utils.data as data
from utils.utils import download_url, check_integrity


class CIFAR10_SLDenoise(data.Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, root, split='train', train_ratio=0.8, trust_ratio=0.1,
                 transform=None, target_transform=None,
                 download=False,
                 mode='normal',
                 n_sample=1):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.split = split  # training set, validation set or test set
        self.train_ratio = train_ratio
        self.trust_ratio = trust_ratio
        self.mode = mode
        self.n_sample = n_sample

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.split == 'test':
            downloaded_list = self.test_list
        else:
            downloaded_list = self.train_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        eps = 0.001

        # split the original train set into train & validation set
        if self.split != 'test':
            num_data = len(self.data)
            trust_num = int(num_data * self.trust_ratio)
            self.num_class = len(np.unique(self.targets))

            remain = num_data - trust_num
            train_num = int(remain * self.train_ratio)
            if self.split == 'train':
                self.data = self.data[:train_num]
                self.targets = self.targets[:train_num]
                # Add softlabel here
                self.softlabel = np.ones([train_num, self.num_class], dtype=np.float32)*eps/self.num_class
                for i in range(train_num):
                    self.softlabel[i, self.targets[i]] = 1 - eps
            elif self.split == 'val':
                self.data = self.data[train_num:remain]
                self.targets = self.targets[train_num:remain]
                self.softlabel = np.ones([(remain-train_num), self.num_class], dtype=np.float32)*eps/self.num_class
                for i in range(remain - train_num):
                    self.softlabel[i, self.targets[i]] = 1 - eps
            else:
                self.data = self.data[remain:]
                self.targets = self.targets[remain:]
                self.softlabel = np.ones([(num_data-remain), self.num_class], dtype=np.float32)*eps/self.num_class
                for i in range(num_data-remain):
                    self.softlabel[i, self.targets[i]] = 1 - eps
        else:
            num_data = len(self.data)
            self.num_class = len(np.unique(self.targets))
            self.softlabel = np.ones([num_data, self.num_class], dtype=np.float32)*eps/self.num_class
            for i in range(num_data):
                self.softlabel[i, self.targets[i]] = 1 - eps

        self.delta_eta = torch.zeros(len(self.targets), 10)

        self._load_meta()

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, delta_eta = self.data[index], self.targets[index], self.delta_eta[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        if self.transform is not None:
            if self.mode == 'normal':
                img = self.transform(img)
            elif self.mode == 'sample':
                img = torch.cat([self.transform(img).unsqueeze(0) for i in range(self.n_sample)], 0)
            else:
                raise ValueError('Mode not exists!')
        if self.target_transform is not None:
            target = self.target_transform(target)

        return index, img, target, delta_eta

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        # extract file
        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

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

    def demean(self):
        m = self.data.mean((0, 1))
        for i in range(self.data.shape[0]):
            self.data[i, 0] = self.data[i, 0] - m
            self.data[i, 1] = self.data[i, 1] - m
            self.data[i, 2] = self.data[i, 2] - m

    def update_labels(self, new_label):
        self.targets[:] = new_label[:]

    def set_delta_eta(self, delta_eta):
        self.delta_eta = delta_eta

    def sample(self):
        self.mode = 'sample'

    def normal(self):
        self.mode = 'normal'

class CIFAR100(CIFAR10_SLDenoise):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }


# Train model with clean data
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
    import pickle as pkl

    from data.CIFAR import CIFAR10
    from network.network import resnet18
    from utils.utils import _init_fn

    # Experiment Setting Control Panel
    SEED: int = 123
    N_EPOCH: int = 60
    LR: float = 1e-3
    WEIGHT_DECAY: float = 5e-3
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    # Data Loading and Processing
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = CIFAR10(root="./data", split="train", train_ratio=TRAIN_VALIDATION_RATIO, download=True, transform=transform_train)
    validset = CIFAR10(root="./data", split="valid", train_ratio=TRAIN_VALIDATION_RATIO, download=True, transform=transform_train)
    testset  = CIFAR10(root='./data', split="test", download=True, transform=transform_test)
    # model_cls_clean = torch.load("./data/CIFAR10_resnet18_clean.pth")

    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, worker_init_fn=_init_fn(worker_id=seed))
    valid_loader = DataLoader(validset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, worker_init_fn=_init_fn(worker_id=seed))
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, worker_init_fn=_init_fn(worker_id=seed))

    model_cls = resnet18(num_classes=10, in_channels=3)
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
    model_file_name = "CIFAR10_resnet18_clean.pth"
    torch.save(model_cls, model_file_name)
