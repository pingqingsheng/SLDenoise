from distutils.command.bdist import show_formats
import os
import sys
import random
from collections import defaultdict
from typing import List, Callable, Optional, Any
sys.path.append("./")
sys.path.append("../")

import torch
import torch.utils.data as data
from torch.utils.data import Sampler, BatchSampler, SequentialSampler
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from tqdm import tqdm 
from termcolor import cprint

from network.network import resnet18

def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

def accimage_loader(path: str) -> Any:
    import accimage
    try:
        return accimage.Image(path)
    except OSError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path: str) -> Any:
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)

def parseClasses(file):
    classes = []
    filenames = []
    with open(file) as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]
    for x in range(0,len(lines)):
        tokens = lines[x].split()
        classes.append(tokens[1])
        filenames.append(tokens[0])
    return filenames,classes

def load_allimages(dir):
    images = []
    if not os.path.isdir(dir):
        sys.exit(-1)
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            #if datasets.folder.is_image_file(fname):
            if datasets.folder.has_file_allowed_extension(fname,['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']):
                path = os.path.join(root, fname)
                item = path
                images.append(item)
    return images


class PairBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, num_iterations=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_iterations = num_iterations

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        random.shuffle(indices)
        for k in range(len(self)):
            if self.num_iterations is None:
                offset = k*self.batch_size
                batch_indices = indices[offset:offset+self.batch_size]
            else:
                batch_indices = random.sample(range(len(self.dataset)), self.batch_size)
            pair_indices = []
            for idx in batch_indices:
                y = self.dataset.get_class(idx)
                pair_indices.append(random.choice(self.dataset.classwise_indices[y]))
            yield batch_indices + pair_indices

    def __len__(self):
        if self.num_iterations is None:
            return (len(self.dataset)+self.batch_size-1) // self.batch_size
        else:
            return self.num_iterations

class TImgNetDatasetTrain(datasets.ImageFolder):
    
    def __init__(self, 
                root: str, 
                transform: Optional[Callable] = None, 
                target_transform: Optional[Callable] = None, 
                loader: Callable[[str], Any] = default_loader, 
                is_valid_file: Optional[Callable[[str], bool]] = None, 
                train_valid_split=0.9, 
                split = 'train'):
        super().__init__(root, transform, target_transform, loader, is_valid_file)
        self.train_valid_split = train_valid_split
        self.split = split

        self._train_valid_split(self.train_valid_split)

        self.classwise_indices = defaultdict(list)
        for i in range(len(self)):
            y = self.targets[i]
            self.classwise_indices[y].append(i)

    def _train_valid_split(self, train_valid_ratio):
        n = len(self.imgs)
        _train_indices = np.random.choice(range(n), int(n*train_valid_ratio), replace=False)
        _valid_indices = np.array(list(set(list(range(n))) - set(_train_indices.tolist())))
        if self.split == 'train':
            self.samples = [self.samples[i] for i in _train_indices]
            self.imgs = [self.imgs[i] for i in _train_indices]
            self.targets = [self.targets[i] for i in _train_indices]
        elif self.split == 'valid':
            self.samples = [self.samples[i] for i in _valid_indices]
            self.imgs = [self.imgs[i] for i in _valid_indices]
            self.targets = [self.targets[i] for i in _valid_indices]
        else:
            raise ValueError(f'Split {self.split} not defined !')

    def update_labels(self, new_targets):
        self.targets[:] = new_targets[:]

    def get_class(self, indice):
        return self.targets[indice]

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return index, sample, target


class TImgNetDatasetTest(data.Dataset):
    """Dataset wrapping images and ground truths."""
    
    def __init__(self, img_path, gt_path, class_to_idx=None, transform=None):
        self.img_path = img_path
        self.transform = transform
        self.gt_path = gt_path
        self.class_to_idx = class_to_idx
        self.targets = []
        self.imgs, self.classnames = parseClasses(gt_path)
        for classname in self.classnames:
            self.targets.append(self.class_to_idx[classname])

    def __getitem__(self, index):
            """
            Args:
                index (int): Index
            Returns:
                tuple: (image, y) where y is the label of the image.
            """
            img = None
            with open(os.path.join(self.img_path, self.imgs[index]), 'rb') as f:
                img = Image.open(f)
                img = img.convert('RGB')
                if self.transform is not None:
                    img = self.transform(img)
            y = self.targets[index]
            return index, img, y

    def __len__(self):
        return len(self.imgs)
    

if __name__ == '__main__':
    
    # Experiment Setting Control Panel
    SEED: int = 123
    N_EPOCH: int = 8
    LR: float = 1e-6
    WEIGHT_DECAY: float = 1e-2
    BATCH_SIZE: int = 128
    SCHEDULER_DECAY_MILESTONE: List = [20, 40, 60]
    TRAIN_VALIDATION_RATIO: float = 0.8
    MONITOR_WINDOW: int = 2
    GPU_IND: str = "2"
    
    DATADIR = "/data/songzhu/tinyimagenet/tiny-imagenet-200"
    
    seed = SEED
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # need to set to True as well

    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_IND
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    
    traindir = os.path.join(DATADIR, 'train')
    testdir  = os.path.join(DATADIR, 'val')
    
    train_transforms = transforms.Compose([
        transforms.Resize(224), 
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transforms = transforms.Compose([
        transforms.Resize(224), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    trainset = TImgNetDatasetTrain(traindir, train_transforms)
    testset = TImgNetDatasetTest(os.path.join(testdir, 'images'), 
                            os.path.join(testdir, 'val_annotations.txt'), 
                            class_to_idx=trainset.class_to_idx.copy(), 
                            transform=test_transforms)
    
    train_loader = data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=2)
    test_loader  = data.DataLoader(testset,  batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=2)
    
    model_cls = resnet18(num_classes=len(trainset.classes), in_channels=3, pretrained=True)
    model_cls = model_cls.to(device)

    optimizer_cls = torch.optim.Adam(model_cls.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler_cls = torch.optim.lr_scheduler.MultiStepLR(optimizer_cls, gamma=0.5, milestones=SCHEDULER_DECAY_MILESTONE)

    criterion_cls = torch.nn.CrossEntropyLoss()

    for inner_epoch in range(N_EPOCH):
        train_correct = 0
        train_total = 0
        train_loss = 0

        for _, (images, labels) in enumerate(tqdm(train_loader, ascii=True, ncols=100)):
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
            for _, (images, labels) in enumerate(tqdm(test_loader, ascii=True, ncols=100)):
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
    for _, (images, labels) in enumerate(test_loader):
        if images.shape[0] == 1:
            continue
        images, labels = images.to(device), labels.to(device)
        outs = model_cls(images)
        _, predict = outs.max(1)
        test_correct += predict.eq(labels).sum().item()
        test_total += len(labels)
    cprint(f"Classification Test Acc: {test_correct / test_total:7.3f}", "cyan")

    # Save clean model
    model_file_name = "TINYIMAGENET_resnet18_clean.pth"
    torch.save(model_cls, model_file_name)
    
    
    
