from distutils.command.bdist import show_formats
import os
import sys
import random
from typing import List
sys.path.append("../")

import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from tqdm import tqdm 
from termcolor import cprint

from network.network import resnet18

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

class TImgNetDataset(data.Dataset):
    """Dataset wrapping images and ground truths."""
    
    def __init__(self, img_path, gt_path, class_to_idx=None, transform=None):
        self.img_path = img_path
        self.transform = transform
        self.gt_path = gt_path
        self.class_to_idx = class_to_idx
        self.classidx = []
        self.imgs, self.classnames = parseClasses(gt_path)
        for classname in self.classnames:
            self.classidx.append(self.class_to_idx[classname])

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
            y = self.classidx[index]
            return img, y

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
    
    DATADIR = "/scr/songzhu/tinyimagenet/tiny-imagenet-200"
    
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
    
    valtest_transforms = transforms.Compose([
        transforms.Resize(224), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    trainset = datasets.ImageFolder(traindir, train_transforms)
    testset = TImgNetDataset(os.path.join(testdir, 'images'), 
                              os.path.join(testdir, 'val_annotations.txt'), 
                              class_to_idx=trainset.class_to_idx.copy())
    
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
    
    
    
