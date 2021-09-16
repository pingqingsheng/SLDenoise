import os
from typing import List
import pdb

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from torchvision import transforms
import numpy as np
import argparse
import random
from tqdm import tqdm
import copy
from termcolor import cprint

from data.MNIST import MNIST
from network.network import resnet18
from utils.utils import _init_fn
from utils.noise import perturb_eta

# Experiment Setting Control Panel
N_EPOCH_OUTER: int = 5
N_EPOCH_INNER: int = 4
LR: float = 1e-3
WEIGHT_DECAY: float = 5e-4
BATCH_SIZE: int = 128
SCHEDULER_DECAY_MILESTONE: List = [20, 40, 60]
TRAIN_VALIDATION_RATIO: float = 0.8
MONITOR_WINDOW: int = 2

def label_correction(y_tilde, eta_corrected):
    pass

def main(args):

    seed = args.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # need to set to True as well

    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

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
    validset = MNIST(root="./data", split="valid", train_ratio=TRAIN_VALIDATION_RATIO, transform=transform_train)
    testset  = MNIST(root="./data", split="test",  download=True, transform=transform_test)

    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, worker_init_fn=_init_fn(worker_id=seed))
    valid_loader = DataLoader(validset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, worker_init_fn=_init_fn(worker_id=seed))
    test_loader  = DataLoader(testset,  batch_size=BATCH_SIZE, shuffle=True, num_workers=2, worker_init_fn=_init_fn(worker_id=seed))

    # Inject noise here
    y_train = copy.deepcopy(trainset.targets)
    y_valid = copy.deepcopy(validset.targets)
    y_test  = testset.targets
    model_cls_clean = torch.load("./data/MNSIT_resnet18_clean.pth")
    cprint(">>> Inject Noise <<<", "green")
    _eta_train_temp_pair = [(torch.softmax(model_cls_clean(images).to(device), 1).detach().cpu(), indices)  for _, (indices, images, labels, _) in enumerate(tqdm(train_loader, ascii=True, ncols=100))]
    _eta_valid_temp_pair = [(torch.softmax(model_cls_clean(images).to(device), 1).detach().cpu(), indices)  for _, (indices, images, labels, _) in enumerate(tqdm(valid_loader, ascii=True, ncols=100))]
    _eta_test_temp_pair  = [(torch.softmax(model_cls_clean(images).to(device), 1).detach().cpu(), indices)  for _, (indices, images, labels, _) in enumerate(tqdm(test_loader, ascii=True, ncols=100))]
    _eta_train_temp = torch.cat([x[0] for x in _eta_train_temp_pair])
    _eta_valid_temp = torch.cat([x[0] for x in _eta_valid_temp_pair])
    _eta_test_temp  = torch.cat([x[0] for x in _eta_test_temp_pair])
    _train_indices = torch.cat([x[1] for x in _eta_train_temp_pair]).squeeze()
    _valid_indices = torch.cat([x[1] for x in _eta_valid_temp_pair]).squeeze()
    _test_indices = torch.cat([x[1] for x in _eta_test_temp_pair]).squeeze()
    eta_train = _eta_train_temp[_train_indices.argsort()]
    eta_valid = _eta_valid_temp[_valid_indices.argsort()]
    eta_test  = _eta_test_temp[_test_indices.argsort()]
    h_star_train = eta_train.argmax(1).squeeze()
    h_star_valid = eta_valid.argmax(1).squeeze()
    h_star_test  = eta_test.argmax(1).squeeze()
    eta_tilde_train = perturb_eta(eta_train, args.noise_type, args.noise_strength)
    eta_tilde_valid = perturb_eta(eta_valid, args.noise_type, args.noise_strength)
    y_tilde_train = [int(np.where(np.random.multinomial(1,x,1).squeeze())[0]) for x in eta_tilde_train]
    y_tilde_valid = [int(np.where(np.random.multinomial(1,x,1).squeeze())[0]) for x in eta_tilde_valid]
    trainset.update_labels(y_tilde_train)
    validset.update_labels(y_tilde_valid)
    train_noise_ind = np.where(np.array(y_train)!=np.array(y_tilde_train))[0]
    valid_noise_ind = np.where(np.array(y_valid)!=np.array(y_tilde_valid))[0]

    print("---------------------------------------------------------")
    print("                   Experiment Setting                    ")
    print("---------------------------------------------------------")
    print(f"Network: \t\t\t ResNet18")
    print(f"Number of Outer Epochs: \t {N_EPOCH_OUTER}")
    print(f"Number of Inner Epochs: \t {N_EPOCH_INNER}")
    print(f"Learning Rate: \t\t\t {LR}")
    print(f"Weight Decay: \t\t\t {WEIGHT_DECAY}")
    print(f"Batch Size: \t\t\t {BATCH_SIZE}")
    print(f"Scheduler Milestones: \t\t {SCHEDULER_DECAY_MILESTONE}")
    print(f"Monitor Window: \t\t {MONITOR_WINDOW}")
    print("---------------------------------------------------------")
    print(f"Number of Training Data Points: \t\t {len(trainset)}")
    print(f"Number of Validation Data Points: \t\t {len(validset)}")
    print(f"Number of Testing Data Points: \t\t\t {len(testset)}")
    print(f"Train - Validation Ratio: \t\t\t {TRAIN_VALIDATION_RATIO}")
    print(f"Clean Training Data Points: \t\t\t {len(trainset)-len(train_noise_ind)}")
    print(f"Clean Validation Data Points: \t\t\t {len(validset)-len(valid_noise_ind)}")
    print(f"Noisy Training Data Points: \t\t\t {len(train_noise_ind)}")
    print(f"Noisy Validation Data Points: \t\t\t {len(valid_noise_ind)}")
    print(f"Noisy Type: \t\t\t\t\t {args.noise_type}")
    print(f"Noisy Strength: \t\t\t\t {args.noise_strength}")
    print("---------------------------------------------------------")

    model_cls = resnet18(num_classes=10, in_channels=1)
    model_conf = resnet18(num_classes=10, in_channels=1)
    model_cls = DataParallel(model_cls)
    model_conf = DataParallel(model_conf)
    model_cls, model_conf = model_cls.to(device), model_conf.to(device)

    optimizer_cls  = torch.optim.Adam(model_cls.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    optimizer_conf = torch.optim.Adam(model_conf.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler_cls  = torch.optim.lr_scheduler.MultiStepLR(optimizer_cls, gamma=0.5, milestones=SCHEDULER_DECAY_MILESTONE)
    scheduler_conf = torch.optim.lr_scheduler.MultiStepLR(optimizer_conf, gamma=0.5, milestones=SCHEDULER_DECAY_MILESTONE)

    criterion_cls = torch.nn.CrossEntropyLoss()
    criterion_conf = torch.nn.MSELoss()

    train_conf_delta = torch.zeros([len(trainset), 10])
    train_conf = torch.zeros([len(trainset), 10])
    valid_conf_delta = torch.zeros([len(validset), 10])
    valid_conf = torch.zeros([len(validset), 10])
    test_conf_delta = torch.zeros([len(testset), 10])
    test_conf  = torch.zeros([len(testset), 10])

    for outer_epoch in range(N_EPOCH_OUTER):

        cprint(f">>>Epoch [{outer_epoch+1}|{N_EPOCH_OUTER}] Train Classifier Model <<<", "green")
        model_cls.train()
        for inner_epoch in range(N_EPOCH_INNER):
            train_correct = 0
            train_total = 0
            train_loss = 0
            for _, (indices, images, labels, _) in enumerate(tqdm(train_loader, ascii=True, ncols=100)):
                if images.shape[0]==1:
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

                onehot_labels  = F.one_hot(labels, num_classes=10)
                onehot_predict = F.one_hot(predict, num_classes=10)
                train_conf_delta[indices] = torch.abs(onehot_predict - onehot_labels).detach().cpu().float()
                train_conf[indices] = conf.detach().cpu()

            train_acc = train_correct/train_total

            if not (inner_epoch+1)%MONITOR_WINDOW:

                valid_correct = 0
                valid_total = 0
                model_cls.eval()
                for _, (indices, images, labels, _) in enumerate(tqdm(valid_loader, ascii=True, ncols=100)):
                    if images.shape[0]==1:
                        continue
                    images, labels = images.to(device), labels.to(device)
                    outs = model_cls(images)

                    _, predict = outs.max(1)
                    valid_correct += predict.eq(labels).sum().item()
                    valid_total += len(labels)

                    onehot_labels = F.one_hot(labels, num_classes=10)
                    onehot_predict = F.one_hot(predict, num_classes=10)
                    valid_conf_delta[indices] = torch.abs(onehot_labels - onehot_predict).detach().cpu().float()
                    valid_conf[indices] = torch.softmax(outs, 1).detach().cpu()

                valid_acc = valid_correct/valid_total
                print(f"Step [{inner_epoch+1}|{N_EPOCH_INNER}] - Train Loss: {train_loss:7.3f} - Train Acc: {train_acc:7.3f} - Valid Acc: {valid_acc:7.3f}")
                model_cls.train() # switch back to train mode
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

            onehot_labels = F.one_hot(labels, num_classes=10)
            onehot_predict = F.one_hot(predict, num_classes=10)
            test_conf_delta[indices] = torch.abs(onehot_labels - onehot_predict).detach().cpu().float()
        cprint(f"Classification Test Acc: {test_correct/test_total:7.3f}", "cyan")

        cprint(f">>>Epoch [{outer_epoch+1}|{N_EPOCH_OUTER}] Train Confidence Model <<<", "green")
        trainset.set_delta_eta(train_conf_delta)
        validset.set_delta_eta(valid_conf_delta)
        testset.set_delta_eta(test_conf_delta)

        model_conf.train()
        for inner_epoch in range(N_EPOCH_INNER):
            total_loss = 0
            for _, (indices, images, _, delta_eta) in enumerate(tqdm(train_loader, ascii=True, ncols=100)):
                if images.shape[0]==1:
                    continue
                optimizer_conf.zero_grad()
                images, delta_eta = images.to(device), delta_eta.to(device)
                outs = model_conf(images)
                loss = criterion_conf(outs, delta_eta)
                loss.backward()
                optimizer_conf.step()

                total_loss+=len(delta_eta)*loss.item()**2
            total_loss=(total_loss/len(trainset))**(1/2)

            if not (inner_epoch+1)%MONITOR_WINDOW:
                valid_loss = 0
                valid_loss_delta_eta = 0
                model_conf.eval()
                for _, (indices, images, _, delta_eta) in enumerate(tqdm(valid_loader, ascii=True, ncols=100)):
                    if images.shape[0]==1:
                        continue
                    images, delta_eta = images.to(device), delta_eta.to(device)
                    outs = model_conf(images)
                    _, predict = outs.max(1)
                    loss = criterion_conf(outs, delta_eta)

                    valid_loss+=len(delta_eta)*(loss.item())**2
                    outs = outs.detach().cpu()
                    one_hot_h_star_valid = F.one_hot(h_star_valid[indices], num_classes=10)
                    delta_eta_valid = torch.abs(one_hot_h_star_valid - torch.softmax(outs, 1))
                    valid_loss_delta_eta+=len(delta_eta)*criterion_conf(outs, delta_eta_valid)**2

                valid_mse = (valid_loss/len(validset))**(1/2)
                valid_mse_delta_eta = (valid_loss_delta_eta/len(validset))**(1/2)
                print(f"Step [{inner_epoch+1}|{N_EPOCH_INNER}] - Train Loss: {train_loss:7.3f} - Valid Loss: {valid_mse:7.3f} - Valid Delta Eta Mse: {valid_mse_delta_eta:7.3f}")
                model_conf.train() # switch back to train mode
            scheduler_conf.step()

        test_loss = 0
        test_loss_delta_eta = 0
        model_conf.eval()
        for _, (indices, images, _, delta_eta) in enumerate(tqdm(test_loader, ascii=True, ncols=100)):
            if images.shape[0] == 1:
                continue
            images, delta_eta = images.to(device), delta_eta.to(device)
            outs = model_conf(images)
            _, predict = outs.max(1)

            loss = criterion_conf(outs, delta_eta)

            test_loss += len(labels)*(loss.item())**2
            outs = outs.detach().cpu()
            one_hot_h_star_test = F.one_hot(h_star_test[indices], num_classes=10)
            delta_eta_test = torch.abs(one_hot_h_star_test - torch.softmax(outs, 1))
            test_loss_delta_eta += len(labels)*criterion_conf(outs, delta_eta_test)**2

        test_mse = (test_loss / len(testset))**(1/2)
        test_mse_delta_eta = (test_loss_delta_eta / len(testset))**(1/2)
        cprint(f"Regression Test Loss: {test_mse:7.3f} - Regression Test Delta Eta Mse: {test_mse_delta_eta:7.3f}", "cyan")

        # TODO: add labels correction here

    return 0

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Arguement for SLDenoise")
    parser.add_argument("--seed", type=int, help="Random seed for the experiment", default=80)
    parser.add_argument("--gpus", type=str, help="Indices of GPUs to be used", default='0')
    parser.add_argument("--noise_type",  type=str, help="Noise type", default='linear', choices={"linear", "random"})
    parser.add_argument("--noise_strength", type=float, help="Noise fraction", default=1)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    main(args)