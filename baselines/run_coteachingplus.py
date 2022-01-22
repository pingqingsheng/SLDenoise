import os
import sys
sys.path.append("../")
from typing import List
import json
import pdb

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from torch import nn, optim
from torchvision import transforms
import numpy as np
import argparse
import random
from tqdm import tqdm
import copy
from termcolor import cprint
import datetime

from data.MNIST import MNIST
from data.CIFAR import CIFAR10
from baseline_network import resnet18
from utils.utils import _init_fn, ECELoss
from utils.noise import perturb_eta, noisify_with_P, noisify_mnist_asymmetric, noisify_cifar10_asymmetric

# Experiment Setting Control Panel
# ---------------------------------------------------
TRAIN_VALIDATION_RATIO: float = 0.8
N_EPOCH_OUTER: int = 1
N_EPOCH_INNER_CLS: int = 200
CONF_RECORD_EPOCH: int = N_EPOCH_INNER_CLS - 1
LR: float = 1e-2
WEIGHT_DECAY: float = 5e-3
BATCH_SIZE: int = 128
SCHEDULER_DECAY_MILESTONE: List = [40, 80, 120]
MONITOR_WINDOW: int = 1
# ----------------------------------------------------


def main(args):

    seed = args.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # need to set to True as well

    global DEVICE
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    # Data Loading and Processing
    global NUM_CLASSES
    if args.dataset == 'mnist':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        trainset = MNIST(root="../data", split="train", train_ratio=TRAIN_VALIDATION_RATIO, download=True,transform=transform_train)
        validset = MNIST(root="../data", split="valid", train_ratio=TRAIN_VALIDATION_RATIO, transform=transform_train)
        testset = MNIST(root="../data", split="test", download=True, transform=transform_test)
        INPUT_CHANNEL = 1
        NUM_CLASSES = 10
        if args.noise_type == 'idl':
            model_cls_clean = torch.load(f"../data/MNIST_resnet18_clean_{int(args.noise_strength*100)}.pth")
        else:
            model_cls_clean = torch.load("../data/MNIST_resnet18_clean.pth")
    elif args.dataset == 'cifar10':
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
        trainset = CIFAR10(root="../data", split="train", train_ratio=TRAIN_VALIDATION_RATIO, download=True, transform=transform_train)
        validset = CIFAR10(root="../data", split="valid", train_ratio=TRAIN_VALIDATION_RATIO, download=True, transform=transform_train)
        testset  = CIFAR10(root='../data', split="test", download=True, transform=transform_test)
        INPUT_CHANNEL = 3
        NUM_CLASSES = 10
        if args.noise_type == 'idl':
            model_cls_clean = torch.load(f"../data/CIFAR10_resnet18_clean_{int(args.noise_strength * 100)}.pth")
        else:
            model_cls_clean = torch.load("../data/CIFAR10_resnet18_clean.pth")

    validset_noise = copy.deepcopy(validset)
    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, worker_init_fn=_init_fn(worker_id=seed))
    valid_loader = DataLoader(validset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, worker_init_fn=_init_fn(worker_id=seed))
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, worker_init_fn=_init_fn(worker_id=seed))
    valid_loader_noise = DataLoader(validset_noise, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, worker_init_fn=_init_fn(worker_id=seed))

    # Inject noise here
    y_train = copy.deepcopy(trainset.targets)
    y_valid = copy.deepcopy(validset.targets)
    y_test = testset.targets

    cprint(">>> Inject Noise <<<", "green")
    _eta_train_temp_pair = [(torch.softmax(model_cls_clean(images).to(DEVICE), 1).detach().cpu(), indices) for
                            _, (indices, images, labels, _) in enumerate(tqdm(train_loader, ascii=True, ncols=100))]
    _eta_valid_temp_pair = [(torch.softmax(model_cls_clean(images).to(DEVICE), 1).detach().cpu(), indices) for
                            _, (indices, images, labels, _) in enumerate(tqdm(valid_loader, ascii=True, ncols=100))]
    _eta_test_temp_pair = [(torch.softmax(model_cls_clean(images).to(DEVICE), 1).detach().cpu(), indices) for
                           _, (indices, images, labels, _) in enumerate(tqdm(test_loader, ascii=True, ncols=100))]
    _eta_train_temp = torch.cat([x[0] for x in _eta_train_temp_pair])
    _eta_valid_temp = torch.cat([x[0] for x in _eta_valid_temp_pair])
    _eta_test_temp = torch.cat([x[0] for x in _eta_test_temp_pair])
    _train_indices = torch.cat([x[1] for x in _eta_train_temp_pair]).squeeze()
    _valid_indices = torch.cat([x[1] for x in _eta_valid_temp_pair]).squeeze()
    _test_indices = torch.cat([x[1] for x in _eta_test_temp_pair]).squeeze()
    eta_train = _eta_train_temp[_train_indices.argsort()]
    eta_valid = _eta_valid_temp[_valid_indices.argsort()]
    eta_test = _eta_test_temp[_test_indices.argsort()]
    h_star_train = eta_train.argmax(1).squeeze()
    h_star_valid = eta_valid.argmax(1).squeeze()
    h_star_test = eta_test.argmax(1).squeeze()

    if args.noise_type=='linear':
        eta_tilde_train = perturb_eta(eta_train, args.noise_type, args.noise_strength)
        eta_tilde_valid = perturb_eta(eta_valid, args.noise_type, args.noise_strength)
        eta_tilde_test  = perturb_eta(eta_test, args.noise_type, args.noise_strength)
        y_tilde_train = [int(np.where(np.random.multinomial(1, x, 1).squeeze())[0]) for x in eta_tilde_train]
        y_tilde_valid = [int(np.where(np.random.multinomial(1, x, 1).squeeze())[0]) for x in eta_tilde_valid]
        trainset.update_labels(y_tilde_train)
    elif args.noise_type=='uniform':
        y_tilde_train, P, _ = noisify_with_P(np.array(copy.deepcopy(h_star_train)), nb_classes=10, noise=args.noise_strength)
        y_tilde_valid, P, _ = noisify_with_P(np.array(copy.deepcopy(h_star_valid)), nb_classes=10, noise=args.noise_strength)
        eta_tilde_train = np.matmul(F.one_hot(h_star_train, num_classes=10), P)
        eta_tilde_valid = np.matmul(F.one_hot(h_star_valid, num_classes=10), P)
        eta_tilde_test  = np.matmul(F.one_hot(h_star_test, num_classes=10), P)
        trainset.update_labels(y_tilde_train)
    elif args.noise_type=='asymmetric':
        if args.dataset == "mnist":
            y_tilde_train, P, _ = noisify_mnist_asymmetric(np.array(copy.deepcopy(h_star_train)), noise=args.noise_strength)
            y_tilde_valid, P, _ = noisify_mnist_asymmetric(np.array(copy.deepcopy(h_star_valid)), noise=args.noise_strength)
        elif args.dataset == 'cifar10':
            y_tilde_train, P, _ = noisify_cifar10_asymmetric(np.array(copy.deepcopy(h_star_train)),noise=args.noise_strength)
            y_tilde_valid, P, _ = noisify_mnist_asymmetric(np.array(copy.deepcopy(h_star_valid)), noise=args.noise_strength)
        eta_tilde_train = np.matmul(F.one_hot(h_star_train, num_classes=10), P)
        eta_tilde_valid = np.matmul(F.one_hot(h_star_valid, num_classes=10), P)
        eta_tilde_test  = np.matmul(F.one_hot(h_star_test, num_classes=10), P)
        trainset.update_labels(y_tilde_train)
    elif args.noise_type=="idl":
        eta_tilde_train = copy.deepcopy(eta_train)
        eta_tilde_valid = copy.deepcopy(eta_valid)
        eta_tilde_test  = copy.deepcopy(eta_test)
        y_tilde_train = [int(np.where(np.random.multinomial(1, x, 1).squeeze())[0]) for x in eta_tilde_train]
        y_tilde_valid = [int(np.where(np.random.multinomial(1, x, 1).squeeze())[0]) for x in eta_tilde_valid]
        trainset.update_labels(y_tilde_train)

    validset_noise = copy.deepcopy(validset)
    validset.update_labels(h_star_valid)
    testset.update_labels(h_star_test)
    validset_noise.update_labels(y_tilde_valid)
    train_noise_ind = np.where(np.array(y_train) != np.array(y_tilde_train))[0]

    print("---------------------------------------------------------")
    print("                   Experiment Setting                    ")
    print("---------------------------------------------------------")
    print(f"Network: \t\t\t ResNet18")
    print(f"Number of Outer Epochs: \t {N_EPOCH_OUTER}")
    print(f"Number of cls Inner Epochs: \t {N_EPOCH_INNER_CLS}")
    print(f"Learning Rate: \t\t\t {LR}")
    print(f"Weight Decay: \t\t\t {WEIGHT_DECAY}")
    print(f"Batch Size: \t\t\t {BATCH_SIZE}")
    print(f"Scheduler Milestones: \t\t {SCHEDULER_DECAY_MILESTONE}")
    print(f"Monitor Window: \t\t {MONITOR_WINDOW}")
    print("---------------------------------------------------------")
    print(f"Number of Training Data Points: \t\t {len(trainset)}")
    print(f"Number of Validation Data Points: \t\t {len(validset)}")
    print(f"Number of Testing Data Points: \t\t\t {len(testset)}")
    print(f"Cls Confidence Recording Epoch: \t\t {CONF_RECORD_EPOCH}")
    print(f"Train - Validation Ratio: \t\t\t {TRAIN_VALIDATION_RATIO}")
    print(f"Clean Training Data Points: \t\t\t {len(trainset) - len(train_noise_ind)}")
    print(f"Noisy Training Data Points: \t\t\t {len(train_noise_ind)}")
    print(f"Noisy Type: \t\t\t\t\t {args.noise_type}")
    print(f"Noisy Strength: \t\t\t\t {args.noise_strength}")
    print(f"Noisy Level: \t\t\t\t\t {len(train_noise_ind) / len(trainset) * 100:.2f}%")
    print("---------------------------------------------------------")

    model_cls_1 = resnet18(num_classes=NUM_CLASSES+1, in_channels=INPUT_CHANNEL)
    model_cls_1 = DataParallel(model_cls_1)
    model_cls_2 = resnet18(num_classes=NUM_CLASSES+1, in_channels=INPUT_CHANNEL)
    model_cls_2 = DataParallel(model_cls_2)
    model_cls_1, model_cls_2 = model_cls_1.to(DEVICE), model_cls_2.to(DEVICE)

    optimizer_1 = torch.optim.SGD(model_cls_1.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, momentum=0.9, nesterov=True)
    scheduler_1 = torch.optim.lr_scheduler.MultiStepLR(optimizer_1, gamma=0.5, milestones=SCHEDULER_DECAY_MILESTONE)
    optimizer_2 = torch.optim.SGD(model_cls_2.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, momentum=0.9, nesterov=True)
    scheduler_2 = torch.optim.lr_scheduler.MultiStepLR(optimizer_2, gamma=0.5, milestones=SCHEDULER_DECAY_MILESTONE)

    criterion_calibrate = ECELoss()
    criterion_l1 = torch.nn.L1Loss()

    # For monitoring purpose
    _loss_record = np.zeros(len(np.linspace(0, N_EPOCH_INNER_CLS - N_EPOCH_INNER_CLS % MONITOR_WINDOW, int(
        (N_EPOCH_INNER_CLS - N_EPOCH_INNER_CLS % MONITOR_WINDOW) / MONITOR_WINDOW + 1))))
    _valid_acc_raw_record = np.zeros(len(np.linspace(0, N_EPOCH_INNER_CLS - N_EPOCH_INNER_CLS % MONITOR_WINDOW, int(
        (N_EPOCH_INNER_CLS - N_EPOCH_INNER_CLS % MONITOR_WINDOW) / MONITOR_WINDOW + 1))))
    _valid_acc_cali_record = np.zeros(len(np.linspace(0, N_EPOCH_INNER_CLS - N_EPOCH_INNER_CLS % MONITOR_WINDOW, int(
        (N_EPOCH_INNER_CLS - N_EPOCH_INNER_CLS % MONITOR_WINDOW) / MONITOR_WINDOW + 1))))
    _ece_raw_record = np.zeros(len(np.linspace(0, N_EPOCH_INNER_CLS - N_EPOCH_INNER_CLS % MONITOR_WINDOW, int(
        (N_EPOCH_INNER_CLS - N_EPOCH_INNER_CLS % MONITOR_WINDOW) / MONITOR_WINDOW + 1))))
    _ece_cali_record = np.zeros(len(np.linspace(0, N_EPOCH_INNER_CLS - N_EPOCH_INNER_CLS % MONITOR_WINDOW, int(
        (N_EPOCH_INNER_CLS - N_EPOCH_INNER_CLS % MONITOR_WINDOW) / MONITOR_WINDOW + 1))))
    _l1_raw_record = np.zeros(len(np.linspace(0, N_EPOCH_INNER_CLS - N_EPOCH_INNER_CLS % MONITOR_WINDOW, int(
        (N_EPOCH_INNER_CLS - N_EPOCH_INNER_CLS % MONITOR_WINDOW) / MONITOR_WINDOW + 1))))
    _l1_cali_record = np.zeros(len(np.linspace(0, N_EPOCH_INNER_CLS - N_EPOCH_INNER_CLS % MONITOR_WINDOW, int(
        (N_EPOCH_INNER_CLS - N_EPOCH_INNER_CLS % MONITOR_WINDOW) / MONITOR_WINDOW + 1))))
    _count = 0
    _best_raw_l1 = np.inf
    _best_raw_ece = np.inf
    _best_cali_l1 = np.inf
    _best_cali_ece = np.inf
    _best_raw_acc = 0
    _best_cali_acc = 0

    # give it additional information for now
    forget_rate = args.noise_strength
    rate_schedule = gen_forget_rate(forget_rate)

    for epoch in range(N_EPOCH_INNER_CLS):

        cprint(f">>>Epoch [{epoch + 1}|{N_EPOCH_INNER_CLS}] Training <<<", "green")

        train_correct_1 = 0
        train_loss_1 = 0
        train_correct_2 = 0
        train_loss_2 = 0
        train_total = 0
        forget_rate = rate_schedule[epoch]

        model_cls_1.train()
        model_cls_2.train()
        for ib, (indices, images, labels, _) in enumerate(tqdm(train_loader, ascii=True, ncols=100)):
            if images.shape[0] == 1:
                continue
            optimizer_1.zero_grad()
            optimizer_2.zero_grad()

            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outs_1 = model_cls_1(images)
            outs_2 = model_cls_2(images)

            if args.calibration_mode:
                if epoch < args.warm_up:
                    loss_1, loss_2 = loss_coteaching_calibrated(outs_1, outs_2, labels, forget_rate)
                else:
                    loss_1, loss_2 = loss_coteaching_plus_calibrated(outs_1, outs_2, labels, forget_rate, epoch*ib)
            else:
                if epoch < args.warm_up:
                    loss_1, loss_2 = loss_coteaching(outs_1, outs_2, labels, forget_rate)
                else:
                    loss_1, loss_2 = loss_coteaching_plus(outs_1, outs_2, labels, forget_rate, epoch*ib)

            loss_1.backward()
            optimizer_1.step()
            loss_2.backward()
            optimizer_2.step()

            train_loss_1 += loss_1.detach().cpu().item()
            train_loss_2 += loss_2.detach().cpu().item()
            _, predict_1 = outs_1.max(1)
            _, predict_2 = outs_2.max(1)
            train_correct_1 += predict_1.eq(labels).sum().item()
            train_correct_2 += predict_2.eq(labels).sum().item()
            train_total += len(labels)

        train_acc_1 = train_correct_1/train_total
        train_acc_2 = train_correct_2/train_total
        train_acc = max(train_acc_1, train_acc_2)
        train_loss = min(train_loss_1, train_loss_2)
        scheduler_1.step()
        scheduler_2.step()

        if not (epoch % MONITOR_WINDOW):

            valid_correct_raw_1 = 0
            valid_correct_raw_2 = 0
            valid_correct_cali_1 = 0
            valid_correct_cali_2 = 0
            valid_total = 0

            # For ECE
            valid_f_raw_1 = torch.zeros(len(validset), NUM_CLASSES).float()
            valid_f_raw_2 = torch.zeros(len(validset), NUM_CLASSES).float()
            valid_f_cali_1 = torch.zeros(len(validset), NUM_CLASSES).float()
            valid_f_cali_2 = torch.zeros(len(validset), NUM_CLASSES).float()
            # For L1
            valid_f_raw_1_target_conf = torch.zeros(len(validset)).float()
            valid_f_raw_2_target_conf = torch.zeros(len(validset)).float()
            valid_f_cali_1_target_conf = torch.zeros(len(validset)).float()
            valid_f_cali_2_target_conf = torch.zeros(len(validset)).float()

            model_cls_1.eval()
            model_cls_2.eval()
            for _, (indices, images, labels, _) in enumerate(tqdm(valid_loader, ascii=True, ncols=100)):
                if images.shape[0] == 1:
                    continue
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outs_1 = model_cls_1(images)
                outs_2 = model_cls_2(images)

                # Raw model result record
                prob_outs_1 = torch.softmax(outs_1[:, :NUM_CLASSES], 1)
                prob_outs_2 = torch.softmax(outs_2[:, :NUM_CLASSES], 1)
                _, predict_1 = prob_outs_1.max(1)
                _, predict_2 = prob_outs_2.max(1)
                correct_prediction_1 = predict_1.eq(labels).float()
                correct_prediction_2 = predict_2.eq(labels).float()

                valid_correct_raw_1 += correct_prediction_1.sum().item()
                valid_correct_raw_2 += correct_prediction_2.sum().item()
                valid_total += len(labels)
                valid_f_raw_1[indices] = prob_outs_1.detach().cpu()
                valid_f_raw_2[indices] = prob_outs_2.detach().cpu()
                valid_f_raw_1_target_conf[indices] = prob_outs_1.max(1)[0].detach().cpu()
                valid_f_raw_2_target_conf[indices] = prob_outs_2.max(1)[0].detach().cpu()

                if args.calibration_mode:
                    # Calibrated model result record
                    _valid_f_cali_1 = torch.sigmoid(outs_1[:, NUM_CLASSES:])
                    _prob_outs = torch.softmax(outs_1[:, :NUM_CLASSES], 1)
                    _, predict = _prob_outs.max(1)
                    _prob_outs = (1 - _valid_f_cali_1)*_prob_outs/(_prob_outs.sum(1) - _prob_outs.max(1)[0]).unsqueeze(1)
                    _prob_outs = _prob_outs.scatter_(1, predict[:, None], _valid_f_cali_1)
                    _, pred = _prob_outs.max(1)
                    valid_correct_cali_1 += pred.eq(labels).sum().item()
                    valid_f_cali_1[indices] = _prob_outs.detach().cpu()
                    valid_f_cali_1_target_conf[indices] = _valid_f_cali_1.detach().cpu().squeeze()

                    _valid_f_cali_2 = torch.sigmoid(outs_2[:, NUM_CLASSES:])
                    _prob_outs = torch.softmax(outs_2[:, :NUM_CLASSES], 1)
                    _, predict = _prob_outs.max(1)
                    _prob_outs = (1 - _valid_f_cali_2) * _prob_outs/(_prob_outs.sum(1) - _prob_outs.max(1)[0]).unsqueeze(1)
                    _prob_outs = _prob_outs.scatter_(1, predict[:, None], _valid_f_cali_2)
                    _loss_2_cali = -(torch.nn.functional.one_hot(labels, NUM_CLASSES) * torch.log(_prob_outs)).mean()
                    _, pred = _prob_outs.max(1)
                    valid_correct_cali_2 += pred.eq(labels).sum().item()
                    valid_f_cali_2[indices] = _prob_outs.detach().cpu()
                    valid_f_cali_2_target_conf[indices] = _valid_f_cali_2.detach().cpu().squeeze()

            valid_acc_raw_1 = valid_correct_raw_1/valid_total
            valid_acc_raw_2 = valid_correct_raw_2/valid_total
            valid_acc_cali_1 = valid_correct_cali_1/valid_total
            valid_acc_cali_2 = valid_correct_cali_2/valid_total
            ece_loss_raw_1 = criterion_calibrate.forward(logits=valid_f_raw_1, labels=torch.tensor(eta_tilde_valid).argmax(1).squeeze())
            ece_loss_raw_2 = criterion_calibrate.forward(logits=valid_f_raw_2, labels=torch.tensor(eta_tilde_valid).argmax(1).squeeze())
            ece_loss_cali_1 = criterion_calibrate.forward(logits=valid_f_cali_1, labels=torch.tensor(eta_tilde_valid).argmax(1).squeeze())
            ece_loss_cali_2 = criterion_calibrate.forward(logits=valid_f_cali_2, labels=torch.tensor(eta_tilde_valid).argmax(1).squeeze())
            l1_loss_raw_1 = criterion_l1(valid_f_raw_1_target_conf, torch.tensor(eta_tilde_valid).max(1)[0])
            l1_loss_raw_2 = criterion_l1(valid_f_raw_2_target_conf, torch.tensor(eta_tilde_valid).max(1)[0])
            l1_loss_cali_1 = criterion_l1(valid_f_cali_1_target_conf, torch.tensor(eta_tilde_valid).max(1)[0])
            l1_loss_cali_2 = criterion_l1(valid_f_cali_2_target_conf, torch.tensor(eta_tilde_valid).max(1)[0])

            # print(f"Step [{epoch + 1}|{N_EPOCH_INNER_CLS}] - Train Loss: {train_loss/train_total:7.3f} - Train Acc: {train_acc:7.3f} - Valid Acc Raw: {valid_acc_raw:7.3f} - Valid Acc Cali: {valid_acc_cali:7.3f}")
            # print(f"ECE Raw: {ece_loss_raw.item()} - ECE Cali: {ece_loss_cali.item()} - L1 Loss: {l1_loss_raw.item()} - L1 Loss: {l1_loss_cali.item()}")

            # For monitoring purpose
            _loss_record[_count] = float(min(train_loss_1, train_loss_2)/train_total)
            _valid_acc_raw_record[_count] = float(max(valid_acc_raw_1, valid_acc_raw_2))
            _valid_acc_cali_record[_count] = float(max(valid_acc_cali_1, valid_acc_cali_2))
            _ece_raw_record[_count] = float(min(ece_loss_raw_1.item(), ece_loss_raw_2.item()))
            _ece_cali_record[_count] = float(min(ece_loss_cali_1.item(), ece_loss_cali_2.item()))
            _l1_raw_record[_count] = float(min(l1_loss_raw_1.item(), l1_loss_raw_2.item()))
            _l1_cali_record[_count] = float(min(l1_loss_cali_1.item(), l1_loss_cali_2.item()))
            _count += 1

            # Testing stage
            test_correct_raw_1 = 0
            test_correct_raw_2 = 0
            test_correct_cali_1 = 0
            test_correct_cali_2 = 0
            test_total = 0

            # For ECE
            test_f_raw_1 = torch.zeros(len(testset), NUM_CLASSES).float()
            test_f_raw_2 = torch.zeros(len(testset), NUM_CLASSES).float()
            test_f_cali_1 = torch.zeros(len(testset), NUM_CLASSES).float()
            test_f_cali_2 = torch.zeros(len(testset), NUM_CLASSES).float()
            # For L1
            test_f_raw_1_target_conf = torch.zeros(len(testset)).float()
            test_f_raw_2_target_conf = torch.zeros(len(testset)).float()
            test_f_cali_1_target_conf = torch.zeros(len(testset)).float()
            test_f_cali_2_target_conf = torch.zeros(len(testset)).float()

            model_cls_1.eval()
            model_cls_2.eval()
            for _, (indices, images, labels, _) in enumerate(tqdm(test_loader, ascii=True, ncols=100)):
                if images.shape[0] == 1:
                    continue
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outs_1 = model_cls_1(images)
                outs_2 = model_cls_2(images)

                # Raw model result record
                prob_outs_1 = torch.softmax(outs_1[:, :NUM_CLASSES], 1)
                prob_outs_2 = torch.softmax(outs_2[:, :NUM_CLASSES], 1)
                _, predict_1 = prob_outs_1.max(1)
                _, predict_2 = prob_outs_2.max(1)
                correct_prediction_1 = predict_1.eq(labels).float()
                correct_prediction_2 = predict_2.eq(labels).float()

                test_correct_raw_1 += correct_prediction_1.sum().item()
                test_correct_raw_2 += correct_prediction_2.sum().item()
                test_total += len(labels)
                test_f_raw_1[indices] = prob_outs_1.detach().cpu()
                test_f_raw_2[indices] = prob_outs_2.detach().cpu()
                test_f_raw_1_target_conf[indices] = prob_outs_1.max(1)[0].detach().cpu()
                test_f_raw_2_target_conf[indices] = prob_outs_2.max(1)[0].detach().cpu()

                if args.calibration_mode:
                    # Calibrated model result record
                    _test_f_cali_1 = torch.sigmoid(outs_1[:, NUM_CLASSES:])
                    _prob_outs = torch.softmax(outs_1[:, :NUM_CLASSES], 1)
                    _, predict = _prob_outs.max(1)
                    _prob_outs = (1 - _test_f_cali_1)*_prob_outs/(_prob_outs.sum(1) - _prob_outs.max(1)[0]).unsqueeze(1)
                    _prob_outs = _prob_outs.scatter_(1, predict[:, None], _test_f_cali_1)
                    _, pred = _prob_outs.max(1)
                    test_correct_cali_1 += pred.eq(labels).sum().item()
                    test_f_cali_1[indices] = _prob_outs.detach().cpu()
                    test_f_cali_1_target_conf[indices] = _test_f_cali_1.detach().cpu().squeeze()

                    _test_f_cali_2 = torch.sigmoid(outs_2[:, NUM_CLASSES:])
                    _prob_outs = torch.softmax(outs_2[:, :NUM_CLASSES], 1)
                    _, predict = _prob_outs.max(1)
                    _prob_outs = (1 - _test_f_cali_2)*_prob_outs/(_prob_outs.sum(1) - _prob_outs.max(1)[0]).unsqueeze(1)
                    _prob_outs = _prob_outs.scatter_(1, predict[:, None], _test_f_cali_2)
                    _, pred = _prob_outs.max(1)
                    test_correct_cali_2 += pred.eq(labels).sum().item()
                    test_f_cali_2[indices] = _prob_outs.detach().cpu()
                    test_f_cali_2_target_conf[indices] = _test_f_cali_2.detach().cpu().squeeze()

            test_acc_raw_1 = test_correct_raw_1/test_total
            test_acc_raw_2 = test_correct_raw_2/test_total
            test_acc_cali_1 = test_correct_cali_1/test_total
            test_acc_cali_2 = test_correct_cali_2/test_total
            ece_loss_raw_1 = criterion_calibrate.forward(logits=test_f_raw_1, labels=torch.tensor(eta_tilde_test).argmax(1).squeeze())
            ece_loss_raw_2 = criterion_calibrate.forward(logits=test_f_raw_2, labels=torch.tensor(eta_tilde_test).argmax(1).squeeze())
            ece_loss_cali_1 = criterion_calibrate.forward(logits=test_f_cali_1, labels=torch.tensor(eta_tilde_test).argmax(1).squeeze())
            ece_loss_cali_2 = criterion_calibrate.forward(logits=test_f_cali_2, labels=torch.tensor(eta_tilde_test).argmax(1).squeeze())
            l1_loss_raw_1 = criterion_l1(test_f_raw_1_target_conf, torch.tensor(eta_tilde_test).max(1)[0])
            l1_loss_raw_2 = criterion_l1(test_f_raw_2_target_conf, torch.tensor(eta_tilde_test).max(1)[0])
            l1_loss_cali_1 = criterion_l1(test_f_cali_1_target_conf, torch.tensor(eta_tilde_test).max(1)[0])
            l1_loss_cali_2 = criterion_l1(test_f_cali_2_target_conf, torch.tensor(eta_tilde_test).max(1)[0])

            test_acc_raw = max(test_acc_raw_1, test_acc_raw_2)
            test_acc_cali = max(test_acc_cali_1, test_acc_cali_2)
            ece_loss_raw = min(ece_loss_raw_1.item(), ece_loss_raw_2.item())
            l1_loss_raw  = min(l1_loss_raw_1.item(), l1_loss_raw_2.item())
            ece_loss_cali = min(ece_loss_cali_1.item(), ece_loss_cali_2.item())
            l1_loss_cali = min(l1_loss_cali_1.item(), l1_loss_cali_2.item())
            if  ece_loss_raw < _best_raw_ece:
                _best_raw_ece = ece_loss_raw
                _best_raw_ece_epoch = epoch
            if l1_loss_raw < _best_raw_l1:
                _best_raw_l1 = l1_loss_raw
                _best_raw_l1_epoch = epoch
            if ece_loss_cali < _best_cali_ece:
                _best_cali_ece = ece_loss_cali
                _best_cali_ece_epoch = epoch
            if l1_loss_cali < _best_cali_l1:
                _best_cali_l1 = l1_loss_cali
                _best_cali_l1_epoch = epoch
            if test_acc_raw > _best_raw_acc:
                _best_raw_acc = test_acc_raw
                _best_raw_acc_epoch = epoch
            if test_acc_cali > _best_cali_acc:
                _best_cali_acc = test_acc_cali
                _best_cali_acc_epoch = epoch

            print(f"Step [{epoch + 1}|{N_EPOCH_INNER_CLS}] - Train Loss: {train_loss/train_total:7.3f} - Train Acc: {train_acc:7.3f} - Valid Acc Raw: {test_acc_raw:7.3f} - Valid Acc Cali: {test_acc_cali:7.3f}")
            print(f"ECE Raw: {ece_loss_raw:.3f}/{_best_raw_ece:.3f} - ECE Cali: {ece_loss_cali:.3f}/{_best_cali_ece:.3f} - Best ECE Raw Epoch: {_best_raw_ece_epoch} - Best ECE Cali Epoch: {_best_cali_ece_epoch}")
            print(f"L1 Raw: {l1_loss_raw:.3f}/{_best_raw_l1:.3f} - L1 Cali: {l1_loss_cali:.3f}/{_best_cali_l1:.3f} - Best L1 Raw: {_best_raw_l1_epoch} - Best L1 Cali: {_best_cali_l1_epoch}")

    if args.figure:
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.use('Agg')
        if not os.path.exists('./figures'):
            os.makedirs('./figures', exist_ok=True)
        fig = plt.figure(figsize=(10, 10))
        plt.subplot(2, 2, 1)
        x_axis = np.linspace(0, N_EPOCH_INNER_CLS - N_EPOCH_INNER_CLS % MONITOR_WINDOW,
                             int((N_EPOCH_INNER_CLS - N_EPOCH_INNER_CLS % MONITOR_WINDOW) / MONITOR_WINDOW + 1))
        plt.plot(x_axis[:-1], _loss_record[:-1], linewidth=2)
        plt.title("Loss Curve")
        plt.subplot(2, 2, 2)
        plt.plot(x_axis[:-1], _valid_acc_raw_record[:-1], linewidth=2, label="Raw")
        plt.plot(x_axis[:-1], _valid_acc_cali_record[:-1], linewidth=2, label="CoteachingPlus")
        plt.title('Acc Curve')
        plt.subplot(2, 2, 3)
        plt.plot(x_axis[:-1], _ece_raw_record[:-1], linewidth=2, label="Raw")
        plt.plot(x_axis[:-1], _ece_cali_record[:-1], linewidth=2, label="CoteachingPlus")
        plt.legend()
        plt.title('ECE Curve')
        plt.subplot(2, 2, 4)
        plt.plot(x_axis[:-1], _l1_raw_record[:-1], linewidth=2, label="Raw")
        plt.plot(x_axis[:-1], _l1_cali_record[:-1], linewidth=2, label="CoteachingPlus")
        plt.legend()
        plt.title('L1 Curve')

        time_stamp = datetime.datetime.strftime(datetime.datetime.today(), "%Y-%m-%d")
        fig.savefig(os.path.join("./figures", f"exp_log_{args.dataset}_{args.noise_type}_{args.noise_strength}_CoteachingPlus_plot_{time_stamp}.png"))

    if args.calibration_mode:
        return  _best_cali_l1, _best_cali_ece, _best_cali_acc
    else:
        return _best_raw_l1, _best_raw_ece, _best_raw_acc


# define drop rate schedule
def gen_forget_rate(forget_rate):
    rate_schedule = np.ones(N_EPOCH_INNER_CLS) * forget_rate
    # Use the default setting from the paper
    rate_schedule[:min(10, N_EPOCH_INNER_CLS)] = np.linspace(0, forget_rate, min(10, N_EPOCH_INNER_CLS))
    return rate_schedule

# Co-Teaching+ Loss functions
def loss_coteaching(outs_1, outs_2, labels, forget_rate):

    loss_1 = F.cross_entropy(outs_1[:, :NUM_CLASSES], labels, reduction='none').flatten()
    ind_1_sorted = loss_1.argsort()
    loss_2 = F.cross_entropy(outs_2[:, :NUM_CLASSES], labels, reduction='none').flatten()
    ind_2_sorted = loss_2.argsort()

    remember_rate = 1 - forget_rate
    with torch.no_grad():
        num_remember = int(remember_rate * len(loss_1))

    ind_1_update = ind_1_sorted[:num_remember]
    ind_2_update = ind_2_sorted[:num_remember]
    if len(ind_1_update) == 0:
        ind_1_update = ind_1_sorted
        ind_2_update = ind_2_sorted
        num_remember = ind_1_update.shape[0]

    loss_1_update = F.cross_entropy(outs_1[ind_2_update, :NUM_CLASSES], labels[ind_2_update])
    loss_2_update = F.cross_entropy(outs_2[ind_1_update, :NUM_CLASSES], labels[ind_1_update])

    return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember

def loss_coteaching_plus(outs_1, outs_2, labels, forget_rate, step):

    _, pred1 = torch.max(outs_1.data[:, :NUM_CLASSES], 1)
    _, pred2 = torch.max(outs_2.data[:, :NUM_CLASSES], 1)

    with torch.no_grad():
        logical_disagree_vec = (pred1!=pred2)
        disagree_id = torch.where(logical_disagree_vec)[0]
        update_step = torch.logical_or(logical_disagree_vec, (step < 5000)*torch.ones(len(logical_disagree_vec)).to(DEVICE).long())

    if len(disagree_id) > 0:
        update_labels = labels[disagree_id]
        update_outs1 = outs_1[disagree_id]
        update_outs2 = outs_2[disagree_id]
        loss_1, loss_2 = loss_coteaching(update_outs1, update_outs2, update_labels, forget_rate)
    else:
        update_labels = labels
        update_outs1 = outs_1
        update_outs2 = outs_2
        cross_entropy_1 = F.cross_entropy(update_outs1[:, :NUM_CLASSES], update_labels)
        cross_entropy_2 = F.cross_entropy(update_outs2[:, :NUM_CLASSES], update_labels)
        loss_1 = torch.sum(update_step * cross_entropy_1) / labels.size()[0]
        loss_2 = torch.sum(update_step * cross_entropy_2) / labels.size()[0]

    return loss_1, loss_2

def loss_coteaching_calibrated(outs_1, outs_2, labels, forget_rate):

    _train_f_cali_1 = torch.sigmoid(outs_1[:, NUM_CLASSES:])
    _prob_outs = torch.softmax(outs_1[:, :NUM_CLASSES], 1)
    _, predict = _prob_outs.max(1)
    _prob_outs = (1 - _train_f_cali_1) * _prob_outs / (_prob_outs.sum(1) - _prob_outs.max(1)[0]).unsqueeze(1)
    _prob_outs = _prob_outs.scatter_(1, predict[:, None], _train_f_cali_1)
    _loss_1_cali = -(torch.nn.functional.one_hot(labels, NUM_CLASSES)*torch.log(_prob_outs)).mean(1)
    ind_1_sorted = _loss_1_cali.argsort()

    _train_f_cali_2 = torch.sigmoid(outs_2[:, NUM_CLASSES:])
    _prob_outs = torch.softmax(outs_2[:, :NUM_CLASSES], 1)
    _, predict = _prob_outs.max(1)
    _prob_outs = (1 - _train_f_cali_2) * _prob_outs / (_prob_outs.sum(1) - _prob_outs.max(1)[0]).unsqueeze(1)
    _prob_outs = _prob_outs.scatter_(1, predict[:, None], _train_f_cali_2)
    _loss_2_cali = -(torch.nn.functional.one_hot(labels, NUM_CLASSES)*torch.log(_prob_outs)).mean(1)
    ind_2_sorted = _loss_2_cali.argsort()

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(labels))

    ind_1_update = ind_1_sorted[:num_remember]
    ind_2_update = ind_2_sorted[:num_remember]
    if len(ind_1_update) == 0:
        ind_1_update = ind_1_sorted
        ind_2_update = ind_2_sorted
        num_remember = ind_1_update.shape[0]

    loss_1_update = F.cross_entropy(outs_1[ind_2_update, :NUM_CLASSES], labels[ind_2_update])
    loss_2_update = F.cross_entropy(outs_2[ind_1_update, :NUM_CLASSES], labels[ind_1_update])

    return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember

def loss_coteaching_plus_calibrated(outs_1, outs_2, labels, forget_rate, step):

    _, pred1 = torch.max(outs_1.data[:, :NUM_CLASSES], 1)
    _, pred2 = torch.max(outs_2.data[:, :NUM_CLASSES], 1)

    with torch.no_grad():
        logical_disagree_vec = (pred1!=pred2)
        disagree_id = torch.where(logical_disagree_vec)[0]
        update_step = torch.logical_or(logical_disagree_vec, (step < 5000)*torch.ones(len(logical_disagree_vec)).to(DEVICE).long())

    if len(disagree_id) > 0:
        update_labels = labels[disagree_id]
        update_outs1 = outs_1[disagree_id]
        update_outs2 = outs_2[disagree_id]
        loss_1, loss_2 = loss_coteaching_calibrated(update_outs1, update_outs2, update_labels, forget_rate)
    else:
        update_labels = labels
        update_outs1 = outs_1
        update_outs2 = outs_2
        cross_entropy_1 = F.cross_entropy(update_outs1[:, :NUM_CLASSES], update_labels)
        cross_entropy_2 = F.cross_entropy(update_outs2[:, :NUM_CLASSES], update_labels)
        loss_1 = torch.sum(update_step * cross_entropy_1) / labels.size()[0]
        loss_2 = torch.sum(update_step * cross_entropy_2) / labels.size()[0]

    return loss_1, loss_2

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguement for SLDenoise")
    parser.add_argument("--seed", type=int, help="Random seed for the experiment", default=80)
    parser.add_argument("--gpus", type=str, help="Indices of GPUs to be used", default='0')
    parser.add_argument("--dataset", type=str, help="Experiment Dataset", default='mnist',
                        choices={'mnist', 'cifar10', 'cifar100'})
    parser.add_argument("--noise_type", type=str, help="Noise type", default='linear',
                        choices={"linear", "uniform", "asymmetric", "idl"})
    parser.add_argument("--noise_strength", type=float, help="Noise fraction", default=1)
    parser.add_argument("--rollWindow", default=3, help="rolling window to calculate the confidence", type=int)
    parser.add_argument("--warm_up", default=2, help="warm-up period", type=int)
    parser.add_argument("--figure", action='store_true', help='True to plot performance log')
    parser.add_argument("--calibration_mode", action="store_true", help="True to use calibration")
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    exp_config = {}
    exp_config['n_epoch_outer'] = N_EPOCH_OUTER
    exp_config['n_epoch_cls'] = N_EPOCH_INNER_CLS
    exp_config['lr'] = LR
    exp_config['weight_decay'] = WEIGHT_DECAY
    exp_config['batch_size'] = BATCH_SIZE
    exp_config['mile_stone'] = SCHEDULER_DECAY_MILESTONE
    exp_config['train_vs_valid'] = TRAIN_VALIDATION_RATIO
    exp_config['monitor_window'] = MONITOR_WINDOW
    exp_config['record_freq'] = CONF_RECORD_EPOCH
    for k, v in args._get_kwargs():
        if k == 'seed':
            continue
        exp_config[k] = v

    exp_config['raw_l1'] = []
    exp_config['raw_ece'] = []
    exp_config['raw_acc'] = []
    exp_config['cali_l1'] = []
    exp_config['cali_ece'] = []
    exp_config['cali_acc'] = []
    exp_config['seed'] = []
    for seed in [77, 78, 79]:
        args.seed = seed
        args.calibration_mode = False
        raw_l1, raw_ece, raw_acc = main(args)
        exp_config['raw_l1'].append(raw_l1)
        exp_config['raw_ece'].append(raw_ece)
        exp_config['raw_acc'].append(raw_acc)
        args.calibration_mode = True
        cali_l1, cali_ece, cali_acc = main(args)
        exp_config['cali_l1'].append(cali_l1)
        exp_config['cali_ece'].append(cali_ece)
        exp_config['cali_acc'].append(cali_acc)
        exp_config['seed'].append(seed)
    print(f"Final Raw Acc  {raw_acc:.3f}\t L1 {raw_l1:.3f}\t ECE {raw_ece:.3f}")
    print(f"Final Cali Acc {cali_acc:.3f}\t L1 {cali_l1:.3f}\t ECE {cali_ece:.3f}")

    dir_date = datetime.datetime.today().strftime("%Y%m%d")
    save_folder = os.path.join("../exp_logs/coteachingplus_"+dir_date)
    os.makedirs(save_folder, exist_ok=True)
    print(save_folder)
    save_file_name = 'coteachingplus_' + datetime.date.today().strftime("%d_%m_%Y") + f"_{args.seed}_{args.dataset}_{args.noise_type}_{args.noise_strength}.json"
    save_file_name = os.path.join(save_folder, save_file_name)
    with open(save_file_name, "w") as f:
        json.dump(exp_config, f, sort_keys=False, indent=4)
    f.close()








