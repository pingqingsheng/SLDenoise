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

    model_cls = resnet18(num_classes=NUM_CLASSES+1, in_channels=INPUT_CHANNEL)
    model_cls = DataParallel(model_cls)
    model_cls = model_cls.to(DEVICE)

    optimizer_cls = torch.optim.SGD(model_cls.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, momentum=0.9, nesterov=True)
    scheduler_cls = torch.optim.lr_scheduler.MultiStepLR(optimizer_cls, gamma=0.5, milestones=SCHEDULER_DECAY_MILESTONE)

    criterion_conf = torch.nn.MSELoss()
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
    eta_corr = torch.zeros(len(trainset), NUM_CLASSES)
    T_hat = torch.eye(NUM_CLASSES).to(DEVICE)

    for epoch in range(N_EPOCH_INNER_CLS):

        cprint(f">>>Epoch [{epoch + 1}|{N_EPOCH_INNER_CLS}] Training <<<", "green")
        train_correct = 0
        train_total = 0
        train_loss = 0

        model_cls.train()
        for _, (indices, images, labels, _) in enumerate(tqdm(train_loader, ascii=True, ncols=100)):
            if images.shape[0] == 1:
                continue
            optimizer_cls.zero_grad()
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outs = model_cls(images)
            prob_outs = torch.softmax(outs[:, :NUM_CLASSES], 1)
            _, predict = prob_outs.max(1)

            delta_prediction = predict.eq(labels).float()
            _train_f_cali = torch.sigmoid(outs[:, NUM_CLASSES:]).squeeze()

            # Forward Correction
            labels_onehot = torch.nn.functional.one_hot(labels, NUM_CLASSES)
            loss_forward = -(labels_onehot*torch.log(torch.matmul(prob_outs, T_hat))).mean()
            loss_cali    = criterion_conf(_train_f_cali.squeeze(), delta_prediction.squeeze())
            loss = loss_forward + loss_cali
            loss.backward()
            optimizer_cls.step()

            train_loss += loss.detach().cpu().item()
            train_correct += predict.eq(labels).sum().item()
            train_total += len(labels)

            if epoch == args.warm_up-1 and args.calibration_mode:
                predict = predict
                _train_f_cali = torch.sigmoid(outs[:, NUM_CLASSES:])
                prob_outs = (1-_train_f_cali) * prob_outs/(prob_outs.sum(1) - prob_outs.max(1)[0]).unsqueeze(1)
                prob_outs = prob_outs.scatter_(1, predict[:, None], _train_f_cali)
                eta_corr[indices] = prob_outs.detach().cpu()
            elif epoch == args.warm_up-1 and not args.calibration_mode:
                eta_corr[indices] = prob_outs.detach().cpu()

        if epoch == args.warm_up:
            with torch.no_grad():
                T_hat = estimate_transition_mat(eta_corr).to(DEVICE)

        train_acc = train_correct / train_total
        scheduler_cls.step()

        if not (epoch % MONITOR_WINDOW):

            valid_correct_raw = 0
            valid_correct_cali = 0
            valid_total = 0
            # For ECE
            valid_f_raw = torch.zeros(len(validset), NUM_CLASSES).float()
            valid_f_cali = torch.zeros(len(validset), NUM_CLASSES).float()
            # For L1
            valid_f_raw_target_conf = torch.zeros(len(validset)).float()
            valid_f_cali_target_conf = torch.zeros(len(validset)).float()

            model_cls.eval()
            for _, (indices, images, labels, _) in enumerate(tqdm(valid_loader, ascii=True, ncols=100)):
                if images.shape[0] == 1:
                    continue
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outs_raw = model_cls(images)

                # Raw model result record
                prob_outs = torch.softmax(outs_raw[:, :NUM_CLASSES], 1)
                _, predict = prob_outs.max(1)
                correct_prediction = predict.eq(labels).float()
                valid_correct_raw += correct_prediction.sum().item()
                valid_total += len(labels)
                valid_f_raw[indices] = prob_outs.detach().cpu()
                valid_f_raw_target_conf[indices] = prob_outs.max(1)[0].detach().cpu()

                # Calibrated model result record
                if args.calibration_mode:
                    _valid_f_cali = torch.sigmoid(outs_raw[:, NUM_CLASSES:])
                    prob_outs = (1-_valid_f_cali) * prob_outs/(prob_outs.sum(1) - prob_outs.max(1)[0]).unsqueeze(1)
                    prob_outs = prob_outs.scatter_(1, predict[:, None], _valid_f_cali)
                    _, predict = prob_outs.max(1)
                    correct_prediction = predict.eq(labels).float()
                    valid_correct_cali += correct_prediction.sum().item()
                    valid_f_cali[indices] = prob_outs.detach().cpu()
                    valid_f_cali_target_conf[indices] = prob_outs.max(1)[0].detach().cpu()

            valid_acc_raw = valid_correct_raw/valid_total
            valid_acc_cali = valid_correct_cali/valid_total
            ece_loss_raw = criterion_calibrate.forward(logits=valid_f_raw, labels=torch.tensor(eta_tilde_valid).argmax(1).squeeze())
            ece_loss_cali = criterion_calibrate.forward(logits=valid_f_cali, labels=torch.tensor(eta_tilde_valid).argmax(1).squeeze())
            l1_loss_raw = criterion_l1(valid_f_raw_target_conf, torch.tensor(eta_tilde_valid).max(1)[0])
            l1_loss_cali = criterion_l1(valid_f_cali_target_conf, torch.tensor(eta_tilde_valid).max(1)[0])

            # print(f"Step [{epoch + 1}|{N_EPOCH_INNER_CLS}] - Train Loss: {train_loss/train_total:7.3f} - Train Acc: {train_acc:7.3f} - Valid Acc Raw: {valid_acc_raw:7.3f} - Valid Acc Cali: {valid_acc_cali:7.3f}")
            # print(f"ECE Raw: {ece_loss_raw.item()} - ECE Cali: {ece_loss_cali.item()} - L1 Loss: {l1_loss_raw.item()} - L1 Loss: {l1_loss_cali.item()}")

            # For monitoring purpose
            _loss_record[_count] = float(train_loss/train_total)
            _valid_acc_raw_record[_count] = float(valid_acc_raw)
            _valid_acc_cali_record[_count] = float(valid_acc_cali)
            _ece_raw_record[_count] = float(ece_loss_raw.item())
            _ece_cali_record[_count] = float(ece_loss_cali.item())
            _l1_raw_record[_count] = float(l1_loss_raw.item())
            _l1_cali_record[_count] = float(l1_loss_cali.item())
            _count += 1

            # Testing stage
            test_correct_raw = 0
            test_correct_cali = 0
            test_total = 0

            # For ECE
            test_f_raw = torch.zeros(len(testset), NUM_CLASSES).float()
            test_f_cali = torch.zeros(len(testset), NUM_CLASSES).float()
            # For L1
            test_f_raw_target_conf = torch.zeros(len(testset)).float()
            test_f_cali_target_conf = torch.zeros(len(testset)).float()

            model_cls.eval()
            for _, (indices, images, labels, _) in enumerate(tqdm(test_loader, ascii=True, ncols=100)):
                if images.shape[0] == 1:
                    continue
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outs_raw = model_cls(images)

                # Raw model result record
                prob_outs = torch.softmax(outs_raw[:, :NUM_CLASSES], 1)
                _, predict = prob_outs.max(1)
                correct_prediction = predict.eq(labels).float()
                test_correct_raw += correct_prediction.sum().item()
                test_total += len(labels)
                test_f_raw[indices] = prob_outs.detach().cpu()
                test_f_raw_target_conf[indices] = prob_outs.max(1)[0].detach().cpu()

                # Calibrated model result record
                if args.calibration_mode:
                    _test_f_cali = torch.sigmoid(outs_raw[:, NUM_CLASSES:])
                    prob_outs = (1-_test_f_cali)*prob_outs/(prob_outs.sum(1)-prob_outs.max(1)[0]).unsqueeze(1)
                    prob_outs = prob_outs.scatter_(1, predict[:, None], _test_f_cali)
                    _, predict = prob_outs.max(1)
                    correct_prediction = predict.eq(labels).float()
                    test_correct_cali += correct_prediction.sum().item()
                    test_f_cali[indices] = prob_outs.detach().cpu()
                    test_f_cali_target_conf[indices] = prob_outs.max(1)[0].detach().cpu()

            test_acc_raw = test_correct_raw/test_total
            test_acc_cali = test_correct_cali/test_total
            ece_loss_raw = criterion_calibrate.forward(logits=test_f_raw, labels=torch.tensor(eta_tilde_test).argmax(1).squeeze())
            ece_loss_cali = criterion_calibrate.forward(logits=test_f_cali, labels=torch.tensor(eta_tilde_test).argmax(1).squeeze())
            l1_loss_raw = criterion_l1(test_f_raw_target_conf, torch.tensor(eta_tilde_test).max(1)[0])
            l1_loss_cali = criterion_l1(test_f_cali_target_conf, torch.tensor(eta_tilde_test).max(1)[0])

            if ece_loss_raw < _best_raw_ece:
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
            print(f"ECE Raw: {ece_loss_raw.item():.3f}/{_best_raw_ece.item():.3f} - ECE Cali: {ece_loss_cali.item():.3f}/{_best_cali_ece.item():.3f} - Best ECE Raw Epoch: {_best_raw_ece_epoch} - Best ECE Cali Epoch: {_best_cali_ece_epoch}")
            print(f"L1 Raw: {l1_loss_raw.item():.3f}/{_best_raw_l1.item():.3f} - L1 Cali: {l1_loss_cali.item():.3f}/{_best_cali_l1.item():.3f} - Best L1 Raw: {_best_raw_l1_epoch} - Best L1 Cali: {_best_cali_l1_epoch}")

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
        plt.plot(x_axis[:-1], _valid_acc_cali_record[:-1], linewidth=2, label="Forward")
        plt.title('Acc Curve')
        plt.subplot(2, 2, 3)
        plt.plot(x_axis[:-1], _ece_raw_record[:-1], linewidth=2, label="Raw")
        plt.plot(x_axis[:-1], _ece_cali_record[:-1], linewidth=2, label="Forward")
        plt.legend()
        plt.title('ECE Curve')
        plt.subplot(2, 2, 4)
        plt.plot(x_axis[:-1], _l1_raw_record[:-1], linewidth=2, label="Raw")
        plt.plot(x_axis[:-1], _l1_cali_record[:-1], linewidth=2, label="Forward")
        plt.legend()
        plt.title('L1 Curve')

        time_stamp = datetime.datetime.strftime(datetime.datetime.today(), "%Y-%m-%d")
        fig.savefig(os.path.join("./figures", f"exp_log_{args.dataset}_{args.noise_type}_{args.noise_strength}_Forward_plot_{time_stamp}.png"))

    if args.calibration_mode:
        return  _best_cali_l1.item(), _best_cali_ece.item(), _best_cali_acc
    else:
        return _best_raw_l1.item(), _best_raw_ece.item(), _best_raw_acc

def estimate_transition_mat(eta_corr: torch.tensor, \
                            row_normalize = True,\
                            alpha = 0,\
                            filter_outlier = False,\
                            cliptozero = False):

    T_hat = torch.zeros(NUM_CLASSES, NUM_CLASSES)

    for i in range(NUM_CLASSES):
        if not filter_outlier:
            idx_best = eta_corr[:, i].squeeze().argmax()
        else:
            eta_thresh = torch.quantile(eta_corr[:, i].squeeze(), 97)
            robust_eta = eta_corr[:, i]
            robust_eta[robust_eta >= eta_thresh] = 0
            idx_best = robust_eta.squeeze().argmax()
        for j in range(NUM_CLASSES):
            T_hat[i, j] = eta_corr[idx_best, j]

    if cliptozero:
        idx = T_hat[T_hat < 1e-6]
        T_hat[idx] = 0.0
    if row_normalize:
        row_sums = T_hat.sum(axis=1)
        T_hat /= row_sums[:, None]
    if alpha > 0.0:
        T_hat = alpha*torch.eye(NUM_CLASSES) + (1-alpha)*T_hat

    return T_hat

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
    parser.add_argument("--calibration_mode", action='store_true', help='True to use calibration')
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
    for seed in [77]:
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
    print("\n")
    print(f"Final Raw Acc  {raw_acc:.3f}\t L1 {raw_l1:.3f}\t ECE {raw_ece:.3f}")
    print(f"Final Cali Acc {cali_acc:.3f}\t L1 {cali_l1:.3f}\t ECE {cali_ece:.3f}")

    dir_date = datetime.datetime.today().strftime("%Y%m%d")
    save_folder = os.path.join("../exp_logs/forward_"+dir_date)
    os.makedirs(save_folder, exist_ok=True)
    print(save_folder)
    save_file_name = 'forward_' + datetime.date.today().strftime("%d_%m_%Y") + f"_{args.seed}_{args.dataset}_{args.noise_type}_{args.noise_strength}.json"
    save_file_name = os.path.join(save_folder, save_file_name)
    with open(save_file_name, "w") as f:
        json.dump(exp_config, f, sort_keys=False, indent=4)
    f.close()








