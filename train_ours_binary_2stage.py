import os
import sys
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
from baselines.baseline_network import resnet18, resnet34
from utils.utils import _init_fn, ECELoss
from utils.noise import perturb_eta, noisify_with_P, noisify_mnist_asymmetric, noisify_cifar10_asymmetric

# Experiment Setting Control Panel
# ---------------------------------------------------
BETA: float = 3
LAMBDA: float = 0.8
N_SAMPLE: int = 20
# General setting
TRAIN_VALIDATION_RATIO: float = 0.8
N_EPOCH_CLS: int = 1
N_EPOCH_CALI: int = 195
CONF_RECORD_EPOCH: int = N_EPOCH_CLS - 1
LR: float = 1e-2
LR_CALI: float = 1e-4
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

    device = torch.device(f"cuda:0" if torch.cuda.is_available() else 'cpu')

    # Data Loading and Processing
    if args.dataset == 'mnist':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        trainset = MNIST(root="./data", split="train", train_ratio=TRAIN_VALIDATION_RATIO, download=True,transform=transform_train)
        validset = MNIST(root="./data", split="valid", train_ratio=TRAIN_VALIDATION_RATIO, transform=transform_train)
        testset = MNIST(root="./data", split="test", download=True, transform=transform_test)
        trainset_cali = copy.deepcopy(trainset)
        INPUT_CHANNEL = 1
        INPUT_SHAPE = 28
        NUM_CLASSES = 10
        if args.noise_type == 'idl':
            model_cls_clean_state_dict = torch.load(f"./data/MNIST_resnet18_clean_{int(args.noise_strength*100)}.pth")
        else:
            model_cls_clean_state_dict = torch.load("./data/MNIST_resnet18_clean.pth")
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
        trainset = CIFAR10(root="./data", split="train", train_ratio=TRAIN_VALIDATION_RATIO, download=True, transform=transform_train)
        validset = CIFAR10(root="./data", split="valid", train_ratio=TRAIN_VALIDATION_RATIO, download=True, transform=transform_test)
        testset  = CIFAR10(root='./data', split="test", download=True, transform=transform_test)
        trainset_cali = copy.deepcopy(trainset)
        INPUT_CHANNEL = 3
        INPUT_SHAPE = 32
        NUM_CLASSES = 10
        if args.noise_type == 'idl':
            model_cls_clean_state_dict = torch.load(f"./data/CIFAR10_resnet18_clean_{int(args.noise_strength * 100)}.pth")
        else:
            model_cls_clean_state_dict = torch.load("./data/CIFAR10_resnet18_clean.pth")

    validset_noise = copy.deepcopy(validset)
    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, worker_init_fn=_init_fn(worker_id=seed))
    valid_loader = DataLoader(validset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, worker_init_fn=_init_fn(worker_id=seed))
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, worker_init_fn=_init_fn(worker_id=seed))
    train_cali_loader = DataLoader(trainset_cali, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, worker_init_fn=_init_fn(worker_id=seed))
    valid_loader_noise = DataLoader(validset_noise, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, worker_init_fn=_init_fn(worker_id=seed))

    # Inject noise here
    y_train = copy.deepcopy(trainset.targets)
    y_valid = copy.deepcopy(validset.targets)
    y_test = testset.targets

    cprint(">>> Inject Noise <<<", "green")
    model_cls_clean = resnet18(num_classes=NUM_CLASSES, in_channels=INPUT_CHANNEL)
    gpu_id_list = [int(x) for x in args.gpus.split(",")]
    model_cls_clean = DataParallel(model_cls_clean, device_ids=[x-gpu_id_list[0] for x in gpu_id_list])
    model_cls_clean.load_state_dict(model_cls_clean_state_dict)
    _eta_train_temp_pair = [(torch.softmax(model_cls_clean(images), 1).detach().cpu(), indices) for
                            _, (indices, images, labels, _) in enumerate(tqdm(train_loader, ascii=True, ncols=100))]
    _eta_valid_temp_pair = [(torch.softmax(model_cls_clean(images), 1).detach().cpu(), indices) for
                            _, (indices, images, labels, _) in enumerate(tqdm(valid_loader, ascii=True, ncols=100))]
    _eta_test_temp_pair = [(torch.softmax(model_cls_clean(images), 1).detach().cpu(), indices) for
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
        y_tilde_test  = [int(np.where(np.random.multinomial(1, x, 1).squeeze())[0]) for x in eta_tilde_test]
        trainset.update_labels(y_tilde_train)
    elif args.noise_type=='uniform':
        y_tilde_train, P, _ = noisify_with_P(np.array(copy.deepcopy(h_star_train)), nb_classes=10, noise=args.noise_strength)
        y_tilde_valid, P, _ = noisify_with_P(np.array(copy.deepcopy(h_star_valid)), nb_classes=10, noise=args.noise_strength)
        y_tilde_test,  P, _ = noisify_with_P(np.array(copy.deepcopy(h_star_test)), nb_classes=10, noise=args.noise_strength)
        eta_tilde_train = np.matmul(F.one_hot(h_star_train, num_classes=10), P)
        eta_tilde_valid = np.matmul(F.one_hot(h_star_valid, num_classes=10), P)
        eta_tilde_test  = np.matmul(F.one_hot(h_star_test, num_classes=10), P)
        trainset.update_labels(y_tilde_train)
    elif args.noise_type=='asymmetric':
        if args.dataset == "mnist":
            y_tilde_train, P, _ = noisify_mnist_asymmetric(np.array(copy.deepcopy(h_star_train)), noise=args.noise_strength)
            y_tilde_valid, P, _ = noisify_mnist_asymmetric(np.array(copy.deepcopy(h_star_valid)), noise=args.noise_strength)
            y_tilde_test,  P, _ = noisify_mnist_asymmetric(np.array(copy.deepcopy(h_star_test)),  noise=args.noise_strength)
        elif args.dataset == 'cifar10':
            y_tilde_train, P, _ = noisify_cifar10_asymmetric(np.array(copy.deepcopy(h_star_train)), noise=args.noise_strength)
            y_tilde_valid, P, _ = noisify_cifar10_asymmetric(np.array(copy.deepcopy(h_star_valid)), noise=args.noise_strength)
            y_tilde_test,  P, _ = noisify_cifar10_asymmetric(np.array(copy.deepcopy(h_star_test)),  noise=args.noise_strength)
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
        y_tilde_test  = [int(np.where(np.random.multinomial(1, x, 1).squeeze())[0]) for x in eta_tilde_test]
        trainset.update_labels(y_tilde_train)

    validset_noise = copy.deepcopy(validset)
    validset.update_labels(h_star_valid)
    testset.update_labels(h_star_test)
    validset_noise.update_labels(y_tilde_valid)
    train_noise_ind = np.where(np.array(h_star_train) != np.array(y_tilde_train))[0]

    print("---------------------------------------------------------")
    print("                   Experiment Setting                    ")
    print("---------------------------------------------------------")
    print(f"Network: \t\t\t ResNet18")
    print(f"Number of CLS Epochs: \t\t {N_EPOCH_CLS}")
    print(f"Number of CALI Epochs: \t\t {N_EPOCH_CALI}")
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

    model_cls = resnet18(num_classes=NUM_CLASSES, in_channels=INPUT_CHANNEL)
    gpu_id_list = [int(x) for x in args.gpus.split(",")]
    model_cls = DataParallel(model_cls, device_ids=[x-gpu_id_list[0] for x in gpu_id_list])
    model_cls = model_cls.to(device)
    model_cali = resnet34(num_classes=1, in_channels=INPUT_CHANNEL)
    gpu_id_list = [int(x) for x in args.gpus.split(",")]
    model_cali = DataParallel(model_cali, device_ids=[x-gpu_id_list[0] for x in gpu_id_list])
    model_cali = model_cali.to(device)

    optimizer_cls = torch.optim.SGD(model_cls.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, momentum=0.9, nesterov=True)
    scheduler_cls = torch.optim.lr_scheduler.MultiStepLR(optimizer_cls, gamma=0.5, milestones=SCHEDULER_DECAY_MILESTONE)
    optimizer_cali = torch.optim.SGD(model_cali.parameters(), lr=LR_CALI, weight_decay=WEIGHT_DECAY, momentum=0.9, nesterov=True)
    scheduler_cali = torch.optim.lr_scheduler.MultiStepLR(optimizer_cali, gamma=0.5, milestones=SCHEDULER_DECAY_MILESTONE)
    
    criterion_cls = torch.nn.CrossEntropyLoss(reduction='none')
    criterion_conf = torch.nn.MSELoss()
    criterion_conf = torch.nn.L1Loss()
    criterion_calibrate = ECELoss()
    criterion_l1 = torch.nn.L1Loss()

    # For monitoring purpose
    _loss_record = np.zeros(len(np.linspace(0, N_EPOCH_CALI - N_EPOCH_CALI % MONITOR_WINDOW, int(
        (N_EPOCH_CALI - N_EPOCH_CALI % MONITOR_WINDOW) / MONITOR_WINDOW + 1))))
    _valid_acc_raw_record = np.zeros(len(np.linspace(0, N_EPOCH_CALI - N_EPOCH_CALI % MONITOR_WINDOW, int(
        (N_EPOCH_CALI - N_EPOCH_CALI % MONITOR_WINDOW) / MONITOR_WINDOW + 1))))
    _valid_acc_cali_record = np.zeros(len(np.linspace(0, N_EPOCH_CALI - N_EPOCH_CALI % MONITOR_WINDOW, int(
        (N_EPOCH_CALI - N_EPOCH_CALI % MONITOR_WINDOW) / MONITOR_WINDOW + 1))))
    _ece_raw_record = np.zeros(len(np.linspace(0, N_EPOCH_CALI - N_EPOCH_CALI % MONITOR_WINDOW, int(
        (N_EPOCH_CALI - N_EPOCH_CALI % MONITOR_WINDOW) / MONITOR_WINDOW + 1))))
    _ece_cali_record = np.zeros(len(np.linspace(0, N_EPOCH_CALI - N_EPOCH_CALI % MONITOR_WINDOW, int(
        (N_EPOCH_CALI - N_EPOCH_CALI % MONITOR_WINDOW) / MONITOR_WINDOW + 1))))
    _l1_raw_record = np.zeros(len(np.linspace(0, N_EPOCH_CALI - N_EPOCH_CALI % MONITOR_WINDOW, int(
        (N_EPOCH_CALI - N_EPOCH_CALI % MONITOR_WINDOW) / MONITOR_WINDOW + 1))))
    _l1_cali_record = np.zeros(len(np.linspace(0, N_EPOCH_CALI - N_EPOCH_CALI % MONITOR_WINDOW, int(
        (N_EPOCH_CALI - N_EPOCH_CALI % MONITOR_WINDOW) / MONITOR_WINDOW + 1))))
    _count = 0
    _best_raw_l1 = np.inf
    _best_raw_ece = np.inf
    _best_cali_l1 = np.inf
    _best_cali_ece = np.inf

    # moving average record for the network predictions
    gamma = args.gamma_initial*torch.ones(len(trainset)).to(device).float()
    gamma_weight = torch.zeros(len(gamma)).to(device).float()
    f_record = torch.zeros([args.rollWindow, len(y_train), NUM_CLASSES])
    correctness_record = torch.zeros(len(trainset))

    # Step I: Train Classifier
    for epoch in range(N_EPOCH_CLS):

        cprint(f">>>Epoch [{epoch + 1}|{N_EPOCH_CLS}] Training Classifier <<<", "green")

        train_correct = 0
        train_total = 0
        train_loss = 0

        # CLS Training
        model_cls.train()
        for ib, (indices, images, labels, _) in enumerate(tqdm(train_loader, ascii=True, ncols=100)):
            if images.shape[0] == 1:
                continue
            optimizer_cls.zero_grad()
            images, labels = images.to(device), labels.to(device)
            outs = model_cls(images)
            _, predict = outs.max(1)
            gamma_weight[indices] = gamma[indices]/gamma[indices].sum()
            loss = (gamma_weight[indices] * criterion_cls(outs, labels)).sum()
            loss.backward()
            optimizer_cls.step()

            train_loss += loss.detach().cpu().item()
            train_correct += predict.eq(labels).sum().item()
            train_total += len(labels)

            # record the network predictions
            outs_prob = torch.softmax(outs, 1)
            with torch.no_grad():
                f_record[epoch % args.rollWindow, indices] = outs_prob.detach().cpu()

        train_acc = train_correct / train_total
        # Update Gamma
        current_weight_increment = min(args.gamma_initial*np.exp(epoch*args.gamma_multiplier), 5)
        gamma[torch.where(correctness_record.bool())] += current_weight_increment
        gamma_weight = gamma / gamma.sum()
        scheduler_cls.step()

        # For monitoring purpose only
        # gt = (torch.tensor(np.array(h_star_train)).squeeze() == torch.tensor(y_tilde_train).squeeze())
        # print(f"Current weight increment: {current_weight_increment:.3f}")
        # print(f"Correct data weight ratio: {sum(gamma_weight[torch.where(torch.tensor(y_tilde_train).squeeze() == torch.tensor(h_star_train).squeeze())]):.3f}")
        # conf_precision = gt[torch.where(correctness_record.bool())].sum().float()/correctness_record.sum().float()
        # conf_recall = gt[torch.where(correctness_record.bool())].sum().float()/gt.sum().float()
        # print(f"Model confidence selection precision | recall: \t [{conf_precision:.3f}|{conf_recall:.3f}]")

        if not (epoch % MONITOR_WINDOW):

            valid_correct_raw = 0
            valid_correct_cali = 0
            valid_total = 0

            model_cls.eval()
            for _, (indices, images, labels, _) in enumerate(tqdm(valid_loader, ascii=True, ncols=100)):
                if images.shape[0] == 1:
                    continue
                images, labels = images.to(device), labels.to(device)
                outs_raw = model_cls(images)

                # Raw model result record
                prob_outs = torch.softmax(outs_raw[:, :NUM_CLASSES], 1)
                _, predict = prob_outs.max(1)
                correct_prediction = predict.eq(labels).float()
                valid_correct_raw += correct_prediction.sum().item()
                valid_total += len(labels)

            valid_acc_raw = valid_correct_raw/valid_total
            print(f"Step [{epoch + 1}|{N_EPOCH_CLS}] - Train Loss:{loss.item():.3f} \t Valid Acc: {valid_acc_raw:.3f}")

    # Testing stage
    test_correct_raw = 0
    test_total = 0

    model_cls.eval()
    for _, (indices, images, labels, _) in enumerate(tqdm(test_loader, ascii=True, ncols=100)):
        if images.shape[0] == 1:
            continue
        images, labels = images.to(device), labels.to(device)
        outs_raw = model_cls(images)

        # Raw model result record
        prob_outs = torch.softmax(outs_raw[:, :NUM_CLASSES], 1)
        _, predict = prob_outs.max(1)
        correct_prediction = predict.eq(labels).float()
        test_correct_raw += correct_prediction.sum().item()
        test_total += len(labels)

    test_acc_raw = test_correct_raw / test_total
    print(f"Final Testing Acc: {test_acc_raw:.3f}")

    # Step II: Calibration the model
    _, predict = f_record.mean(0).max(1)
    correctness = predict.eq(torch.tensor(y_tilde_train)).squeeze().float()
    trainset_cali.update_labels(correctness)

    for epoch in range(N_EPOCH_CALI):

        cprint(f">>>Epoch [{epoch + 1}|{N_EPOCH_CALI}] Training <<<", "green")

        train_loss = 0

        model_cali.train()
        for ib, (indices, images, response, _) in enumerate(tqdm(train_cali_loader, ascii=True, ncols=100)):
            if images.shape[0] == 1:
                continue
            optimizer_cali.zero_grad()
            images, response = images.to(device), response.to(device)
            #outs = model_cali(images)

            # Regress onto the correctness
            n_images = images.shape[0]
            images_aug = images.unsqueeze(1).repeat(1, N_SAMPLE, 1, 1, 1)
            images_aug = images_aug + torch.normal(0, 0.1, size=images_aug.shape).to(device)
            images_aug = images_aug.reshape(-1, INPUT_CHANNEL, INPUT_SHAPE, INPUT_SHAPE)
            n_aug = images_aug.shape[0]
            n_iter = np.ceil(n_aug / BATCH_SIZE)
            _train_f_cali = []
            for ib in range(int(n_iter)):
                _prob_outs_train = torch.clip(model_cali(images_aug[ib*BATCH_SIZE:min(n_aug, (ib+1)*BATCH_SIZE)]), 0, 1)
                _train_f_cali.append(_prob_outs_train.unsqueeze(0))
            train_prob = torch.clip(torch.cat(_train_f_cali).reshape(n_images, N_SAMPLE, -1).mean(1).squeeze(), 1e-4, 1-1e-4)
            loss_cali = -(response*train_prob.log() + (1-response)*(1-train_prob).log()).mean()

            # loss_cali = criterion_conf(outs, response.float())
            loss_cali.backward()
            optimizer_cali.step()

            train_loss+= loss_cali.item()

        if not epoch%MONITOR_WINDOW:

            # For ECE
            valid_f_raw = torch.zeros(len(validset), NUM_CLASSES).float()
            valid_f_cali = torch.zeros(len(validset), NUM_CLASSES).float()
            # For L1
            valid_f_raw_target_conf = torch.zeros(len(validset)).float()
            valid_f_cali_target_conf = torch.zeros(len(validset)).float()

            model_cls.eval()
            model_cali.eval()
            for _, (indices, images, labels, _) in enumerate(tqdm(valid_loader, ascii=True, ncols=100)):
                if images.shape[0] == 1:
                    continue
                images, labels = images.to(device), labels.to(device)
                outs_raw = model_cls(images)

                # Raw model result record
                prob_outs = torch.softmax(outs_raw, 1)
                _, predict = prob_outs.max(1)
                correct_prediction = predict.eq(labels).float()
                valid_correct_raw += correct_prediction.sum().item()
                valid_total += len(labels)
                valid_f_raw[indices] = prob_outs.detach().cpu()
                valid_f_raw_target_conf[indices] = prob_outs.max(1)[0].detach().cpu()

                # We only calibrate single class and replace the predicted prob to be the calibrated prob
                n_images = images.shape[0]
                images_aug = images.unsqueeze(1).repeat(1, N_SAMPLE, 1, 1, 1)
                images_aug = images_aug + torch.normal(0, 0.1, size=images_aug.shape).to(device)
                images_aug = images_aug.reshape(-1, INPUT_CHANNEL, INPUT_SHAPE, INPUT_SHAPE)
                n_aug = images_aug.shape[0]
                n_iter = np.ceil(n_aug/BATCH_SIZE)
                _valid_f_cali = []
                for ib in range(int(n_iter)):
                    image_outs_raw = torch.clip(model_cali(images_aug[ib*BATCH_SIZE:min(n_aug, (ib+1)*BATCH_SIZE)]), 0, 1)
                    _valid_f_cali.append(image_outs_raw.squeeze())
                _valid_f_cali = torch.cat(_valid_f_cali).reshape(n_images, N_SAMPLE, -1).mean(1).detach().cpu()

                # _valid_f_cali = torch.clip(model_cali(images), 0, 1).detach().cpu()
                valid_f_raw[indices] = prob_outs.detach().cpu()
                valid_f_cali[indices] = prob_outs.detach().cpu().scatter_(1, predict.detach().cpu()[:, None], _valid_f_cali)
                valid_f_cali_target_conf[indices] = _valid_f_cali.squeeze()
                valid_correct_cali += valid_f_cali[indices].max(1)[1].eq(labels.detach().cpu()).sum().item()

            valid_acc_raw = valid_correct_raw / valid_total
            valid_acc_cali = valid_correct_cali / valid_total
            ece_loss_raw = criterion_calibrate.forward(logits=valid_f_raw, labels=torch.tensor(y_tilde_valid))
            ece_loss_cali = criterion_calibrate.forward(logits=valid_f_cali, labels=torch.tensor(y_tilde_valid))
            l1_loss_raw = criterion_l1(valid_f_raw_target_conf, torch.tensor(eta_tilde_valid).max(1)[0])
            l1_loss_cali = criterion_l1(valid_f_cali_target_conf, torch.tensor(eta_tilde_valid).max(1)[0])
            print(f"Step [{epoch + 1}|{N_EPOCH_CALI}] - Train Loss: {loss_cali.item():.3f}" +
                  f"- Valid Acc Raw:{valid_acc_raw:7.3f} - ECE Loss:{ece_loss_raw.item():7.3f}" +
                  f"- Valide Acc Cali:{valid_acc_cali:.3f} - ECE Loss Cali:{ece_loss_cali.item():.3f}")

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
        test_total = 0
        # For ECE
        test_f_raw = torch.zeros(len(testset), NUM_CLASSES).float()
        test_f_cali = torch.zeros(len(testset), NUM_CLASSES).float()
        # For L1
        test_f_raw_target_conf = torch.zeros(len(testset)).float()
        test_f_cali_target_conf = torch.zeros(len(testset)).float()

        model_cls.eval()
        model_cali.eval()
        for _, (indices, images, labels, _) in enumerate(tqdm(test_loader, ascii=True, ncols=100)):
            if images.shape[0] == 1:
                continue
            images, labels = images.to(device), labels.to(device)
            outs_raw = model_cls(images)

            # Raw model result record
            prob_outs = torch.softmax(outs_raw, 1)
            _, predict = prob_outs.max(1)
            correct_prediction = predict.eq(labels).float()
            test_correct_raw += correct_prediction.sum().item()
            test_total += len(labels)
            test_f_raw[indices] = prob_outs.detach().cpu()
            test_f_raw_target_conf[indices] = prob_outs.max(1)[0].detach().cpu()

            n_images = images.shape[0]
            images_aug = images.unsqueeze(1).repeat(1, N_SAMPLE, 1, 1, 1)
            images_aug = images_aug + torch.normal(0, 0.1, size=images_aug.shape).to(device)
            images_aug = images_aug.reshape(-1, INPUT_CHANNEL, INPUT_SHAPE, INPUT_SHAPE)
            n_aug = images_aug.shape[0]
            n_iter = np.ceil(n_aug / BATCH_SIZE)
            _test_conf_cali = []
            for ib in range(int(n_iter)):
                image_outs_raw = torch.clip(model_cali(images_aug[ib * BATCH_SIZE:min(n_aug, (ib+1)*BATCH_SIZE)]), 0, 1)
                _test_conf_cali.append(image_outs_raw.squeeze())
            _test_conf_cali = torch.cat(_test_conf_cali).reshape(n_images, N_SAMPLE, -1).mean(1).detach().cpu()

            # _test_conf_cali = torch.clip(model_cali(images), 0, 1).detach().cpu()
            test_f_cali[indices, :] = test_f_raw[indices, :].scatter_(1, predict.detach().cpu()[:, None], _test_conf_cali)
            test_f_cali_target_conf[indices] = _test_conf_cali.squeeze()

        scheduler_cali.step()

        print(f"Real {torch.tensor(eta_tilde_test).max(1)[0][:5]}")
        print(f"Naive {test_f_raw.max(1)[0][:5]}")
        print(f"Ours {test_f_cali_target_conf[:5]}")
        print(f"Cali output {_test_conf_cali.detach().cpu()[:5].squeeze()}")

        ece_loss_raw = criterion_calibrate.forward(logits=test_f_raw, labels=torch.tensor(y_tilde_test))
        ece_loss_cali = criterion_calibrate.forward(logits=test_f_cali, labels=torch.tensor(y_tilde_test))
        l1_loss_raw = criterion_l1(test_f_raw_target_conf, torch.tensor(eta_tilde_test).max(1)[0])
        l1_loss_cali = criterion_l1(test_f_cali_target_conf, torch.tensor(eta_tilde_test).max(1)[0])

        if l1_loss_raw < _best_raw_l1:
            _best_raw_l1 =l1_loss_raw.item()
            _best_raw_l1_epoch = epoch
        if l1_loss_cali < _best_cali_l1:
            _best_cali_l1 = l1_loss_cali.item()
            _best_cali_l1_epoch = epoch
        if ece_loss_raw < _best_raw_ece:
            _best_raw_ece = ece_loss_raw.item()
            _best_raw_ece_epoch = epoch
        if ece_loss_cali < _best_cali_ece:
            _best_cali_ece = ece_loss_cali.item()
            _best_cali_ece_epoch = epoch

        print(f"[Current | Best] L1 - Conf: \t [{l1_loss_raw.item():.3f} | {_best_raw_l1:.3f}] \t epoch {_best_raw_l1_epoch}")
        print(f"[Current | Best] L1 - Ours: \t [{l1_loss_cali.item():.3f} | {_best_cali_l1:.3f}] \t epoch {_best_cali_l1_epoch}")
        print(f"[Current | Best] ECE - Conf: \t [{ece_loss_raw.item():.3f} | {_best_raw_ece:.3f}] \t epoch {_best_raw_ece_epoch}")
        print(f"[Current | Best] ECE - Ours: \t [{ece_loss_cali.item():.3f} | {_best_cali_ece:.3f}] \t epoch {_best_cali_ece_epoch}")
        print(f"Final Test Acc: {test_correct_raw/test_total*100:.3f}%")


    if args.figure:
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.use('Agg')
        if not os.path.exists('./figures'):
            os.makedirs('./figures', exist_ok=True)
        fig = plt.figure(figsize=(10, 10))
        plt.subplot(2, 2, 1)
        x_axis = np.linspace(0, N_EPOCH_CLS - N_EPOCH_CLS % MONITOR_WINDOW,
                             int((N_EPOCH_CLS - N_EPOCH_CLS % MONITOR_WINDOW) / MONITOR_WINDOW + 1))
        plt.plot(x_axis[:-1], _loss_record[:-1], linewidth=2)
        plt.title("Loss Curve")
        plt.subplot(2, 2, 2)
        plt.plot(x_axis[:-1], _valid_acc_raw_record[:-1], linewidth=2, label="Raw")
        plt.plot(x_axis[:-1], _valid_acc_cali_record[:-1], linewidth=2, label="Ours2Stage")
        plt.title('Acc Curve')
        plt.subplot(2, 2, 3)
        plt.plot(x_axis[:-1], _ece_raw_record[:-1], linewidth=2, label="Raw")
        plt.plot(x_axis[:-1], _ece_cali_record[:-1], linewidth=2, label="Ours2Stage")
        plt.legend()
        plt.title('ECE Curve')
        plt.subplot(2, 2, 4)
        plt.plot(x_axis[:-1], _l1_raw_record[:-1], linewidth=2, label="Raw")
        plt.plot(x_axis[:-1], _l1_cali_record[:-1], linewidth=2, label="Ours2Stage")
        plt.legend()
        plt.title('L1 Curve')

        time_stamp = datetime.datetime.strftime(datetime.datetime.today(), "%Y-%m-%d")
        fig.savefig(os.path.join("./figures", f"exp_log_{args.dataset}_{args.noise_type}_{args.noise_strength}_Ours2Stage_plot_{time_stamp}.png"))

    return l1_loss_raw.item(), ece_loss_raw.item(), l1_loss_cali.item(), ece_loss_cali.item(), _best_cali_l1, _best_cali_ece


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from  2"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

class ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguement for SLDenoise")
    parser.add_argument("--seed", type=int, help="Random seed for the experiment", default=77)
    parser.add_argument("--gpus", type=str, help="Indices of GPUs to be used", default='0')
    parser.add_argument("--dataset", type=str, help="Experiment Dataset", default='cifar10',
                        choices={'mnist', 'cifar10', 'cifar100'})
    parser.add_argument("--noise_type", type=str, help="Noise type", default='uniform',
                        choices={"linear", "uniform", "asymmetric", "idl"})
    parser.add_argument("--noise_strength", type=float, help="Noise fraction", default=0.8)
    parser.add_argument("--rollWindow", default=3, help="rolling window to calculate the confidence", type=int)
    parser.add_argument("--warm_up", default=2, help="warm-up period", type=int)
    parser.add_argument("--figure", action='store_true', help='True to plot performance log')
    # algorithm hp
    parser.add_argument("--alpha", type=float, help="CE loss multiplier", default=100)
    parser.add_argument("--gamma_initial", type=float, help="Gamma initial value", default=0.8)
    parser.add_argument("--gamma_multiplier", type=float, help="Gamma increment multiplier", default=0.2)
    # Not useful currently
    parser.add_argument("--delta", default=0.1, help="initial threshold", type=float)
    parser.add_argument("--inc", default=0.1, help="increment", type=float)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    exp_config = {}
    exp_config['n_epoch_cls']  = N_EPOCH_CLS
    exp_config['n_epoch_cali'] = N_EPOCH_CALI
    exp_config['lr'] = LR
    exp_config['weight_decay'] = WEIGHT_DECAY
    exp_config['batch_size'] = BATCH_SIZE
    exp_config['mile_stone'] = SCHEDULER_DECAY_MILESTONE
    exp_config['train_vs_valid'] = TRAIN_VALIDATION_RATIO
    exp_config['monitor_window'] = MONITOR_WINDOW
    exp_config['record_freq'] = CONF_RECORD_EPOCH
    for k, v in args._get_kwargs():
        exp_config[k] = v

    exp_config['oursv2_l1'] = []
    exp_config['oursv2_ece'] = []
    exp_config['oursv2_best_l1'] = []
    exp_config['oursv2_best_ece'] = []
    for seed in [77, 78, 79]:
        args.seed = seed
        _, _, ours_l1,  ours_ece, best_l1, best_ece = main(args)
        exp_config['ours2stage_l1'].append(ours_l1)
        exp_config['ours2stage_ece'].append(ours_ece)
        exp_config['ours2stage_best_l1'].append(best_l1)
        exp_config['ours2stage_best_ece'].append(best_ece)

    dir_date = datetime.datetime.today().strftime("%Y%m%d")
    save_folder = os.path.join("./exp_logs/ours2stage_"+dir_date)
    os.makedirs(save_folder, exist_ok=True)
    save_file_name = 'ours2stage_' + datetime.date.today().strftime("%d_%m_%Y") + f"_{args.seed}_{args.dataset}_{args.noise_type}_{args.noise_strength}.json"
    save_file_name = os.path.join(save_folder, save_file_name)
    print(save_file_name)
    with open(save_file_name, "w") as f:
        json.dump(exp_config, f, sort_keys=False, indent=4)
    f.close()