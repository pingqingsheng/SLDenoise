import os
from typing import List
import json
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
import datetime

from data.MNIST import MNIST, MNIST_Combo
from data.CIFAR import CIFAR10, CIFAR10_Combo
from network.network import resnet18, resnet34
from utils.utils import _init_fn, ECELoss, my_logits
from utils.noise import perturb_eta, noisify_with_P, noisify_mnist_asymmetric, noisify_cifar10_asymmetric

# Experiment Setting Control Panel
# ---------------------------------------------------
TRAIN_VALIDATION_RATIO: float = 0.8
N_EPOCH_OUTER: int = 1
N_EPOCH_INNER_CLS: int = 200
CONF_RECORD_EPOCH: int = N_EPOCH_INNER_CLS - 1
LR: float = 1e-3
WEIGHT_DECAY: float = 5e-3
BATCH_SIZE: int = 128
SCHEDULER_DECAY_MILESTONE: List = [40, 80, 120]
MONITOR_WINDOW: int = 1
# ----------------------------------------------------

def lrt_correction(y_tilde, f_x, current_delta=0.3, delta_increment=0.1):
    """
    Label correction using likelihood ratio test.
    In effect, it gradually decreases the threshold according to Algorithm 1.

    current_delta: The initial threshold $\theta$
    delta_increment: The step size, corresponding to the $\beta$ in Algorithm 1.
    """
    corrected_count = 0
    y_noise = torch.tensor(y_tilde).clone()
    n = len(y_noise)
    f_m = f_x.max(1)[0]
    y_mle = f_x.argmax(1)
    LR = []
    for i in range(len(y_noise)):
        LR.append(float(f_x[i][int(y_noise[i])] / f_m[i]))

    for i in range(int(len(y_noise))):
        if LR[i] < current_delta:
            y_noise[i] = y_mle[i]
            corrected_count += 1

    if corrected_count < 0.001 * n:
        current_delta += delta_increment
        current_delta = min(current_delta, 0.9)
        print("Update Critical Value -> {}".format(current_delta))

    return y_noise, current_delta


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
    if args.dataset == 'mnist':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        trainset = MNIST(root="./data", split="train", train_ratio=TRAIN_VALIDATION_RATIO, download=True,
                         transform=transform_train)
        validset = MNIST(root="./data", split="valid", train_ratio=TRAIN_VALIDATION_RATIO, transform=transform_train)
        testset = MNIST(root="./data", split="test", download=True, transform=transform_test)
        input_channel = 1
        num_classes = 10
        if args.noise_type == 'idl':
            model_cls_clean = torch.load(f"./data/MNIST_resnet18_clean_{int(args.noise_strength * 100)}.pth", map_location=device).module
        else:
            model_cls_clean = torch.load("./data/MNIST_resnet18_clean.pth", map_location=device).module
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
        trainset = CIFAR10(root="./data", split="train", train_ratio=TRAIN_VALIDATION_RATIO, download=True,
                           transform=transform_train)
        validset = CIFAR10(root="./data", split="valid", train_ratio=TRAIN_VALIDATION_RATIO, download=True,
                           transform=transform_train)
        testset = CIFAR10(root='./data', split="test", download=True, transform=transform_test)
        input_channel = 3
        num_classes = 10
        if args.noise_type == 'idl':
            model_cls_clean = torch.load(f"./data/CIFAR10_resnet18_clean_{int(args.noise_strength * 100)}.pth", map_location=device).module
        else:
            model_cls_clean = torch.load("./data/CIFAR10_resnet18_clean.pth", map_location=device).module

    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, worker_init_fn=_init_fn(worker_id=seed))
    valid_loader = DataLoader(validset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, worker_init_fn=_init_fn(worker_id=seed))
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, worker_init_fn=_init_fn(worker_id=seed))

    # Inject noise here
    y_train = copy.deepcopy(trainset.targets)
    y_valid = copy.deepcopy(validset.targets)
    y_test = testset.targets

    cprint(">>> Inject Noise <<<", "green")
    gpu_id_list = [int(x) for x in args.gpus.split(",")]
    model_cls_clean = DataParallel(model_cls_clean, device_ids=[x-gpu_id_list[0] for x in gpu_id_list])
    _eta_train_temp_pair = [(torch.softmax(model_cls_clean(images).to(device), 1).detach().cpu(), indices) for
                            _, (indices, images, labels, _) in enumerate(tqdm(train_loader, ascii=True, ncols=100))]
    _eta_valid_temp_pair = [(torch.softmax(model_cls_clean(images).to(device), 1).detach().cpu(), indices) for
                            _, (indices, images, labels, _) in enumerate(tqdm(valid_loader, ascii=True, ncols=100))]
    _eta_test_temp_pair = [(torch.softmax(model_cls_clean(images).to(device), 1).detach().cpu(), indices) for
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

    if args.noise_type == 'linear':
        eta_tilde_train = perturb_eta(eta_train, args.noise_type, args.noise_strength)
        eta_tilde_valid = perturb_eta(eta_valid, args.noise_type, args.noise_strength)
        eta_tilde_test = perturb_eta(eta_test, args.noise_type, args.noise_strength)
        y_tilde_train = [int(np.where(np.random.multinomial(1, x, 1).squeeze())[0]) for x in eta_tilde_train]
        y_tilde_valid = [int(np.where(np.random.multinomial(1, x, 1).squeeze())[0]) for x in eta_tilde_valid]
        trainset.update_labels(y_tilde_train)
    elif args.noise_type == 'uniform':
        y_tilde_train, P, _ = noisify_with_P(np.array(copy.deepcopy(h_star_train)), nb_classes=10,
                                             noise=args.noise_strength)
        eta_tilde_train = np.matmul(F.one_hot(h_star_train, num_classes=10), P)
        eta_tilde_valid = np.matmul(F.one_hot(h_star_valid, num_classes=10), P)
        eta_tilde_test = np.matmul(F.one_hot(h_star_test, num_classes=10), P)
        trainset.update_labels(y_tilde_train)
    elif args.noise_type == 'asymmetric':
        if args.dataset == "mnist":
            y_tilde_train, P, _ = noisify_mnist_asymmetric(np.array(copy.deepcopy(h_star_train)),
                                                           noise=args.noise_strength)
        elif args.dataset == 'cifar10':
            y_tilde_train, P, _ = noisify_cifar10_asymmetric(np.array(copy.deepcopy(h_star_train)),
                                                             noise=args.noise_strength)
        eta_tilde_train = np.matmul(F.one_hot(h_star_train, num_classes=10), P)
        eta_tilde_valid = np.matmul(F.one_hot(h_star_valid, num_classes=10), P)
        eta_tilde_test = np.matmul(F.one_hot(h_star_test, num_classes=10), P)
        trainset.update_labels(y_tilde_train)
    elif args.noise_type == "idl":
        eta_tilde_train = copy.deepcopy(eta_train)
        eta_tilde_valid = copy.deepcopy(eta_valid)
        eta_tilde_test = copy.deepcopy(eta_test)
        y_tilde_train = [int(np.where(np.random.multinomial(1, x, 1).squeeze())[0]) for x in eta_tilde_train]
        y_tilde_valid = [int(np.where(np.random.multinomial(1, x, 1).squeeze())[0]) for x in eta_tilde_valid]
        trainset.update_labels(y_tilde_train)

    validset.update_labels(h_star_valid)
    testset.update_labels(h_star_test)
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

    model_cls = resnet34(num_classes=num_classes + 2, in_channels=input_channel)
    model_cls = DataParallel(model_cls, device_ids=[x-gpu_id_list[0] for x in gpu_id_list])
    model_cls = model_cls.to(device)

    optimizer_cls = torch.optim.Adam(model_cls.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler_cls = torch.optim.lr_scheduler.MultiStepLR(optimizer_cls, gamma=0.5, milestones=SCHEDULER_DECAY_MILESTONE)

    criterion_cls = torch.nn.CrossEntropyLoss(reduction='none')
    criterion_conf = torch.nn.MSELoss()
    criterion_select = torch.nn.BCELoss()
    criterion_calibrate = ECELoss()
    criterion_l1 = torch.nn.L1Loss()

    test_conf = torch.zeros([len(testset), num_classes])

    # For monitoring purpose
    _loss_record = np.zeros(len(np.linspace(0, N_EPOCH_INNER_CLS - N_EPOCH_INNER_CLS % MONITOR_WINDOW, int(
        (N_EPOCH_INNER_CLS - N_EPOCH_INNER_CLS % MONITOR_WINDOW) / MONITOR_WINDOW + 1))))
    _acc_record = np.zeros(len(np.linspace(0, N_EPOCH_INNER_CLS - N_EPOCH_INNER_CLS % MONITOR_WINDOW, int(
        (N_EPOCH_INNER_CLS - N_EPOCH_INNER_CLS % MONITOR_WINDOW) / MONITOR_WINDOW + 1))))
    _ece_record_conf = np.zeros(len(np.linspace(0, N_EPOCH_INNER_CLS - N_EPOCH_INNER_CLS % MONITOR_WINDOW, int(
        (N_EPOCH_INNER_CLS - N_EPOCH_INNER_CLS % MONITOR_WINDOW) / MONITOR_WINDOW + 1))))
    _ece_record_ours = np.zeros(len(np.linspace(0, N_EPOCH_INNER_CLS - N_EPOCH_INNER_CLS % MONITOR_WINDOW, int(
        (N_EPOCH_INNER_CLS - N_EPOCH_INNER_CLS % MONITOR_WINDOW) / MONITOR_WINDOW + 1))))
    _mse_record_conf = np.zeros(len(np.linspace(0, N_EPOCH_INNER_CLS - N_EPOCH_INNER_CLS % MONITOR_WINDOW, int(
        (N_EPOCH_INNER_CLS - N_EPOCH_INNER_CLS % MONITOR_WINDOW) / MONITOR_WINDOW + 1))))
    _mse_record_ours = np.zeros(len(np.linspace(0, N_EPOCH_INNER_CLS - N_EPOCH_INNER_CLS % MONITOR_WINDOW, int(
        (N_EPOCH_INNER_CLS - N_EPOCH_INNER_CLS % MONITOR_WINDOW) / MONITOR_WINDOW + 1))))
    _mse_record_conf_toclean = np.zeros(len(np.linspace(0, N_EPOCH_INNER_CLS - N_EPOCH_INNER_CLS % MONITOR_WINDOW, int(
        (N_EPOCH_INNER_CLS - N_EPOCH_INNER_CLS % MONITOR_WINDOW) / MONITOR_WINDOW + 1))))
    _mse_record_ours_toclean = np.zeros(len(np.linspace(0, N_EPOCH_INNER_CLS - N_EPOCH_INNER_CLS % MONITOR_WINDOW, int(
        (N_EPOCH_INNER_CLS - N_EPOCH_INNER_CLS % MONITOR_WINDOW) / MONITOR_WINDOW + 1))))
    _count = 0
    _best_naive_l1 = np.inf
    _best_ours_l1 = np.inf
    _best_naive_ece = np.inf
    _best_ours_ece = np.inf

    # moving average record for the network predictions
    gamma = args.gamma_initial*torch.ones(len(trainset)).to(device).float()
    gamma_weight = gamma/gamma.sum().item()
    f_record = torch.zeros([args.rollWindow, len(y_train), num_classes])
    pred_correctness_record = torch.zeros(len(trainset))
    correctness_record = torch.zeros(len(trainset))
    last_picked = torch.ones(len(trainset)).bool()
    current_delta = args.delta  # for LRT

    for outer_epoch in range(N_EPOCH_OUTER):

        cprint(f">>>Epoch [{outer_epoch + 1}|{N_EPOCH_OUTER}] Train Share Backbone Model <<<", "green")

        for inner_epoch in range(N_EPOCH_INNER_CLS):
            train_correct = 0
            train_total = 0
            train_loss = 0

            model_cls.train()
            for _, (indices, images, labels, _) in enumerate(tqdm(train_loader, ascii=True, ncols=100)):
                if images.shape[0] == 1:
                    continue
                optimizer_cls.zero_grad()
                images, labels = images.to(device), labels.to(device)
                outs = model_cls(images)
                _, predict = outs[:, :num_classes].max(1)
                with torch.no_grad():
                    delta_prediction = predict.eq(labels).float()

                loss_ce = (args.alpha*gamma_weight[indices]*criterion_cls(outs[:, :num_classes], labels)).sum()
                loss_cali = criterion_conf(torch.sigmoid(outs[:, num_classes:(num_classes+1)]).squeeze(), delta_prediction.squeeze())
                loss_sl = criterion_select(torch.sigmoid(outs[:, (num_classes+1):]).squeeze(), delta_prediction.squeeze())
                # loss_en = -(torch.softmax(outs[:, :num_classes], 1) * torch.log(torch.softmax(outs[:, :num_classes], 1))).mean()
                # loss_sm = -((1 / num_classes * torch.ones(outs[:, :num_classes].shape).to(device)) * torch.log(torch.softmax(outs[:, :num_classes], 1))).mean()
                # loss = loss_main + loss_en + loss_sm
                loss = loss_ce + loss_cali + loss_sl
                loss.backward()
                optimizer_cls.step()

                train_loss += loss.detach().cpu().item()
                train_correct += predict.eq(labels).sum().item()
                train_total += len(labels)

                # record the network predictions
                with torch.no_grad():
                    f_record[inner_epoch % args.rollWindow, indices] = F.softmax(outs.detach().cpu()[:, :num_classes],dim=1)
                    gamma[indices] = torch.sigmoid(outs[:, (num_classes+1):]).squeeze()
                    correctness_record[indices] = delta_prediction.detach().cpu()
                    pred_correctness_record[indices] = (torch.sigmoid(outs[:, (num_classes+1):]).squeeze() > 0.5).detach().cpu().float()

            scheduler_cls.step()

            train_acc = train_correct / train_total
            current_weight_increment = min(args.gamma_initial*np.exp(inner_epoch*args.gamma_multiplier), 20)
            gamma[torch.where(correctness_record.bool())] += current_weight_increment
            # gamma[torch.where(pred_correctness_record.bool())] += current_weight_increment
            gamma[torch.where(correctness_record.bool() & pred_correctness_record.bool())] += current_weight_increment
            gamma_weight = gamma/gamma.sum()

            gt = (torch.tensor(np.array(h_star_train)).squeeze() == torch.tensor(y_tilde_train).squeeze())
            print(f"Current weight increment: {current_weight_increment:.3f}")
            print(f"Correct data weight ratio: {sum(gamma_weight[torch.where(torch.tensor(y_tilde_train).squeeze() == torch.tensor(h_star_train).squeeze())]):.3f}")
            # pred_neuron_precision = gt[torch.where(pred_correctness_record.bool())].sum().float()/pred_correctness_record.sum().float()
            # pred_neuron_recall = gt[torch.where(pred_correctness_record.bool())].sum().float()/gt.sum().float()
            conf_precision = gt[torch.where(correctness_record.bool())].sum().float()/correctness_record.sum().float()
            conf_recall = gt[torch.where(correctness_record.bool())].sum().float()/gt.sum().float()
            # last_pick_precision = gt[torch.where(last_picked.bool())].sum().float()/last_picked.sum().float()
            # last_pick_recall = gt[torch.where(last_picked.bool())].sum().float()/gt.sum().float()

            # print(f"Prediction neurons selection precision | recall: \t [{pred_neuron_precision}|{pred_neuron_recall}]")
            print(f"Model confidence selection precision | recall: \t [{conf_precision}|{conf_recall}]")
            # print(f"Last picked selection precision | recall: \t [{last_pick_precision}|{last_pick_recall}]")

            if not (inner_epoch % MONITOR_WINDOW):

                valid_correct = 0
                valid_total = 0
                ece_loss = 0
                valid_predict = torch.zeros(len(validset), num_classes).float()
                # For ECE
                valid_f_raw = torch.zeros(len(validset), num_classes).float()
                valid_f_cali = torch.zeros(len(validset), num_classes).float()
                # For L1
                valid_f_cali_predict_class = torch.zeros(len(validset)).float()

                model_cls.eval()
                for _, (indices, images, labels, _) in enumerate(tqdm(valid_loader, ascii=True, ncols=100)):
                    if images.shape[0] == 1:
                        continue
                    images, labels = images.to(device), labels.to(device)
                    outs = model_cls(images)

                    prob_outs = torch.softmax(outs[:, :num_classes], 1)
                    _, predict = prob_outs.max(1)
                    correct_prediction = predict.eq(labels).float()
                    valid_correct += correct_prediction.sum().item()
                    valid_total += len(labels)

                    # We only calibrate single class and replace the predicted prob to be the calibrated prob
                    _valid_f_cali = torch.sigmoid(outs[:, num_classes:(num_classes+1)]).detach().cpu()
                    reg_loss = criterion_l1(_valid_f_cali, torch.tensor(eta_tilde_valid[indices]).max(1)[0])
                    valid_f_raw[indices] = prob_outs.detach().cpu()
                    # replace corresponding element
                    valid_f_cali[indices] = prob_outs.detach().cpu().scatter_(1, predict.detach().cpu()[:, None], _valid_f_cali)
                    valid_f_cali_predict_class[indices] = _valid_f_cali.squeeze()

                valid_acc = valid_correct / valid_total
                ece_loss = criterion_calibrate.forward(logits=valid_f_cali,labels=torch.tensor(eta_tilde_valid).argmax(1).squeeze())
                print(f"Step [{inner_epoch + 1}|{N_EPOCH_INNER_CLS}] - Train Loss:[{loss_ce.item():.3f} | {loss_cali.item():.3f}] - Train Acc: {train_acc:7.3f} - Valid Acc: {valid_acc:7.3f}-Valid Reg Loss (L1): {reg_loss:7.3f}-ECE Loss: {ece_loss.item():7.3f}")

                # For monitoring purpose
                _loss_record[_count] = float(criterion_cls(outs[:, :num_classes], labels).mean())
                _acc_record[_count] = float(valid_acc)
                _ece_record_conf[_count] = float(criterion_calibrate.forward(logits=valid_f_raw, labels=torch.tensor(eta_tilde_valid).argmax(1).squeeze()).item())
                _ece_record_ours[_count] = float(ece_loss)
                _mse_record_conf[_count] = float(criterion_l1(valid_f_raw.max(1)[0], eta_tilde_valid.max(1)[0]))
                _mse_record_ours[_count] = float(criterion_l1(valid_f_cali_predict_class, eta_tilde_valid.max(1)[0]))
                _mse_record_conf_toclean[_count] = float(criterion_l1(valid_f_raw.max(1)[0], eta_valid.max(1)[0]))
                _mse_record_ours_toclean[_count] = float(criterion_l1(valid_f_cali_predict_class, eta_valid.max(1)[0]))
                _count += 1

            model_cls.eval()
            # Classification Final Test
            test_correct = 0
            test_total = 0
            test_predict = torch.zeros(len(testset), num_classes).float()
            # For ECE
            test_conf_raw = torch.zeros(len(testset), num_classes)
            test_conf_cali = torch.zeros(len(testset), num_classes)
            # For L1
            test_conf_cali_predict_class = torch.zeros(len(testset))

            model_cls.eval()
            for _, (indices, images, labels, _) in enumerate(test_loader):
                if images.shape[0] == 1:
                    continue
                images, labels = images.to(device), labels.to(device)
                outs = model_cls(images)
                _, predict = outs[:, :num_classes].max(1)
                test_correct += predict.eq(labels).sum().item()
                test_total += len(labels)

                _test_conf_cali = torch.sigmoid(outs[:, num_classes:(num_classes+1)]).detach().cpu()
                test_conf_raw[indices, :] = torch.softmax(outs[:, :num_classes], 1).detach().cpu()
                test_conf_cali[indices, :] = test_conf_raw[indices, :].scatter_(1, predict.detach().cpu()[:, None],_test_conf_cali)
                test_conf_cali_predict_class[indices] = _test_conf_cali.squeeze()
                # Temperature Scaling Baseline

                # ---------------------------------------------------- Debugging Purpose -------------------------------------------------------------
                # # Perform label correction
                # if (outer_epoch + 1) >= args.warm_up:
                #     f_x = f_record.mean(0)
                #     y_tilde = trainset.targets
                #     y_corrected, current_delta = lrt_correction(np.array(y_tilde).copy(), f_x, current_delta=current_delta, delta_increment=args.inc)
                #     trainset.targets = y_corrected.numpy().copy()  # update the training labels
                #     # y_corrected = f_x.argmax(1).squeeze()
                #     # trainset.targets = y_corrected
                #     cprint(f"Performed Label Correction", "yellow")
                # --------------------------------------------------------------------------------------------------------------------------------------

            naive_l1 = criterion_l1(test_conf_raw.max(1)[0], torch.tensor(eta_tilde_test).max(1)[0])
            ours_l1 = criterion_l1(test_conf_cali_predict_class, torch.tensor(eta_tilde_test).max(1)[0])
            naive_ece = criterion_calibrate.forward(logits=test_conf_raw,labels=torch.tensor(eta_tilde_test).argmax(1).squeeze())
            ours_ece = criterion_calibrate.forward(logits=test_conf_cali,labels=torch.tensor(eta_tilde_test).argmax(1).squeeze())
            print(f"Real {torch.tensor(eta_tilde_test).max(1)[0][:5]}")
            print(f"Naive {test_conf_raw.max(1)[0][:5]}")
            print(f"Ours {test_conf_cali_predict_class[:5]}")

            if naive_l1 <= _best_naive_l1:
                _best_naive_l1 = naive_l1.item()
                _best_naive_l1_epoch = inner_epoch
            if ours_l1 <= _best_ours_l1:
                _best_ours_l1 = ours_l1.item()
                _best_ours_l1_epoch = inner_epoch
            if naive_ece <= _best_naive_ece:
                _best_naive_ece = naive_ece.item()
                _best_naive_ece_epoch = inner_epoch
            if ours_ece <= _best_ours_ece:
                _best_ours_ece = ours_ece.item()
                _best_ours_ece_epoch = inner_epoch

            # print("Test eta_tilde: ", eta_tilde_test[:5])
            # print("Test test_confidence: ", test_conf[:5])
            # print("Test calibrated conf: ", test_conf_cali[:5])
            print(f"[Current | Best] L1 - Conf: \t [{naive_l1.item():.3f} | {_best_naive_l1:.3f}] \t epoch {_best_naive_l1_epoch}")
            print(f"[Current | Best] L1 - Ours: \t [{ours_l1.item():.3f} | {_best_ours_l1:.3f}] \t epoch {_best_ours_l1_epoch}")
            print(f"[Current | Best] ECE - Conf: \t [{naive_ece.item():.3f} | {_best_naive_ece:.3f}] \t epoch {_best_naive_ece_epoch}")
            print(f"[Current | Best] ECE - Ours: \t [{ours_ece.item():.3f} | {_best_ours_ece:.3f}] \t epoch {_best_ours_ece_epoch}")
            print("Final Test Acc: ", test_correct / test_total * 100, "%")

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
        plt.plot(x_axis[:-1], _acc_record[:-1], linewidth=2)
        plt.title('Acc Curve')
        plt.subplot(2, 2, 3)
        plt.plot(x_axis[:-1], _ece_record_conf[:-1], linewidth=2, label="Confidence")
        plt.plot(x_axis[:-1], _ece_record_ours[:-1], linewidth=2, label="Ours_weighted")
        plt.legend()
        plt.title('ECE Curve')
        plt.subplot(2, 2, 4)
        plt.plot(x_axis[:-1], _mse_record_conf[:-1], linewidth=2, label="Confidence v.s. Noisy")
        plt.plot(x_axis[:-1], _mse_record_ours[:-1], linewidth=2, label="Ours_weighted v.s. Noisy")
        plt.plot(x_axis[:-1], _mse_record_conf_toclean[:-1], linestyle='--' , linewidth=2, label="Confidence v.s. Clean")
        plt.plot(x_axis[:-1], _mse_record_ours_toclean[:-1], linestyle='--' , linewidth=2, label="Ours_weighted v.s. Clean")
        plt.legend()
        plt.title('L1 Curve')

        time_stamp = datetime.datetime.strftime(datetime.datetime.today(), "%Y-%m-%d")
        fig.savefig(os.path.join("./figures",f"exp_log_{args.dataset}_{args.noise_type}_{args.noise_strength}_binary_share_weighted_plot_{time_stamp}.png"))

    return _best_naive_l1, _best_ours_l1, _best_naive_ece, _best_ours_ece
    # return naive_l1, ours_l1, naive_ece, ours_ece


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
    parser.add_argument("--figure", action='store_true', help='True to plot performance log')
    # algorithm hp
    parser.add_argument("--alpha", type=float, help="CE loss multiplier", default=100)
    parser.add_argument("--gamma_initial", type=float, help="Gamma initial value", default=0.5)
    parser.add_argument("--gamma_multiplier", type=float, help="Gamma increment multiplier", default=0.1)
    # Not useful currently
    parser.add_argument("--warm_up", default=1, help="warm-up period", type=int)
    parser.add_argument("--delta", default=0.1, help="initial threshold", type=float)
    parser.add_argument("--inc", default=0.1, help="increment", type=float)
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
        exp_config[k] = v

    exp_config['naive_l1'] = []
    exp_config['naive_ece'] = []
    exp_config['ours_l1'] = []
    exp_config['ours_ece'] = []
    for seed in [77]:
        args.seed = seed
        naive_l1, ours_l1, naive_ece, ours_ece = main(args)
        exp_config['naive_l1'].append(naive_l1)
        exp_config['naive_ece'].append(naive_ece)
        exp_config['ours_l1'].append(ours_l1)
        exp_config['ours_ece'].append(ours_ece)

    dir_date = datetime.datetime.today().strftime("%Y%m%d")
    save_folder = os.path.join("./exp_logs/oursweighted_"+dir_date)
    os.makedirs(save_folder, exist_ok=True)
    save_file_name = 'oursweighted_' + datetime.date.today().strftime("%d_%m_%Y") + f"_{args.seed}_{args.dataset}_{args.noise_type}_{args.noise_strength}.json"
    save_file_name = os.path.join(save_folder, save_file_name)
    print(save_file_name)
    with open(save_file_name, "w") as f:
        json.dump(exp_config, f, sort_keys=False, indent=4)
    f.close()

