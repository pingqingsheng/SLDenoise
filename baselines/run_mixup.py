import os
import sys
sys.path.append("../")
from typing import List
import json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn import DataParallel
from torchvision import transforms
import numpy as np
from sklearn.mixture import GaussianMixture
import argparse
import random
from tqdm import tqdm
import copy
from termcolor import cprint
import datetime

from data.MNIST_MIXUP import MNIST_MIXUP
from data.CIFAR_MIXUP import CIFAR10_MIXUP
from baseline_network import resnet18
from utils.utils import _init_fn, ECELoss
from utils.noise import perturb_eta, noisify_with_P, noisify_mnist_asymmetric, noisify_cifar10_asymmetric

# Experiment Setting Control Panel
# ---------------------------------------------------
# Mix-up specific
ALPHA :float = 4
LAMBDA_U:float = 25
P_THRESH:float = 0.5
T:float = 0.5
# General set-up
TRAIN_VALIDATION_RATIO: float = 0.8
N_EPOCH_OUTER: int = 1
N_EPOCH_INNER_CLS: int = 200
CONF_RECORD_EPOCH: int = N_EPOCH_INNER_CLS - 1
LR: float = 5e-2
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
        trainset_warm = MNIST_MIXUP(root="../data", split="train", train_ratio=TRAIN_VALIDATION_RATIO, download=True,transform=transform_train, mode="warm")
        trainset_labeled = MNIST_MIXUP(root="../data", split="train", train_ratio=TRAIN_VALIDATION_RATIO, download=True,transform=transform_train, mode="train")
        trainset_unlabeled = MNIST_MIXUP(root="../data", split="train", train_ratio=TRAIN_VALIDATION_RATIO, download=True,transform=transform_train, mode="train")
        validset = MNIST_MIXUP(root="../data", split="valid", train_ratio=TRAIN_VALIDATION_RATIO, transform=transform_train, mode='eval')
        testset = MNIST_MIXUP(root="../data", split="test", download=True, transform=transform_test, mode='eval')
        INPUT_CHANNEL = 1
        NUM_CLASSES = 10
        INIT_EPOCH = 0
        if args.noise_type == 'idl':
            model_cls_clean = torch.load(f"../data/MNIST_resnet18_clean_{int(args.noise_strength*100)}.pth", map_location=DEVICE).module
        else:
            model_cls_clean = torch.load("../data/MNIST_resnet18_clean.pth", map_location=DEVICE).module
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
        trainset_warm = CIFAR10_MIXUP(root="../data", split="train", train_ratio=TRAIN_VALIDATION_RATIO, download=True, transform=transform_train, mode='warm')
        trainset_labeled = CIFAR10_MIXUP(root="../data", split="train", train_ratio=TRAIN_VALIDATION_RATIO, download=True, transform=transform_train, mode='train')
        trainset_unlabeled = CIFAR10_MIXUP(root="../data", split="train", train_ratio=TRAIN_VALIDATION_RATIO, download=True, transform=transform_train, mode='train')
        validset = CIFAR10_MIXUP(root="../data", split="valid", train_ratio=TRAIN_VALIDATION_RATIO, download=True, transform=transform_train, mode='eval')
        testset  = CIFAR10_MIXUP(root='../data', split="test", download=True, transform=transform_test, mode='eval')
        INPUT_CHANNEL = 3
        NUM_CLASSES = 10
        INIT_EPOCH = 20
        if args.noise_type == 'idl':
            model_cls_clean = torch.load(f"../data/CIFAR10_resnet18_clean_{int(args.noise_strength * 100)}.pth", map_location=DEVICE).module
        else:
            model_cls_clean = torch.load("../data/CIFAR10_resnet18_clean.pth", map_location=DEVICE).module

    # mixup-specific
    train_loader_warmup = DataLoader(copy.deepcopy(trainset_warm), batch_size=BATCH_SIZE, shuffle=True, num_workers=2, worker_init_fn=_init_fn(worker_id=seed))
    train_loader_labeled = DataLoader(copy.deepcopy(trainset_labeled), batch_size=BATCH_SIZE, shuffle=True, num_workers=2, worker_init_fn=_init_fn(worker_id=seed))
    train_loader_unlabeled = DataLoader(copy.deepcopy(trainset_unlabeled), batch_size=BATCH_SIZE, shuffle=True, num_workers=2, worker_init_fn=_init_fn(worker_id=seed))
    validset_noise = copy.deepcopy(validset)
    valid_loader = DataLoader(validset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, worker_init_fn=_init_fn(worker_id=seed))
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, worker_init_fn=_init_fn(worker_id=seed))
    valid_loader_noise = DataLoader(validset_noise, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, worker_init_fn=_init_fn(worker_id=seed))

    # Inject noise here
    y_train = copy.deepcopy(trainset_warm.targets)
    y_valid = copy.deepcopy(validset.targets)
    y_test = testset.targets

    cprint(">>> Inject Noise <<<", "green")
    _eta_train_temp_pair = [(torch.softmax(model_cls_clean(images.to(DEVICE)), 1).detach().cpu(), indices) for
                            _, (indices, images, labels, _) in enumerate(tqdm(train_loader_warmup, ascii=True, ncols=100))]
    _eta_valid_temp_pair = [(torch.softmax(model_cls_clean(images.to(DEVICE)), 1).detach().cpu(), indices) for
                            _, (indices, images, labels, _) in enumerate(tqdm(valid_loader, ascii=True, ncols=100))]
    _eta_test_temp_pair = [(torch.softmax(model_cls_clean(images.to(DEVICE)), 1).detach().cpu(), indices) for
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
        trainset_warm.update_labels(y_tilde_train)
        trainset_labeled.update_labels(y_tilde_train)
    elif args.noise_type=='uniform':
        y_tilde_train, P, _ = noisify_with_P(np.array(copy.deepcopy(h_star_train)), nb_classes=10, noise=args.noise_strength)
        y_tilde_valid, P, _ = noisify_with_P(np.array(copy.deepcopy(h_star_valid)), nb_classes=10, noise=args.noise_strength)
        eta_tilde_train = np.matmul(F.one_hot(h_star_train, num_classes=10), P)
        eta_tilde_valid = np.matmul(F.one_hot(h_star_valid, num_classes=10), P)
        eta_tilde_test  = np.matmul(F.one_hot(h_star_test, num_classes=10), P)
        trainset_warm.update_labels(y_tilde_train)
        trainset_labeled.update_labels(y_tilde_train)
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
        trainset_warm.update_labels(y_tilde_train)
        trainset_labeled.update_labels(y_tilde_train)
    elif args.noise_type=="idl":
        eta_tilde_train = copy.deepcopy(eta_train)
        eta_tilde_valid = copy.deepcopy(eta_valid)
        eta_tilde_test  = copy.deepcopy(eta_test)
        y_tilde_train = [int(np.where(np.random.multinomial(1, x, 1).squeeze())[0]) for x in eta_tilde_train]
        y_tilde_valid = [int(np.where(np.random.multinomial(1, x, 1).squeeze())[0]) for x in eta_tilde_valid]
        trainset_warm.update_labels(y_tilde_train)
        trainset_labeled.update_labels(y_tilde_train)

    validset_noise = copy.deepcopy(validset)
    validset.update_labels(h_star_valid)
    testset.update_labels(h_star_test)
    validset_noise.update_labels(y_tilde_valid)
    train_noise_ind = np.where(np.array(h_star_train) != np.array(y_tilde_train))[0]

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
    print(f"Number of Training Data Points: \t\t {len(trainset_warm)}")
    print(f"Number of Validation Data Points: \t\t {len(validset)}")
    print(f"Number of Testing Data Points: \t\t\t {len(testset)}")
    print(f"Cls Confidence Recording Epoch: \t\t {CONF_RECORD_EPOCH}")
    print(f"Train - Validation Ratio: \t\t\t {TRAIN_VALIDATION_RATIO}")
    print(f"Clean Training Data Points: \t\t\t {len(trainset_warm) - len(train_noise_ind)}")
    print(f"Noisy Training Data Points: \t\t\t {len(train_noise_ind)}")
    print(f"Noisy Type: \t\t\t\t\t {args.noise_type}")
    print(f"Noisy Strength: \t\t\t\t {args.noise_strength}")
    print(f"Noisy Level: \t\t\t\t\t {len(train_noise_ind) / len(trainset_warm) * 100:.2f}%")
    print("---------------------------------------------------------")

    model_cls_1 = resnet18(num_classes=NUM_CLASSES, in_channels=INPUT_CHANNEL)
    model_cls_1 = DataParallel(model_cls_1)
    model_cls_2 = resnet18(num_classes=NUM_CLASSES, in_channels=INPUT_CHANNEL)
    model_cls_2 = DataParallel(model_cls_2)
    model_cls_1, model_cls_2 = model_cls_1.to(DEVICE), model_cls_2.to(DEVICE)

    # TODO: add calibration model here
    # model_cls_cali = resnet18(num_classes=NUM_CLASSES, in_channels=INPUT_CHANNEL)
    # model_cls_cali = DataParallel(model_cls_cali)
    # model_cls_cali = model_cls_cali.to(DEVICE)

    optimizer_1 = torch.optim.SGD(model_cls_1.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, momentum=0.9, nesterov=True)
    scheduler_1 = torch.optim.lr_scheduler.MultiStepLR(optimizer_1, gamma=0.5, milestones=SCHEDULER_DECAY_MILESTONE)
    optimizer_2 = torch.optim.SGD(model_cls_2.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, momentum=0.9, nesterov=True)
    scheduler_2 = torch.optim.lr_scheduler.MultiStepLR(optimizer_2, gamma=0.5, milestones=SCHEDULER_DECAY_MILESTONE)

    criterion_ce = torch.nn.CrossEntropyLoss()
    criterion_mixmatch = SemiLoss()
    criterion_calibrate = ECELoss()
    criterion_l1 = torch.nn.L1Loss()
    criterion_asy = NegEntropy()

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
    _best_ts_l1 = np.inf
    _best_ts_ece = np.inf
    loss_1_record = torch.zeros(len(trainset_warm))
    loss_2_record = torch.zeros(len(trainset_warm))
    gmm = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)

    for epoch in range(N_EPOCH_INNER_CLS):

        # Warm-up and record loss
        cprint(f">>>Epoch [{epoch + 1}|{N_EPOCH_INNER_CLS}] Training <<<", "green")
        model_cls_1.train()
        model_cls_2.train()
        for ib, (indices, images, labels, _) in enumerate(tqdm(train_loader_warmup, ascii=True, ncols=100)):
            if images.shape[0] == 1:
                continue
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outs_1 = model_cls_1(images)
            outs_2 = model_cls_2(images)
            loss_1 = criterion_ce(outs_1, labels)
            loss_2 = criterion_ce(outs_2, labels)
            penalty_1 = criterion_asy(outs_1) if args.noise_type == 'asymmetric' else 0
            penalty_2 = criterion_asy(outs_2) if args.noise_type == 'asymmetric' else 0
            loss_1_record[indices] = loss_1.detach().cpu() + penalty_1
            loss_2_record[indices] = loss_2.detach().cpu() + penalty_2
            loss = loss_1+loss_2
            if epoch < args.warm_up:
                optimizer_1.zero_grad()
                optimizer_2.zero_grad()
                loss_1.backward()
                loss_2.backward()
                optimizer_1.step()
                optimizer_2.step()

        # Select data
        loss_1_record = (loss_1_record - min(loss_1_record))/(max(loss_1_record)-min(loss_1_record))
        loss_2_record = (loss_2_record - min(loss_2_record))/(max(loss_2_record)-min(loss_2_record))
        gmm.fit(loss_1_record.unsqueeze(1).numpy())
        prob_1 = gmm.predict_proba(loss_1_record.unsqueeze(1).numpy())
        prob_1 = prob_1[:, gmm.means_.argmin()]
        gmm.fit(loss_2_record.unsqueeze(1).numpy())
        prob_2 = gmm.predict_proba(loss_2_record.unsqueeze(1).numpy())
        prob_2 = prob_2[:, gmm.means_.argmin()]
        pred_1 = (prob_1 > P_THRESH)
        pred_2 = (prob_2 > P_THRESH)

        # For monitoring purpose
        label_correctness = (torch.tensor(y_tilde_train).squeeze() == torch.tensor(h_star_train).squeeze()).long()
        ind1 = torch.from_numpy(pred_1).squeeze()
        ind2 = torch.from_numpy(pred_2).squeeze()
        print(f"Net1 selection total {sum(pred_1)} - precision {sum(label_correctness[ind1])/ind1.sum():.3f} - recall {sum(label_correctness[ind1])/label_correctness.sum():.3f}")
        print(f"Net2 selection total {sum(pred_2)} - precision {sum(label_correctness[ind2])/ind2.sum():.3f} - recall {sum(label_correctness[ind2])/label_correctness.sum():.3f}")

        if epoch > args.warm_up:
            # Training model 1 with divide and mix
            trainset_labeled.select_data(np.where(pred_2)[0].flatten())
            trainset_unlabeled.select_data(np.where(1-pred_2)[0].flatten())
            train_loader_unlabeled_iter = iter(train_loader_unlabeled)
            num_iter = len(train_loader_labeled.dataset)//BATCH_SIZE+1
            model_cls_1.train()
            model_cls_2.eval()
            for ib, (indices_x, images_x1, images_x2, labels, _) in enumerate(tqdm(train_loader_labeled, ascii=True, ncols=100)):
                if images.shape[0]==1:
                    continue
                try:
                    indices_u, images_u1, images_u2, _, _ = train_loader_unlabeled_iter.next()
                except:
                    # If unlabeled set runs out of data then reset the loader
                    train_loader_unlabeled_iter = iter(DataLoader(trainset_unlabeled, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, worker_init_fn=_init_fn(worker_id=seed)))
                    indices_u, images_u1, images_u2, _, _ = train_loader_unlabeled_iter.next()

                common_num = min(images_x1.shape[0], images_u1.shape[0])
                labels_onehot = F.one_hot(labels[:common_num], NUM_CLASSES)
                w_x = torch.from_numpy(prob_2[indices_x])[:common_num]
                images_x1, images_x2 = images_x1.to(DEVICE)[:common_num], images_x2.to(DEVICE)[:common_num]
                images_u1, images_u2 = images_u1.to(DEVICE)[:common_num], images_u2.to(DEVICE)[:common_num]
                labels_onehot = labels_onehot.to(DEVICE)[:common_num]
                w_x  = w_x.to(DEVICE)

                with torch.no_grad():
                    # label co-guessing of unlabeled samples
                    outputs_u11 = model_cls_1(images_u1)
                    outputs_u12 = model_cls_1(images_u2)
                    outputs_u21 = model_cls_2(images_u1)
                    outputs_u22 = model_cls_2(images_u2)
                    pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) +
                          torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1))/4
                    ptu = pu ** (1 / T)  # temparature sharpening
                    targets_u = ptu / ptu.sum(dim=1, keepdim=True)  # normalize
                    targets_u = targets_u.detach()
                    # label refinement of labeled samples
                    outputs_x1 = model_cls_1(images_x1)
                    outputs_x2 = model_cls_1(images_x2)
                    px = (torch.softmax(outputs_x1, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
                    px = w_x.unsqueeze(1) * labels_onehot + (1 - w_x.unsqueeze(1)) * px
                    ptx = px ** (1 / T)  # temparature sharpening
                    targets_x = ptx / ptx.sum(dim=1, keepdim=True)  # normalize
                    targets_x = targets_x.detach()

                # mixmatch
                l = np.random.beta(ALPHA, ALPHA)
                l = max(l, 1 - l)
                all_inputs = torch.cat([images_x1, images_x2, images_u1, images_u2], dim=0)
                all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)
                idx = torch.randperm(all_inputs.size(0))
                input_a, input_b = all_inputs, all_inputs[idx]
                target_a, target_b = all_targets, all_targets[idx]
                mixed_input = l * input_a + (1 - l) * input_b
                mixed_target = l * target_a + (1 - l) * target_b
                logits = model_cls_1(mixed_input)
                logits_x = logits[:common_num*2]
                logits_u = logits[common_num*2:]

                Lx, Lu, lamb = criterion_mixmatch(logits_x,
                                                  mixed_target[:common_num * 2],
                                                  logits_u, mixed_target[common_num * 2:],
                                                  epoch + ib/num_iter,
                                                  args.warm_up)
                # regularization
                prior = torch.ones(NUM_CLASSES)/NUM_CLASSES
                prior = prior.cuda()
                pred_mean = torch.softmax(logits, dim=1).mean(0)
                penalty = torch.sum(prior*torch.log(prior/pred_mean))
                loss = Lx + lamb * Lu  + penalty
                # compute gradient and do SGD step
                optimizer_1.zero_grad()
                loss.backward()
                optimizer_1.step()

            scheduler_1.step()

            # Training model 2 with divide and mix
            trainset_labeled.select_data(np.where(pred_1)[0].flatten())
            trainset_unlabeled.select_data(np.where(1-pred_1)[0].flatten())
            train_loader_unlabeled_iter = iter(train_loader_unlabeled)
            num_iter = len(train_loader_labeled.dataset)//BATCH_SIZE+1
            model_cls_1.eval()
            model_cls_2.train()
            for ib, (indices_x, images_x1, images_x2, labels, _) in enumerate(tqdm(train_loader_labeled, ascii=True, ncols=100)):
                if images.shape[0]==1:
                    continue
                try:
                    indices_u, images_u1, images_u2, _, _ = train_loader_unlabeled_iter.next()
                except:
                    # If unlabeled set runs out of data then reset the loader
                    train_loader_unlabeled_iter = iter(DataLoader(trainset_unlabeled, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, worker_init_fn=_init_fn(worker_id=seed)))
                    indices_u, images_u1, images_u2, _, _ = train_loader_unlabeled_iter.next()

                common_num = min(images_x1.shape[0], images_u1.shape[0])
                labels_onehot = F.one_hot(labels[:common_num], NUM_CLASSES)
                w_x = torch.from_numpy(prob_2[indices_x])[:common_num]
                images_x1, images_x2 = images_x1.to(DEVICE)[:common_num], images_x2.to(DEVICE)[:common_num]
                images_u1, images_u2 = images_u1.to(DEVICE)[:common_num], images_u2.to(DEVICE)[:common_num]
                labels_onehot = labels_onehot.to(DEVICE)
                w_x  = w_x.to(DEVICE)

                with torch.no_grad():
                    # label co-guessing of unlabeled samples
                    outputs_u11 = model_cls_1(images_u1)
                    outputs_u12 = model_cls_1(images_u2)
                    outputs_u21 = model_cls_2(images_u1)
                    outputs_u22 = model_cls_2(images_u2)
                    pu = (torch.softmax(outputs_u11, dim=1) + torch.softmax(outputs_u12, dim=1) +
                          torch.softmax(outputs_u21, dim=1) + torch.softmax(outputs_u22, dim=1))/4
                    ptu = pu ** (1 / T)  # temparature sharpening
                    targets_u = ptu / ptu.sum(dim=1, keepdim=True)  # normalize
                    targets_u = targets_u.detach()
                    # label refinement of labeled samples
                    outputs_x1 = model_cls_2(images_x1)
                    outputs_x2 = model_cls_2(images_x2)
                    px = (torch.softmax(outputs_x1, dim=1) + torch.softmax(outputs_x2, dim=1)) / 2
                    px = w_x.unsqueeze(1) * labels_onehot + (1 - w_x.unsqueeze(1)) * px
                    ptx = px ** (1 / T)  # temparature sharpening
                    targets_x = ptx / ptx.sum(dim=1, keepdim=True)  # normalize
                    targets_x = targets_x.detach()

                # mixmatch
                l = np.random.beta(ALPHA, ALPHA)
                l = max(l, 1 - l)
                all_inputs = torch.cat([images_x1, images_x2, images_u1, images_u2], dim=0)
                all_targets = torch.cat([targets_x, targets_x, targets_u, targets_u], dim=0)
                idx = torch.randperm(all_inputs.size(0))
                input_a, input_b = all_inputs, all_inputs[idx]
                target_a, target_b = all_targets, all_targets[idx]
                mixed_input = l * input_a + (1 - l) * input_b
                mixed_target = l * target_a + (1 - l) * target_b
                logits = model_cls_2(mixed_input)
                logits_x = logits[:common_num*2]
                logits_u = logits[common_num*2:]

                Lx, Lu, lamb = criterion_mixmatch(logits_x,
                                                  mixed_target[:common_num * 2],
                                                  logits_u, mixed_target[common_num * 2:],
                                                  epoch + ib/num_iter,
                                                  args.warm_up)
                # regularization
                prior = torch.ones(NUM_CLASSES)/NUM_CLASSES
                prior = prior.cuda()
                pred_mean = torch.softmax(logits, dim=1).mean(0)
                penalty = torch.sum(prior*torch.log(prior/pred_mean))
                loss = Lx + lamb * Lu  + penalty
                # compute gradient and do SGD step
                optimizer_2.zero_grad()
                loss.backward()
                optimizer_2.step()
            scheduler_2.step()

        # Validation and monitoring
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

            model_cls_1.eval()
            model_cls_2.eval()
            for _, (indices, images, labels, _) in enumerate(tqdm(valid_loader, ascii=True, ncols=100)):
                if images.shape[0] == 1:
                    continue
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outs_1 = model_cls_1(images)
                outs_2 = model_cls_2(images)
                # Raw model result record
                outs = outs_1+outs_2
                prob_outs = torch.softmax(outs, 1)
                _, predict = prob_outs.max(1)
                correct_prediction = predict.eq(labels).float()
                valid_correct_raw += correct_prediction.sum().item()
                valid_total += len(labels)
                valid_f_raw[indices] = prob_outs.detach().cpu()
                valid_f_raw_target_conf[indices] = prob_outs.max(1)[0].detach().cpu()

                # Calibrated model result record
                # prob_outs = torch.softmax(outs_cali, 1)
                # _, predict = prob_outs.max(1)
                # correct_prediction = predict.eq(labels).float()
                # valid_correct_cali += correct_prediction.sum().item()
                # valid_f_cali[indices] = prob_outs.detach().cpu()
                # valid_f_cali_target_conf[indices] = prob_outs.max(1)[0].detach().cpu()

            valid_acc_raw = valid_correct_raw/valid_total
            valid_acc_cali = valid_correct_cali/valid_total
            ece_loss_raw = criterion_calibrate.forward(logits=valid_f_raw, labels=torch.tensor(eta_tilde_valid).argmax(1).squeeze())
            #ece_loss_cali = criterion_calibrate.forward(logits=valid_f_cali, labels=torch.tensor(eta_tilde_valid).argmax(1).squeeze())
            l1_loss_raw = criterion_l1(valid_f_raw_target_conf, torch.tensor(eta_tilde_valid).max(1)[0])
            #l1_loss_cali = criterion_l1(valid_f_cali_target_conf, torch.tensor(eta_tilde_valid).max(1)[0])
            # print(f"Step [{epoch + 1}|{N_EPOCH_INNER_CLS}] - Train Loss: {train_loss/train_total:7.3f} - Train Acc: {train_acc:7.3f} - Valid Acc Raw: {valid_acc_raw:7.3f} - Valid Acc Cali: {valid_acc_cali:7.3f}")
            # print(f"ECE Raw: {ece_loss_raw.item()} - ECE Cali: {ece_loss_cali.item()} - L1 Loss: {l1_loss_raw.item()} - L1 Loss: {l1_loss_cali.item()}")

            # For monitoring purpose
            _loss_record[_count] = float(loss.item()/len(trainset_warm))
            _valid_acc_raw_record[_count] = float(valid_acc_raw)
            # _valid_acc_cali_record[_count] = float(valid_acc_cali)
            _ece_raw_record[_count] = float(ece_loss_raw)
            # _ece_cali_record[_count] = float(ece_loss_cali.item())
            _l1_raw_record[_count] = float(l1_loss_raw)
            # _l1_cali_record[_count] = float(l1_loss_cali.item())
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

            model_cls_1.eval()
            model_cls_2.eval()
            for _, (indices, images, labels, _) in enumerate(tqdm(test_loader, ascii=True, ncols=100)):
                if images.shape[0] == 1:
                    continue
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outs_1 = model_cls_1(images)
                outs_2 = model_cls_2(images)

                # Raw model result record
                outs = outs_1 + outs_2
                prob_outs = torch.softmax(outs, 1)
                _, predict = prob_outs.max(1)
                correct_prediction = predict.eq(labels).float()
                test_correct_raw += correct_prediction.sum().item()
                test_total += len(labels)
                test_f_raw[indices] = prob_outs.detach().cpu()
                test_f_raw_target_conf[indices] = prob_outs.max(1)[0].detach().cpu()

                # Calibrated model result record
                # prob_outs = torch.softmax(outs_cali, 1)
                # _, predict = prob_outs.max(1)
                # correct_prediction = predict.eq(labels).float()
                # test_correct_cali += correct_prediction.sum().item()
                # test_f_cali[indices] = prob_outs.detach().cpu()
                # test_f_cali_target_conf[indices] = prob_outs.max(1)[0].detach().cpu()

            test_acc_raw = test_correct_raw/test_total
            # test_acc_cali = test_correct_cali/test_total
            ece_loss_raw = criterion_calibrate.forward(logits=test_f_raw, labels=torch.tensor(eta_tilde_test).argmax(1).squeeze())
            # ece_loss_cali = criterion_calibrate.forward(logits=test_f_cali, labels=torch.tensor(eta_tilde_test).argmax(1).squeeze())
            l1_loss_raw = criterion_l1(test_f_raw_target_conf, torch.tensor(eta_tilde_test).max(1)[0])
            # l1_loss_cali = criterion_l1(test_f_cali_target_conf, torch.tensor(eta_tilde_test).max(1)[0])

            if  ece_loss_raw < _best_raw_ece:
                _best_raw_ece = ece_loss_raw
                _best_raw_ece_epoch = epoch
            if l1_loss_raw < _best_raw_l1:
                _best_raw_l1 = l1_loss_raw
                _best_raw_l1_epoch = epoch
            # if ece_loss_cali < _best_ts_ece:
            #     _best_ts_ece = ece_loss_cali
            #     _best_ts_ece_epoch = epoch
            # if l1_loss_cali < _best_ts_l1:
            #     _best_ts_l1 = l1_loss_cali
            #     _best_ts_l1_epoch = epoch

            # print(f"Step [{epoch + 1}|{N_EPOCH_INNER_CLS}] - Train Loss: {train_loss/train_total:7.3f} - Train Acc: {train_acc:7.3f} - Valid Acc Raw: {test_acc_raw:7.3f} - Valid Acc Cali: {test_acc_cali:7.3f}")
            # print(f"ECE Raw: {ece_loss_raw.item():.3f}/{_best_raw_ece.item():.3f} - ECE Cali: {ece_loss_cali.item():.3f}/{_best_ts_ece.item():.3f} - Best ECE Raw Epoch: {_best_raw_ece_epoch} - Best ECE Cali Epoch: {_best_ts_ece_epoch}")
            # print(f"L1 Raw: {l1_loss_raw.item():.3f}/{_best_raw_l1.item():.3f} - L1 Cali: {l1_loss_cali.item():.3f}/{_best_ts_l1.item():.3f} - Best L1 Raw: {_best_raw_l1_epoch} - Best L1 Cali: {_best_ts_l1_epoch}")
            print(f"Step [{epoch + 1}|{N_EPOCH_INNER_CLS}] - Train Loss: {loss.item():.3f} - Valid Acc Raw: {valid_acc_raw:.3f} - Test Acc Raw: {test_acc_raw:.3f}")
            print(f"ECE Raw: {ece_loss_raw.item():.3f}/{_best_raw_ece.item():.3f} - Best ECE Raw Epoch: {_best_raw_ece_epoch}")
            print(f"L1 Raw: {l1_loss_raw.item():.3f}/{_best_raw_l1.item():.3f} - Best L1 Epoch: {_best_raw_l1_epoch}")


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
        plt.plot(x_axis[:-1], _valid_acc_cali_record[:-1], linewidth=2, label="MixUp")
        plt.title('Acc Curve')
        plt.subplot(2, 2, 3)
        plt.plot(x_axis[:-1], _ece_raw_record[:-1], linewidth=2, label="Raw")
        plt.plot(x_axis[:-1], _ece_cali_record[:-1], linewidth=2, label="MixUp")
        plt.legend()
        plt.title('ECE Curve')
        plt.subplot(2, 2, 4)
        plt.plot(x_axis[:-1], _l1_raw_record[:-1], linewidth=2, label="Raw")
        plt.plot(x_axis[:-1], _l1_cali_record[:-1], linewidth=2, label="MixUp")
        plt.legend()
        plt.title('L1 Curve')

        time_stamp = datetime.datetime.strftime(datetime.datetime.today(), "%Y-%m-%d")
        fig.savefig(os.path.join("./figures", f"exp_log_{args.dataset}_{args.noise_type}_{args.noise_strength}_MixUp_plot_{time_stamp}.png"))

    return _best_raw_l1.item(), _best_raw_ece.item()
    # return l1_loss_raw, ece_loss_raw


def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return LAMBDA_U*float(current)


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, warm_up):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, linear_rampup(epoch,warm_up)

class NegEntropy(object):
    def __call__(self,outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log()*probs, dim=1))


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

    exp_config['mixup_l1'] = []
    exp_config['mixup_ece'] = []
    for seed in [77]:
        args.seed = seed
        ours_l1,  ours_ece = main(args)
        exp_config['mixup_l1'].append(ours_l1)
        exp_config['mixup_ece'].append(ours_ece)

    dir_date = datetime.datetime.today().strftime("%Y%m%d")
    save_folder = os.path.join("../exp_logs/mixup_"+dir_date)
    os.makedirs(save_folder, exist_ok=True)
    print(save_folder)
    save_file_name = 'mixup_' + datetime.date.today().strftime("%d_%m_%Y") + f"_{args.seed}_{args.dataset}_{args.noise_type}_{args.noise_strength}.json"
    save_file_name = os.path.join(save_folder, save_file_name)
    with open(save_file_name, "w") as f:
        json.dump(exp_config, f, sort_keys=False, indent=4)
    f.close()








