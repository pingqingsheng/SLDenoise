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

from data.MNIST import MNIST, MNIST_Combo
from data.CIFAR import CIFAR10, CIFAR10_Combo
from network.network import resnet18
from utils.utils import _init_fn, ECELoss, my_logits
from utils.noise import perturb_eta, noisify_with_P, noisify_mnist_asymmetric, noisify_cifar10_asymmetric
from baselines.temperature_scaling.temperature_scaling import ModelWithTemperature

# Experiment Setting Control Panel
# ---------------------------------------------------
N_EPOCH_OUTER: int = 1
N_EPOCH_INNER_CLS: int = 25
CONF_RECORD_EPOCH: int = N_EPOCH_INNER_CLS - 1
LR: float = 1e-3
WEIGHT_DECAY: float = 5e-3
BATCH_SIZE: int = 128
SCHEDULER_DECAY_MILESTONE: List = [5, 10, 15]
TRAIN_VALIDATION_RATIO: float = 0.8
MONITOR_WINDOW: int = 5
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
        trainset = MNIST(root="./data", split="train", train_ratio=TRAIN_VALIDATION_RATIO, download=True,transform=transform_train)
        validset = MNIST(root="./data", split="valid", train_ratio=TRAIN_VALIDATION_RATIO, transform=transform_train)
        testset = MNIST(root="./data", split="test", download=True, transform=transform_test)
        input_channel = 1
        num_classes = 10
        if args.noise_strength == 0.2:
            model_cls_clean = torch.load("./data/MNIST_resnet18_clean_20.pth")
        elif args.noise_strength == 0.4:
            model_cls_clean = torch.load("./data/MNIST_resnet18_clean_40.pth")
        elif args.noise_strength == 0.6:
            model_cls_clean = torch.load("./data/MNIST_resnet18_clean_60.pth")
        else:
            model_cls_clean = torch.load("./data/MNIST_resnet18_clean_80.pth")
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
        validset = CIFAR10(root="./data", split="valid", train_ratio=TRAIN_VALIDATION_RATIO, download=True, transform=transform_train)
        testset  = CIFAR10(root='./data', split="test", download=True, transform=transform_test)
        input_channel = 3
        num_classes = 10
        if args.noise_strength == 0.2:
            model_cls_clean = torch.load("./data/CIFAR10_resnet18_clean_20.pth")
        elif args.noise_strength == 0.4:
            model_cls_clean = torch.load("./data/CIFAR10_resnet18_clean_40.pth")
        elif args.noise_strength == 0.6:
            model_cls_clean = torch.load("./data/CIFAR10_resnet18_clean_60.pth")
        else:
            model_cls_clean = torch.load("./data/CIFAR10_resnet18_clean_80.pth")

    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, worker_init_fn=_init_fn(worker_id=seed))
    valid_loader = DataLoader(validset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, worker_init_fn=_init_fn(worker_id=seed))
    test_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, worker_init_fn=_init_fn(worker_id=seed))

    # Inject noise here
    y_train = copy.deepcopy(trainset.targets)
    y_valid = copy.deepcopy(validset.targets)
    y_test = testset.targets

    cprint(">>> Inject Noise <<<", "green")
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

    if args.noise_type=='linear':
        eta_tilde_train = perturb_eta(eta_train, args.noise_type, args.noise_strength)
        eta_tilde_valid = perturb_eta(eta_valid, args.noise_type, args.noise_strength)
        eta_tilde_test  = perturb_eta(eta_test, args.noise_type, args.noise_strength)
        y_tilde_train = [int(np.where(np.random.multinomial(1, x, 1).squeeze())[0]) for x in eta_tilde_train]
        y_tilde_valid = [int(np.where(np.random.multinomial(1, x, 1).squeeze())[0]) for x in eta_tilde_valid]
        trainset.update_labels(y_tilde_train)
    elif args.noise_type=='uniform':
        y_tilde_train, P, _ = noisify_with_P(np.array(copy.deepcopy(h_star_train)), nb_classes=10, noise=args.noise_strength)
        eta_tilde_train = np.matmul(F.one_hot(h_star_train, num_classes=10), P)
        eta_tilde_valid = np.matmul(F.one_hot(h_star_valid, num_classes=10), P)
        eta_tilde_test  = np.matmul(F.one_hot(h_star_test, num_classes=10), P)
        trainset.update_labels(y_tilde_train)
    elif args.noise_type=='asymmetric':
        if args.dataset == "mnist":
            y_tilde_train, P, _ = noisify_mnist_asymmetric(np.array(copy.deepcopy(h_star_train)), noise=args.noise_strength)
        elif args.dataset == 'cifar10':
            y_tilde_train, P, _ = noisify_cifar10_asymmetric(np.array(copy.deepcopy(h_star_train)),noise=args.noise_strength)
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

    model_cls = resnet18(num_classes=num_classes*2, in_channels=input_channel)
    model_cls = DataParallel(model_cls)
    model_cls = model_cls.to(device)

    optimizer_cls = torch.optim.Adam(model_cls.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler_cls = torch.optim.lr_scheduler.MultiStepLR(optimizer_cls, gamma=0.5, milestones=SCHEDULER_DECAY_MILESTONE)

    criterion_cls = torch.nn.CrossEntropyLoss()
    criterion_conf = torch.nn.MSELoss()
    criterion_calibrate = ECELoss()

    train_conf_delta = torch.zeros([len(trainset)])
    train_conf = torch.zeros([len(trainset), 10])
    valid_conf_delta = torch.zeros([len(validset)])
    valid_conf = torch.zeros([len(validset), 10])
    test_conf_delta = torch.zeros([len(testset)])
    test_conf = torch.zeros([len(testset), 10])
    test_conf_delta_pred = torch.zeros(len(testset))

    # moving average record for the network predictions
    f_record = torch.zeros([args.rollWindow, len(y_train), 10])  # MNIST: num_class=10
    current_delta = args.delta # for LRT

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
                    delta_prediction = F.one_hot(labels.squeeze(), num_classes=num_classes) - F.one_hot(predict.squeeze(), num_classes=num_classes).float()
                loss = criterion_cls(outs[:, :num_classes], labels) + criterion_conf(torch.sigmoid(outs[:, num_classes:]), delta_prediction)
                loss.backward()
                optimizer_cls.step()

                train_loss += loss.detach().cpu().item()
                train_correct += predict.eq(labels).sum().item()
                train_total += len(labels)

                # record the network predictions
                f_record[inner_epoch % args.rollWindow, indices] = F.softmax(outs.detach().cpu()[:, :num_classes], dim=1)

            train_acc = train_correct / train_total
            scheduler_cls.step()

            if not (inner_epoch + 1) % MONITOR_WINDOW:

                valid_correct = 0
                valid_total = 0
                ece_loss = 0
                model_cls.eval()
                for _, (indices, images, labels, _) in enumerate(tqdm(valid_loader, ascii=True, ncols=100)):
                    if images.shape[0] == 1:
                        continue
                    images, labels = images.to(device), labels.to(device)
                    outs = model_cls(images)

                    _, predict = outs[:, :num_classes].max(1)
                    correct_prediction = predict.eq(labels).float()
                    valid_correct += correct_prediction.sum().item()
                    valid_total += len(labels)

                    pred_onehot = F.one_hot(predict.detach().cpu(), num_classes=num_classes).float()
                    f_calibrate = torch.sigmoid(outs[:, num_classes:]).detach().cpu()+pred_onehot
                    reg_loss = criterion_conf(f_calibrate, torch.tensor(eta_tilde_valid[indices]))
                    ece_loss += criterion_calibrate.forward(logits=my_logits(f_calibrate, eps=1e-4), labels=torch.tensor(eta_tilde_valid[indices]).argmax(1).squeeze())

                valid_acc = valid_correct / valid_total
                print(f"Step [{inner_epoch + 1}|{N_EPOCH_INNER_CLS}] - Train Loss: {train_loss / train_total:7.3f} - Train Acc: {train_acc:7.3f} - Valid Acc: {valid_acc:7.3f} - Valid Reg Loss: {reg_loss:7.3f} - ECE Loss: {ece_loss.item():7.3f}")

            if inner_epoch == CONF_RECORD_EPOCH:
                # Epoch to record neural network's confidence
                test_correct = 0
                test_total = 0
                test_conf = torch.zeros(len(testset), num_classes)
                model_cls.eval()
                for _, (indices, images, labels, _) in enumerate(test_loader):
                    if images.shape[0] == 1:
                        continue
                    images, labels = images.to(device), labels.to(device)
                    outs = model_cls(images)
                    _, predict = outs[:, :num_classes].max(1)
                    correct_prediction = predict.eq(labels).float()
                    test_correct += correct_prediction.sum().item()
                    test_total += len(labels)

                    test_conf[indices, :] = torch.softmax(outs[:, :num_classes], 1).detach().cpu()

                    pred_onehot = F.one_hot(predict.detach().cpu(), num_classes=num_classes).float()
                    reg_loss = criterion_conf(torch.sigmoid(outs[:, num_classes:]).detach().cpu()+pred_onehot, torch.tensor(eta_tilde_valid[indices]))

                cprint(f"Classification Test Acc: {test_correct / test_total:7.3f} - Test Reg Loss: {reg_loss:7.3f}", "cyan")

        # Classification Final Test
        test_correct = 0
        test_total = 0
        test_predict = torch.zeros(len(testset), num_classes).float()
        test_conf_predict = torch.zeros(len(testset), num_classes).float()

        model_cls.eval()
        for _, (indices, images, labels, _) in enumerate(test_loader):
            if images.shape[0] == 1:
                continue
            images, labels = images.to(device), labels.to(device)
            outs = model_cls(images)
            _, predict = outs[:, :num_classes].max(1)
            test_correct += predict.eq(labels).sum().item()
            test_total += len(labels)

            test_predict[indices] = F.one_hot(predict.detach().cpu(), num_classes=num_classes).float()
            test_conf_predict[indices] = torch.sigmoid(outs[:, num_classes:]).detach().cpu()

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

    print("Test eta_tilde: ", eta_tilde_test[:5])
    print("Test test_confidence: ", test_conf[:5])
    print("Test test_conf_delta_pred: ", test_conf_predict[:5])
    print("MSE - Model Conf: ", criterion_conf(test_conf_predict+test_predict, torch.tensor(eta_tilde_test)))
    print("MSE - Ours: ", criterion_conf(test_conf_predict.squeeze(), torch.tensor(eta_tilde_test)))
    print("ECE - Model Conf: ", criterion_calibrate.forward(logits=my_logits(test_conf_predict.squeeze(), eps=1e-4), labels=torch.tensor(eta_tilde_test).argmax(1).squeeze()))
    print("ECE - Ours: ", criterion_calibrate.forward(logits=my_logits(test_conf_predict + test_predict, eps=1e-4), labels=torch.tensor(eta_tilde_test).argmax(1).squeeze()))
    print("Final Test Acc: ", test_correct/test_total*100, "%")


    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguement for SLDenoise")
    parser.add_argument("--seed", type=int, help="Random seed for the experiment", default=80)
    parser.add_argument("--gpus", type=str, help="Indices of GPUs to be used", default='0')
    parser.add_argument("--dataset", type=str, help="Experiment Dataset", default='mnist', choices={'mnist', 'cifar10', 'cifar100'})
    parser.add_argument("--noise_type", type=str, help="Noise type", default='linear', choices={"linear", "uniform", "asymmetric", "idl"})
    parser.add_argument("--noise_strength", type=float, help="Noise fraction", default=1)
    parser.add_argument("--rollWindow", default=3, help="rolling window to calculate the confidence", type=int)
    parser.add_argument("--warm_up", default=2, help="warm-up period", type=int)
    parser.add_argument("--delta", default=0.1, help="initial threshold", type=float)
    parser.add_argument("--inc", default=0.1, help="increment", type=float)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    main(args)