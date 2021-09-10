import os

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data as data
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR as MultiStepLR
import torchvision.transforms as transforms
from termcolor import cprint
import numpy as np
import argparse
from tqdm import tqdm
import copy
import logging
import time

from data_clothing1m import Clothing1M
from clothing_resnet import resnet50
from preact_resnet import preact_resnet18, preact_resnet34, preact_resnet101, initialize_weights, conv_init
from network import LogisticRegression
from utils import *

# @profile
def main(args):

    np.random.seed(123)

    log_out_dir = './logs/'
    time_stamp = time.strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(log_out_dir, 'log-' + time_stamp + '.txt')
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s',
                        filename=log_dir,
                        filemode='w')

    #opt_gpus = [i for i in range(args.gpu, (args.gpu + args.n_gpus))]
    #if len(args.gpus) > 1:
    print("Using GPUs:", args.gpus)
    #os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in opt_gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Parameters Setting
    batch_size: int = 24
    num_workers: int = 4
    # train_val_ratio: float = 0.9
    lr: float = 5e-4
    current_delta: float = 0.4
    flip_threshold = np.ones(args.nepochs) * 0.5
    initial_threshold = np.array([0.8, 0.8, 0.7, 0.6])
    flip_threshold[:len(initial_threshold)] = initial_threshold[:]

    # data augmentation
    transform_train = transforms.Compose([
        # transforms.RandomCrop(256, padding=4),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    transform_test = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    data_root = '/home/songzhu/PycharmProjects/cloth'
    trainset = Clothing1M(data_root=data_root, split='train', transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    valset = Clothing1M(data_root=data_root, split='val', transform=transform_test)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size * 4, shuffle=False, num_workers=num_workers)

    testset = Clothing1M(data_root=data_root, split='test', transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size * 4, shuffle=False, num_workers=num_workers)

    num_class = 14
    in_channel = 3

    if args.network == 'logistic':
        f = LogisticRegression(input_dim, num_class)
    elif args.network == 'preact_resnet34':
        f = preact_resnet34(num_classes=num_class, num_input_channels=in_channel)
        feature_size = 128
    elif args.network == 'resnet50':
        f = resnet50(num_classes=num_class, pretrained=True)
    f = nn.DataParallel(f)
    f.to(device)

    print("\n")
    print("============= Parameter Setting ================")
    print("Using Clothing1M dataset")
    print("Using Network Architecture : {}".format(args.network))
    print("Training Epoch : {} | Batch Size : {} | Learning Rate : {} ".format(args.nepochs, batch_size, lr))
    print("================================================")
    print("\n")

    print("============= Start Training =============")
    print("Start Label Correction at Epoch : {}".format(args.warm_up))

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(f.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=[6, 11], gamma=0.5)
    f_record = torch.zeros([args.rollWindow, len(trainset), num_class])

    test_acc = None
    best_val_acc = 0
    best_val_acc_epoch = 0
    best_test_acc = 0
    best_test_acc_epoch = 0
    best_weight = None

    for epoch in range(args.nepochs):
        train_loss = 0
        train_correct = 0
        train_total = 0

        f.train()
        # for _, (features, labels, indices) in enumerate(tqdm(train_loader, ascii=True, ncols=50)):
        for iteration, (features, labels, indices) in enumerate(tqdm(train_loader, ascii=True, ncols=50)):
            if features.shape[0] == 1:
                continue

            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = f(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_total += features.size(0)
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()

            f_record[epoch % args.rollWindow, indices] = F.softmax(outputs.detach().cpu(), dim=1)

            # ----------------------------------------------------------------------
            # Evaluation if necessary
            if iteration % args.eval_freq == 0:
                print("\n>> Validation <<")
                f.eval()
                test_loss = 0
                test_correct = 0
                test_total = 0

                for _, (features, labels, indices) in enumerate(val_loader):
                    if features.shape[0] == 1:
                        continue

                    features, labels = features.to(device), labels.to(device)
                    outputs = f(features)
                    loss = criterion(outputs, labels)

                    test_loss += loss.item()
                    test_total += features.size(0)
                    _, predicted = outputs.max(1)
                    test_correct += predicted.eq(labels).sum().item()

                val_acc = test_correct / test_total * 100
                cprint(">> [Epoch: {}] Val Acc: {:3.3f}%\n".format(epoch, val_acc), "blue")
                if best_val_acc < val_acc:
                    best_val_acc = val_acc
                    best_val_acc_epoch = epoch
                f.train()

            if iteration % args.eval_freq == 0:
                print("\n>> Testing <<")
                f.eval()
                test_loss = 0
                test_correct = 0
                test_total = 0

                for _, (features, labels, indices) in enumerate(test_loader):
                    if features.shape[0] == 1:
                        continue

                    features, labels = features.to(device), labels.to(device)
                    outputs = f(features)
                    loss = criterion(outputs, labels)

                    test_loss += loss.item()
                    test_total += features.size(0)
                    _, predicted = outputs.max(1)
                    test_correct += predicted.eq(labels).sum().item()

                test_acc = test_correct / test_total * 100
                cprint(">> [Epoch: {}] Test Acc: {:3.3f}%\n".format(epoch, test_acc), "yellow")
                if best_test_acc < test_acc:
                    best_test_acc = test_acc
                    best_test_acc_epoch = epoch
                    best_weight = copy.deepcopy(f.state_dict())

                    print(">> Saving best weight ...")
                    torch.save(best_weight, './checkpoints/best.pth')

                f.train()

                print(">> Best validation accuracy: {:3.3f}%, at epoch {}".format(best_val_acc, best_val_acc_epoch))
                print(">> Best testing accuracy: {:3.3f}%, at epoch {}".format(best_test_acc, best_test_acc_epoch))
            # ----------------------------------------------------------------------

        train_acc = train_correct / train_total * 100
        cprint("Epoch [{}|{}] \t Train Acc {:.3f}%".format(epoch+1, args.nepochs, train_acc), "yellow")
        cprint("Epoch [{}|{}] \t Best Val Acc {:.3f}% \t Best Test Acc {:.3f}%".format(epoch+1, args.nepochs, best_val_acc, best_test_acc), "yellow")
        scheduler.step()

        if epoch >= args.warm_up:
            f_x = f_record.mean(0)
            y_tilde = trainset.targets
            # y_corrected, current_delta = lrt_correction(y_tilde, f_x, epoch-args.warm_up,
            #                                                method=1, current_delta=current_delta)

            f_x_np = f_x.detach().cpu().numpy()
            # y_corrected = prob_correction(f_x_np, top_k=args.top_k, thd=flip_threshold[epoch])

            y_corrected, current_delta = prob_correction_v2(y_tilde, f_x, random_state=0, thd=0.1, current_delta=current_delta)

            logging.info('Current delta:\t{}\n'.format(current_delta))

            # y_corrected = y_corrected.tolist()

            trainset.update_corrupted_label(y_corrected)

    # Final testing
    f.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0

    for _, (features, labels, indices) in enumerate(test_loader):
        if features.shape[0] == 1:
            continue

        features, labels = features.to(device), labels.to(device)
        outputs = f(features)
        loss = criterion(outputs, labels)

        test_loss += loss.item()
        test_total += features.size(0)
        _, predicted = outputs.max(1)
        test_correct += predicted.eq(labels).sum().item()

    test_acc = test_correct / test_total * 100
    cprint(">> Test Acc: {:3.3f}%\n".format(test_acc), "yellow")
    if best_test_acc < test_acc:
        best_test_acc = test_acc
        best_test_acc_epoch = epoch
    print(">> Best validation accuracy: {:3.3f}%, at epoch {}".format(best_val_acc, best_val_acc_epoch))
    print(">> Best testing accuracy: {:3.3f}%, at epoch {}".format(best_test_acc, best_test_acc_epoch))

    # print("Final Test Accuracy {:3.3f}%".format(test_acc))
    # print("Final Recovered Ratio {:3.3f}%".format((n_corrected/float(n_corrupted))*100))
    # print("Final Delta Used {}".format(current_delta))

    # import matplotlib.pyplot as plt
    # import matplotlib.tri as tri
    # from MulticoreTSNE import MulticoreTSNE as TSNE
    # import umap
    # from scipy import interpolate
    #
    # cols = {0: "red", 1: "black"}
    # # reps_orig = TSNE(n_components=2, verbose=True, n_iter=1000, n_jobs=20, perplexity=50).fit_transform(X)
    # reps_orig = umap.UMAP(verbose=True).fit_transform(X)
    # classes = np.unique(Y)
    # plt.figure(figsize=(10, 10))
    # plt.subplot(2, 2, 1)
    # for i in range(len(classes)):
    #     plt.scatter(reps_orig[Y == i, 0], reps_orig[Y == i, 1], c=cols[i], label=classes[i], s=2)
    #     plt.title("Original Data")
    #     plt.legend()
    #
    # ax1 = plt.subplot(2, 2, 2)
    # xi = np.linspace(reps_orig[:, 0].min(), reps_orig[:, 0].max(), 200)
    # yi = np.linspace(reps_orig[:, 1].min(), reps_orig[:, 1].max(), 200)
    # triang = tri.Triangulation(reps_orig[:, 0], reps_orig[:, 1])
    # interpolator = tri.LinearTriInterpolator(triang, eta_1)
    # x_temp, y_temp = np.meshgrid(xi, yi)
    # eta_temp = interpolator(x_temp, y_temp)
    # # ax1.contour(x_temp, y_temp, eta_temp, colors="k", levels=4)
    # cntr1 = plt.contourf(x_temp, y_temp, eta_temp, cmap="RdBu_r", levels=4)
    # plt.colorbar(cntr1)
    #
    # plt.subplot(2, 2, 3)
    # for i in range(len(classes)):
    #     plt.scatter(reps_orig[:len(y_train_tilde)][y_train_tilde == i, 0],
    #                 reps_orig[:len(y_train_tilde)][y_train_tilde == i, 1], c=cols[i], label=classes[i], s=2)
    #     plt.title("Noisy Data")
    #     plt.legend()
    # y_final = trainset.target
    # plt.subplot(2,2,4)
    # for i in range(len(classes)):
    #     plt.scatter(reps_orig[:len(y_final)][y_final == i, 0],
    #                 reps_orig[:len(y_final)][y_final == i, 1], c=cols[i], label=classes[i], s=2)
    #     plt.title("Corrected Data")
    #     plt.legend()
    #
    # plt.savefig(args.dataset + "_" + str(args.noise_type) + ".png")

    return 0


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--noise_type", default=0, help="noise level", type=int)
    parser.add_argument("--network", default='preact_resnet34', help="network architecture", type=str)
    parser.add_argument("--nepochs", default=15, help="number of training epochs", type=int)
    parser.add_argument("--rollWindow", default=1, help="rolling window to calculate the confidence", type=int)
    parser.add_argument("--gpus", default='0', help="how many GPUs to use", type=str)
    parser.add_argument("--warm_up", default=1, help="warm-up period", type=int)
    parser.add_argument("--eval_freq", default=2000, help="evaluation frequency (every a few iterations)", type=int)
    parser.add_argument("--top_k", default=3, help="Flip to the top k categories", type=int)
    args = parser.parse_args()

    main(args)
