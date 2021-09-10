import os

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data as data
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import MultiStepLR as MultiStepLR
from termcolor import cprint
import numpy as np
import argparse
from tqdm import tqdm
import copy
import logging
import time
import pickle as pkl

from data.data_clothing1m import Clothing1M, Clothing1M_confidence
from network.clothing_resnet import resnet50
from network.preact_resnet import preact_resnet34


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

    print("Using GPUs:", args.gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Parameters Setting
    batch_size: int = 96
    num_workers: int = 2
    # train_val_ratio: float = 0.9
    lr: float = 2e-4
    weight_decay: float = 5e-4

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

    data_root = '/data/songzhu/cloth/cloth'
    trainset = Clothing1M(data_root=data_root, split='train', transform=transform_train)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)

    valset = Clothing1M(data_root=data_root, split='val', transform=transform_test)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    testset = Clothing1M(data_root=data_root, split='test', transform=transform_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    num_class = 14
    in_channel = 3

    if args.network == 'preact_resnet34':
        model = preact_resnet34(num_classes=num_class, num_input_channels=in_channel)
        model_confidence = preact_resnet34(num_classes=num_class, in_channel=in_channel)
    elif args.network == 'resnet50':
        model = resnet50(num_classes=num_class, pretrained=True)
        model_confidence = resnet50(num_classes=num_class, pretrained=True)
    model = nn.DataParallel(model)
    model_confidence = nn.DataParallel(model_confidence)
    model = model.to(device)
    model_confidence = model_confidence.to(device)

    print("\n")
    print("============= Parameter Setting ================")
    print("Using Clothing1M dataset")
    print("Using Network Architecture : {}".format(args.network))
    print("Training Epoch : {} | Batch Size : {} | Learning Rate : {} ".format(args.nepochs, batch_size, lr))
    print("================================================")
    print("\n")

    print("============= Fitting Classifier =============")
    criterion = torch.nn.CrossEntropyLoss()
    criterion_confidence = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
    optimizer_confidence = torch.optim.SGD(model_confidence.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=[5, 10, 15], gamma=0.5)
    scheduler_confidence = MultiStepLR(optimizer_confidence, milestones=[5, 10, 15], gamma=0.5)

    if os.path.exists('./checkpoints/best_result.pkl'):
        best_result = pkl.load(open('./checkpoints/best_result.pkl', "rb"))
        best_model = best_result['best_model']
        f_confidence_delta = best_result['confidence_delta']
        test_confidence_delta = best_result['test_confidence_delta']
    else:
        test_acc = None
        best_val_acc = 0
        best_val_acc_epoch = 0
        best_test_acc = 0
        best_test_acc_epoch = 0
        best_weight = None
        f_confidence = torch.zeros(len(trainset), num_class) # save for further usage
        f_confidence_delta = torch.zeros(len(trainset))
        f_prediction = torch.zeros(len(trainset))
        f_confidence_val = torch.zeros(len(valset), num_class)
        f_confidence_delta_val = torch.zeros(len(valset))
        f_prediction_val = torch.zeros(len(valset))

        for epoch in range(args.nepochs):
            train_loss = 0
            train_correct = 0
            train_total = 0


            model.train()
            for iteration, (features, labels, indices) in enumerate(tqdm(train_loader, ascii=True, ncols=50)):
                if features.shape[0] == 1:
                    continue

                features, labels = features.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_total += features.size(0)
                _, predicted = outputs.max(1)
                train_correct += predicted.eq(labels).sum().item()

                f_confidence[indices] = F.softmax(outputs.detach().cpu(), dim=1).float()
                f_confidence_delta[indices] = torch.abs(predicted - labels).detach().cpu().float()
                f_prediction[indices] = predicted.detach().cpu().float()
                # ----------------------------------------------------------------------
                # Evaluation if necessary
                if (iteration % args.eval_freq == 0) and (iteration > 0):
                    print("\n>> Validation <<")
                    model.eval()
                    test_loss = 0
                    test_correct = 0
                    test_total = 0

                    for _, (features, labels, indices) in enumerate(val_loader):
                        if features.shape[0] == 1:
                            continue

                        features, labels = features.to(device), labels.to(device)
                        outputs = model(features)
                        loss = criterion(outputs, labels)

                        test_loss += loss.item()
                        test_total += features.size(0)
                        _, predicted = outputs.max(1)
                        test_correct += predicted.eq(labels).sum().item()

                        f_confidence_val[indices] = F.softmax(outputs.detach().cpu(), dim=1).float()
                        f_confidence_delta_val[indices] = torch.abs(predicted - labels).detach().cpu().float()
                        f_prediction_val[indices] = predicted.detach().cpu().float()

                    val_acc = test_correct / test_total * 100
                    cprint(">> [Epoch: {}] Val Acc: {:3.3f}%\n".format(epoch, val_acc), "blue")
                    if best_val_acc < val_acc:
                        best_val_acc = val_acc
                        best_val_acc_epoch = epoch
                    model.train() # reset training status

                if (iteration % args.eval_freq == 0) and (iteration > 0):
                    print("\n>> Testing <<")
                    model.eval()
                    test_loss = 0
                    test_correct = 0
                    test_total = 0

                    for _, (features, labels, indices) in enumerate(test_loader):
                        if features.shape[0] == 1:
                            continue

                        features, labels = features.to(device), labels.to(device)
                        outputs = model(features)
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
                        best_weight = copy.deepcopy(model.state_dict())
                        best_confidence = f_confidence
                        print(">> Saving best weight ...")
                        torch.save(best_weight, './checkpoints/best.pth')
                    model.train() # reset training status

                    print(">> Best validation accuracy: {:3.3f}%, at epoch {}".format(best_val_acc, best_val_acc_epoch))
                    print(">> Best testing accuracy: {:3.3f}%, at epoch {}".format(best_test_acc, best_test_acc_epoch))
                # ----------------------------------------------------------------------

            train_acc = train_correct / train_total * 100
            cprint("Epoch [{}|{}] \t Train Acc {:.3f}%".format(epoch+1, args.nepochs, train_acc), "yellow")
            cprint("Epoch [{}|{}] \t Best Val Acc {:.3f}% \t Best Test Acc {:.3f}%".format(epoch+1, args.nepochs, best_val_acc, best_test_acc), "yellow")
            scheduler.step()

        # Final testing
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        test_confidence = torch.zeros(len(testset), num_class)
        test_confidence_delta = torch.zeros(len(testset))
        test_prediction = torch.zeros(len(testset))

        for _, (features, labels, indices) in enumerate(test_loader):
            if features.shape[0] == 1:
                continue

            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            test_total += features.size(0)
            _, predicted = outputs.max(1)
            test_correct += predicted.eq(labels).sum().item()

            test_confidence[indices] = F.softmax(outputs.detach().cpu(), dim=1).float()
            test_confidence_delta[indices] = torch.abs(predicted - labels).detach().cpu().float()
            test_prediction[indices] = predicted.detach().cpu().float()

        test_acc = test_correct / test_total * 100
        cprint(">> Test Acc: {:3.3f}%\n".format(test_acc), "yellow")
        if best_test_acc < test_acc:
            best_test_acc = test_acc
            best_test_acc_epoch = epoch
            best_model = model

            best_result = {}
            best_result['best_model'] = model
            best_result['confidence_delta'] = f_confidence_delta
            best_result['test_confidence_delta'] = test_confidence_delta

            best_result_file = f"./checkpoints/best_result_{time_stamp}.pkl"
            with open(best_result_file, 'wb') as f:
                pkl.dump(best_result, f)
            f.close()

        print(">> Best validation accuracy: {:3.3f}%, at epoch {}".format(best_val_acc, best_val_acc_epoch))
        print(">> Best testing accuracy: {:3.3f}%, at epoch {}".format(best_test_acc, best_test_acc_epoch))

    # >>> Collect training confidence and fit confidence model <<<
    print("============= Fitting Confidence Model =============")
    trainset_confidence = Clothing1M_confidence(data_root=data_root, labels=f_confidence_delta, transform=transform_train)
    testset_confidence = Clothing1M_confidence(data_root=data_root, labels=test_confidence_delta, transform=transform_test)
    trainloader_confidence = torch.utils.data.DataLoader(trainset_confidence, batch_size=64, shuffle=True, pin_memory=True)
    testloader_confidence = torch.utils.data.DataLoader(testset_confidence, batch_size=64, shuffle=True, pin_memory=True)

    model_confidence.train()
    for epoch in range(args.nepochs):
        model_confidence.train()
        for iteration, (features, labels, indices) in enumerate(tqdm(trainloader_confidence, ascii=True, ncols=50)):

            if features.shape[0] == 1:
                continue

            features, labels = features.to(device), labels.to(device)
            optimizer_confidence.zero_grad()
            outputs = model_confidence(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_confidence.step()

            if (iteration % args.eval_freq == 0) and (iteration > 0):
                print("\n>> Testing <<")
                model_confidence.eval()
                test_loss = 0
                test_total = 0

                for _, (features, labels, indices) in enumerate(testloader_confidence):
                    if features.shape[0] == 1:
                        continue

                    features, labels = features.to(device), labels.to(device)
                    outputs = model_confidence(features)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()
                    test_total += len(labels)
                test_loss /= test_total
                cprint(">> [Epoch: {}] Test Acc: {:3.3f}%\n".format(epoch, test_loss), "yellow")
                model_confidence.train() # switch back to train status

        scheduler_confidence.step()

    # Final testing
    model.eval()
    test_loss = 0
    test_total = 0
    test_confidence = torch.zeros(len(testset), num_class)
    test_confidence_delta = torch.zeros(len(testset))
    test_prediction = torch.zeros(len(testset))

    for _, (features, labels, indices) in enumerate(test_loader):
        if features.shape[0] == 1:
            continue

        features, labels = features.to(device), labels.to(device)
        outputs = model_confidence(features)
        loss = criterion(outputs, labels)
        _, predicted = outputs.max(1)
        test_loss += loss.item()
        test_total += len(labels)

        test_confidence[indices] = F.softmax(outputs.detach().cpu(), dim=1).float()
        test_confidence_delta[indices] = torch.abs(predicted - labels).detach().cpu().float()
        test_prediction[indices] = predicted.detach().cpu().float()

    cprint(">> Final Test MSE: {:3.3f}".format(test_loss/test_total))

    os.makedirs("./result", exist_ok=True)
    with open("./result/test_confidence_delta.pkl", 'wb') as f:
        pkl.dump(test_confidence_delta, f)
    f.close()

    return 0

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--noise_type", default=0, help="noise level", type=int)
    parser.add_argument("--network", default='preact_resnet34', help="network architecture", type=str)
    parser.add_argument("--nepochs", default=20, help="number of training epochs", type=int)
    parser.add_argument("--gpus", default='0', help="how many GPUs to use", type=str)
    parser.add_argument("--warm_up", default=1, help="warm-up period", type=int)
    parser.add_argument("--eval_freq", default=2000, help="evaluation frequency (every a few iterations)", type=int)
    args = parser.parse_args()

    main(args)
