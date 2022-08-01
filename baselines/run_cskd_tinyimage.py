import os
import sys
import json
from typing import List
sys.path.append("../")
sys.path.append("/home/songzhu/SLDenoise/")

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, BatchSampler, SequentialSampler
from torch.nn import DataParallel
from torch import nn, optim
from torchvision import transforms
import numpy as np
import argparse
import random
from tqdm import tqdm
from termcolor import cprint
import datetime
import pickle as pkl

from data.TINYIMAGENET_CSKD import TImgNetDatasetTrain, TImgNetDatasetTest, PairBatchSampler
from network.network import resnet18, resnet34
from utils.utils import _init_fn, ECELoss

# Experiment Setting Control Panel
# ---------------------------------------------------
# Data Dir 
DATADIR = "/scr/songzhu/tinyimagenet/tiny-imagenet-200"
# Algorithm setting
TEMPERATURE: float = 4.0 
LAMBDA: float = 1.0
# General setting
TRAIN_VALIDATION_RATIO: float = 0.9
N_EPOCH_OUTER: int = 1
N_EPOCH_INNER_CLS: int = 1
CONF_RECORD_EPOCH: int = N_EPOCH_INNER_CLS - 1
LR: float = 1e-4
WEIGHT_DECAY: float = 1e-3
BATCH_SIZE: int = 64 
SCHEDULER_DECAY_MILESTONE: List = [40, 80, 120]
MONITOR_WINDOW: int = 1
NUM_SAMPLES: int = 5
# ----------------------------------------------------

class KDLoss(nn.Module):
    def __init__(self, temp_factor):
        super(KDLoss, self).__init__()
        self.temp_factor = temp_factor
        self.kl_div = nn.KLDivLoss(reduction="sum")

    def forward(self, input, target):
        log_p = torch.log_softmax(input/self.temp_factor, dim=1)
        q = torch.softmax(target/self.temp_factor, dim=1)
        loss = self.kl_div(log_p, q)*(self.temp_factor**2)/input.size(0)
        return loss

def main(args):

    seed = args.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # need to set to True as well

    device = torch.device(f"cuda:0" if torch.cuda.is_available() else 'cpu')

    traindir = os.path.join(DATADIR, 'train')
    testdir  = os.path.join(DATADIR, 'val')

    train_transforms = transforms.Compose([
        transforms.Resize(224), 
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transforms = transforms.Compose([
        transforms.Resize(224), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    trainset = TImgNetDatasetTrain(traindir, train_transforms, split='train', train_valid_split=TRAIN_VALIDATION_RATIO)
    validset = TImgNetDatasetTrain(traindir, test_transforms, split='valid', train_valid_split=TRAIN_VALIDATION_RATIO)
    testset  = TImgNetDatasetTest(os.path.join(testdir, 'images'), 
                            os.path.join(testdir, 'val_annotations.txt'), 
                            class_to_idx=trainset.class_to_idx.copy(), 
                            transform=test_transforms)
    y_tilde_valid = validset.targets
    y_tilde_test  = testset.targets
    num_classes = len(trainset.classes)
    
    get_train_sampler = lambda d: PairBatchSampler(d, BATCH_SIZE)
    get_test_sampler  = lambda d: BatchSampler(SequentialSampler(d), BATCH_SIZE, False)

    train_loader = DataLoader(trainset, pin_memory=True, num_workers=1, worker_init_fn=_init_fn(worker_id=seed), batch_sampler=get_train_sampler(trainset))
    valid_loader = DataLoader(validset, pin_memory=True, num_workers=1, worker_init_fn=_init_fn(worker_id=seed), batch_sampler=get_test_sampler(validset))
    test_loader  = DataLoader(testset,  pin_memory=True, num_workers=1, worker_init_fn=_init_fn(worker_id=seed), batch_sampler=get_test_sampler(testset))

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
    print("---------------------------------------------------------")

    model_cls = resnet18(num_classes=num_classes, in_channels=3)
    model_cls = DataParallel(model_cls)
    model_cls = model_cls.to(device)

    optimizer_cls = torch.optim.Adam(model_cls.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler_cls = torch.optim.lr_scheduler.MultiStepLR(optimizer_cls, gamma=0.5, milestones=SCHEDULER_DECAY_MILESTONE)

    criterion_cls = torch.nn.CrossEntropyLoss()
    criterion_kd  = KDLoss(TEMPERATURE)
    criterion_calibrate = ECELoss()

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
    _count = 0
    _best_cali_ece = np.inf
    _best_cali_acc = 0

    for epoch in range(N_EPOCH_INNER_CLS):

        cprint(f">>>Epoch [{epoch + 1}|{N_EPOCH_INNER_CLS}] Training <<<", "green")

        train_correct = 0
        train_total = 0
        train_loss = 0

        model_cls.train()
        for ib, (indices, images, labels) in enumerate(tqdm(train_loader, ascii=True, ncols=100)):
            if images.shape[0] == 1:
                continue
            
            optimizer_cls.zero_grad()
            images, labels = images.to(device), labels.to(device)
            x, y, x_tilde, _ = images[:BATCH_SIZE], labels[:BATCH_SIZE], images[BATCH_SIZE:], labels[BATCH_SIZE:]
            
            if len(x_tilde)==0:
                x_tilde = x
            
            outs = model_cls(x)
            with torch.no_grad():
                outs_tilde = model_cls(x_tilde)
            _, predict = outs[:, :num_classes].max(1)

            loss_ce = criterion_cls(outs, y)
            loss_kd = criterion_kd(outs, outs_tilde.detach())
            loss = loss_ce + LAMBDA*loss_kd # CS-KD loss

            loss.backward()
            optimizer_cls.step()

            train_loss += loss.detach().cpu().item()
            train_correct += predict.eq(y).sum().item()
            train_total += len(y)

        train_acc = train_correct / train_total
        scheduler_cls.step()

        if not (epoch % MONITOR_WINDOW):

            valid_correct_cali = 0
            valid_total = 0

            # For ECE
            valid_f_cali = torch.zeros(len(validset), num_classes).float()

            model_cls.eval()
            for _, (indices, images, labels) in enumerate(tqdm(valid_loader, ascii=True, ncols=100)):
                if images.shape[0] == 1:
                    continue
                images, labels = images.to(device), labels.to(device)
                outs_cali = model_cls(images)

                # Calibrated model result record
                prob_outs = torch.softmax(outs_cali, 1)
                _, predict = prob_outs.max(1)
                correct_prediction = predict.eq(labels).float()
                valid_correct_cali += correct_prediction.sum().item()
                valid_f_cali[indices] = prob_outs.detach().cpu()
                valid_total += len(images)

            valid_acc_cali = valid_correct_cali/valid_total
            ece_loss_cali = criterion_calibrate.forward(logits=valid_f_cali, labels=torch.tensor(y_tilde_valid))
            print(f"Step [{epoch + 1}|{N_EPOCH_INNER_CLS}] - Train Loss:[{loss.item():.3f}]" +
                  f"- Train Acc:{train_acc:7.3f}"+
                  f"- Valide Acc Cali:{valid_acc_cali:.3f} - ECE Loss Cali:{ece_loss_cali.item():.3f}")

            # For monitoring purpose
            _loss_record[_count] = float(train_loss/train_total)
            _valid_acc_cali_record[_count] = float(valid_acc_cali)
            _ece_cali_record[_count] = float(ece_loss_cali.item())
            _count += 1

        # Testing stage
        test_correct_cali = 0
        test_total = 0
        # For ECE
        test_f_cali = torch.zeros(len(testset), num_classes).float()
        # For selection 
        test_f_pred = torch.zeros(len(testset), dtype=torch.long) 
        test_gt = torch.zeros(len(testset), dtype=torch.long)

        model_cls.eval()
        for _, (indices, images, labels) in enumerate(tqdm(test_loader, ascii=True, ncols=100)):
            if images.shape[0] == 1:
                continue
            images, labels = images.to(device), labels.to(device)
            outs_cali = model_cls(images)
            
            # Calibrated model result record
            prob_outs = torch.softmax(outs_cali, 1)
            _, predict = prob_outs.max(1)
            correct_prediction = predict.eq(labels).float()
            test_correct_cali += correct_prediction.sum().item()
            test_f_cali[indices] = prob_outs.detach().cpu()
            test_total += len(images)
           
            test_f_pred[indices] = predict.detach().cpu()
            test_gt[indices] = labels.detach().cpu()
           
        test_acc_cali = test_correct_cali/test_total
        ece_loss_cali = criterion_calibrate.forward(logits=test_f_cali, labels=torch.tensor(y_tilde_test))

        if ece_loss_cali < _best_cali_ece:
            _best_cali_ece = ece_loss_cali.item()
            _best_cali_ece_epoch = epoch
        if test_acc_cali > _best_cali_acc:
            _best_cali_acc = test_acc_cali
            _best_cali_acc_epoch = epoch

        print(f"[Current | Best] ECE - Ours: \t [{ece_loss_cali.item():.3f} | {_best_cali_ece:.3f}] \t epoch {_best_cali_ece_epoch}")
        print(f"[Current | Best] Acc - Ours: \t [{test_acc_cali:.3f} | {_best_cali_acc:.3f}] \t epoch {_best_cali_acc_epoch}")
        print(f"Final Test Acc: {test_acc_cali*100:.3f}%")

    if args.figure:
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.use('Agg')
        if not os.path.exists('./figures'):
            os.makedirs('./figures', exist_ok=True)
        fig = plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        x_axis = np.linspace(0, N_EPOCH_INNER_CLS - N_EPOCH_INNER_CLS % MONITOR_WINDOW,
                             int((N_EPOCH_INNER_CLS - N_EPOCH_INNER_CLS % MONITOR_WINDOW) / MONITOR_WINDOW + 1))
        plt.plot(x_axis[:-1], _loss_record[:-1], linewidth=2)
        plt.title("Loss Curve")
        plt.subplot(1, 3, 2)
        plt.plot(x_axis[:-1], _valid_acc_raw_record[:-1], linewidth=2, label="Raw")
        plt.plot(x_axis[:-1], _valid_acc_cali_record[:-1], linewidth=2, label="CSKD")
        plt.title('Acc Curve')
        plt.subplot(1, 3, 3)
        plt.plot(x_axis[:-1], _ece_raw_record[:-1], linewidth=2, label="Raw")
        plt.plot(x_axis[:-1], _ece_cali_record[:-1], linewidth=2, label="CSKD")
        plt.legend()
        plt.title('ECE Curve')

        time_stamp = datetime.datetime.strftime(datetime.datetime.today(), "%Y-%m-%d")
        fig.savefig(os.path.join("./figures", f"exp_log_tinyimagenet_cskd_plot_{time_stamp}.png"))

    return ece_loss_cali.item(), _best_cali_ece, test_acc_cali, _best_cali_acc, test_f_cali, test_f_pred, test_gt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguement for SLDenoise")
    parser.add_argument("--seed", type=int, help="Random seed for the experiment", default=77)
    parser.add_argument("--gpus", type=str, help="Indices of GPUs to be used", default='7')
    parser.add_argument("--rollWindow", default=3, help="rolling window to calculate the confidence", type=int)
    parser.add_argument("--warm_up", default=2, help="warm-up period", type=int)
    parser.add_argument("--figure", action='store_true', help='True to plot performance log')
    # algorithm hp
    parser.add_argument("--alpha", type=float, help="CE loss multiplier", default=100)
    parser.add_argument("--gamma_initial", type=float, help="Gamma initial value", default=0.8)
    parser.add_argument("--gamma_multiplier", type=float, help="Gamma increment multiplier", default=0.2)
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
    exp_config['temperature'] = TEMPERATURE
    exp_config['lambda'] = LAMBDA
    for k, v in args._get_kwargs():
        exp_config[k] = v

    exp_config['ece_cali'] = []
    exp_config['best_ece_cali'] = []
    exp_config['test_acc_cali'] = []
    exp_config['best_acc_cali'] = []
    
    data_save_dict = {}
    
    dir_date = datetime.datetime.today().strftime("%Y%m%d")
    save_folder = os.path.join("./exp_logs/cskd_"+dir_date)
    os.makedirs(save_folder, exist_ok=True)
    save_file_name = 'cskd_' + datetime.date.today().strftime("%d_%m_%Y") + f"tinyimagenet.json"
    save_file_name = os.path.join(save_folder, save_file_name)
    print(save_file_name)
    
    data_save_folder = os.path.join("./rebuttal_data/cskd_"+dir_date)
    os.makedirs(data_save_folder, exist_ok=True)
    data_save_file_name = 'cskd_' + datetime.date.today().strftime("%d_%m_%Y") + f"tinyimagenet.pkl"
    data_save_file_name = os.path.join(data_save_folder, data_save_file_name)
    print(data_save_file_name)
    
    for seed in [77, 78, 79]:
        args.seed = seed
        ece_cali, best_ece_cali, test_acc_cali, best_acc_cali, test_f_cali, test_f_pred, test_gt = main(args)

        exp_config['ece_cali'].append(ece_cali)
        exp_config['best_ece_cali'].append(best_ece_cali)
        exp_config['test_acc_cali'].append(test_acc_cali)
        exp_config['best_acc_cali'].append(best_acc_cali)
        
        data_save_dict[seed] = (test_f_cali, test_gt, test_f_pred)

        with open(save_file_name, "w") as f:
            json.dump(exp_config, f, sort_keys=False, indent=4)
        f.close()
        
        with open(data_save_file_name, 'wb') as f:
            pkl.dump(data_save_dict, f)
        f.close()