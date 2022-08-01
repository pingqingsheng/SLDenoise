import os
import sys
import json
from typing import List
sys.path.append("../")

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
from termcolor import cprint
import datetime
import pickle as pkl

from data.TINYIMAGENET import TImgNetDatasetTrain, TImgNetDatasetTest
from network.network import resnet18, resnet34
from utils.utils import _init_fn, ECELoss

# Experiment Setting Control Panel
# ---------------------------------------------------
# Data Dir 
DATADIR = "/scr/songzhu/tinyimagenet/tiny-imagenet-200"
# General setting
TRAIN_VALIDATION_RATIO: float = 0.9
N_EPOCH_OUTER: int = 1
N_EPOCH_INNER_CLS: int = 40
CONF_RECORD_EPOCH: int = N_EPOCH_INNER_CLS - 1
LR: float = 1e-4
WEIGHT_DECAY: float = 1e-3
BATCH_SIZE: int = 64
SCHEDULER_DECAY_MILESTONE: List = [40, 80, 120]
MONITOR_WINDOW: int = 1
NUM_SAMPLES: int = 5
# ----------------------------------------------------

def main(args):

    seed = args.seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # need to set to True as well

    device = torch.device(f"cuda:{args.gpus}" if torch.cuda.is_available() else 'cpu')

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
    y_tilde_test  = testset.classidx
    num_classes = len(trainset.classes)
    
    train_loader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=2, worker_init_fn=_init_fn(worker_id=seed))
    valid_loader = DataLoader(validset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=2, worker_init_fn=_init_fn(worker_id=seed))
    test_loader  = DataLoader(testset,  batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=2, worker_init_fn=_init_fn(worker_id=seed))

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

    model_cls = resnet18(num_classes=num_classes, in_channels=3, pretrained=True)
    gpu_id_list = [int(x) for x in args.gpus.split(",")]
    model_cls = DataParallel(model_cls, device_ids=[x-gpu_id_list[0] for x in gpu_id_list])
    model_cls = model_cls.to(device)

    optimizer_cls = torch.optim.Adam(model_cls.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler_cls = torch.optim.lr_scheduler.MultiStepLR(optimizer_cls, gamma=0.5, milestones=SCHEDULER_DECAY_MILESTONE)

    criterion_cls = torch.nn.CrossEntropyLoss()
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
    _best_raw_ece  = np.inf
    _best_cali_ece = np.inf
    _best_raw_acc  = 0
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
            outs = model_cls(images)
            _, predict = outs[:, :num_classes].max(1)

            loss = criterion_cls(outs, labels)
            loss.backward()
            optimizer_cls.step()

            train_loss += loss.detach().cpu().item()
            train_correct += predict.eq(labels).sum().item()
            train_total += len(labels)

        train_acc = train_correct / train_total
        scheduler_cls.step()

        if not (epoch % MONITOR_WINDOW):

            valid_correct_raw = 0
            valid_correct_cali = 0
            valid_total = 0
            model_cls_cali = ModelWithTemperature(model_cls)
            model_cls_cali.set_temperature(valid_loader)

            # For ECE
            valid_f_raw  = torch.zeros(len(validset), num_classes).float()
            valid_f_cali = torch.zeros(len(validset), num_classes).float()

            model_cls.eval()
            model_cls_cali.eval()
            for _, (indices, images, labels) in enumerate(tqdm(valid_loader, ascii=True, ncols=100)):
                if images.shape[0] == 1:
                    continue
                images, labels = images.to(device), labels.to(device)
                outs_raw  = model_cls(images)
                outs_cali = model_cls_cali(images) 

                # Raw model result record
                prob_outs = torch.softmax(outs_raw[:, :num_classes], 1)
                _, predict = prob_outs.max(1)
                correct_prediction = predict.eq(labels).float()
                valid_correct_raw += correct_prediction.sum().item()
                valid_total += len(labels)
                valid_f_raw[indices] = prob_outs.detach().cpu()
               
                # Calibrated model result record
                prob_outs = torch.softmax(outs_cali, 1)
                _, predict = prob_outs.max(1)
                correct_prediction = predict.eq(labels).float()
                valid_correct_cali += correct_prediction.sum().item()
                valid_f_cali[indices] = prob_outs.detach().cpu()
                
            valid_acc_raw  = valid_correct_raw/valid_total
            valid_acc_cali = valid_correct_cali/valid_total
            ece_loss_raw  = criterion_calibrate.forward(logits=valid_f_raw, labels=torch.tensor(y_tilde_valid))
            ece_loss_cali = criterion_calibrate.forward(logits=valid_f_cali, labels=torch.tensor(y_tilde_valid))
            print(f"Step [{epoch + 1}|{N_EPOCH_INNER_CLS}] - Train Loss:[{loss.item():.3f}]" +
                  f"- Train Acc:{train_acc:7.3f} - Valid Acc Raw:{valid_acc_raw:7.3f} -  ECE Loss:{ece_loss_raw.item():7.3f}"+
                  f"- Valide Acc Cali:{valid_acc_cali:.3f} - ECE Loss Cali:{ece_loss_cali.item():.3f}")

            # For monitoring purpose
            _loss_record[_count] = float(train_loss/train_total)
            _valid_acc_raw_record[_count]  = float(valid_acc_raw)
            _valid_acc_cali_record[_count] = float(valid_acc_cali)
            _ece_raw_record[_count]  = float(ece_loss_raw.item())
            _ece_cali_record[_count] = float(ece_loss_cali.item())
            _count += 1

        # Testing stage
        test_correct_raw = 0
        test_correct_cali = 0
        test_total = 0
        # For ECE
        test_f_raw  = torch.zeros(len(testset), num_classes).float()
        test_f_cali = torch.zeros(len(testset), num_classes).float()
        # For selection 
        test_f_pred = torch.zeros(len(testset), dtype=torch.long) 
        test_gt = torch.zeros(len(testset), dtype=torch.long)

        model_cls.eval()
        model_cls_cali.eval()
        for _, (indices, images, labels) in enumerate(tqdm(test_loader, ascii=True, ncols=100)):
            if images.shape[0] == 1:
                continue
            images, labels = images.to(device), labels.to(device)
            outs_raw  = model_cls(images)
            outs_cali = model_cls_cali(images) 

            # Raw model result record
            prob_outs = torch.softmax(outs_raw[:, :num_classes], 1)
            _, predict = prob_outs.max(1)
            correct_prediction = predict.eq(labels).float()
            test_correct_raw += correct_prediction.sum().item()
            test_total += len(labels)
            test_f_raw[indices] = prob_outs.detach().cpu()

            # Calibrated model result record
            prob_outs = torch.softmax(outs_cali, 1)
            _, predict = prob_outs.max(1)
            correct_prediction = predict.eq(labels).float()
            test_correct_cali += correct_prediction.sum().item()
            test_f_cali[indices] = prob_outs.detach().cpu()
            
            test_f_pred[indices] = predict.detach().cpu()
            test_gt[indices] = labels.detach().cpu()
           
        test_acc_raw  = test_correct_raw/test_total
        test_acc_cali = test_correct_cali/test_total
        ece_loss_raw  = criterion_calibrate.forward(logits=test_f_raw, labels=torch.tensor(y_tilde_test))
        ece_loss_cali = criterion_calibrate.forward(logits=test_f_cali, labels=torch.tensor(y_tilde_test))

        if ece_loss_raw < _best_raw_ece:
            _best_raw_ece = ece_loss_raw.item()
            _best_raw_ece_epoch = epoch
        if ece_loss_cali < _best_cali_ece:
            _best_cali_ece = ece_loss_cali.item()
            _best_cali_ece_epoch = epoch
        if test_acc_raw > _best_raw_acc:
            _best_raw_acc = test_acc_raw
            _best_raw_acc_epoch = epoch
        if test_acc_cali > _best_cali_acc:
            _best_cali_acc = test_acc_cali
            _best_cali_acc_epoch = epoch

        print(f"[Current | Best] ECE - Conf: \t [{ece_loss_raw.item():.3f} | {_best_raw_ece:.3f}] \t epoch {_best_raw_ece_epoch}")
        print(f"[Current | Best] ECE - Ours: \t [{ece_loss_cali.item():.3f} | {_best_cali_ece:.3f}] \t epoch {_best_cali_ece_epoch}")
        print(f"[Current | Best] Acc - Conf: \t [{test_acc_raw:.3f} | {_best_raw_acc:.3f}] \t epoch {_best_raw_acc_epoch}")
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
        plt.plot(x_axis[:-1], _valid_acc_cali_record[:-1], linewidth=2, label="ts")
        plt.legend()
        plt.title('Acc Curve')
        plt.subplot(1, 3, 3)
        plt.plot(x_axis[:-1], _ece_raw_record[:-1], linewidth=2, label="Raw")
        plt.plot(x_axis[:-1], _ece_cali_record[:-1], linewidth=2, label="ts")
        plt.legend()
        plt.title('ECE Curve')

        time_stamp = datetime.datetime.strftime(datetime.datetime.today(), "%Y-%m-%d")
        fig.savefig(os.path.join("./figures", f"exp_log_tinyimagenet_ts_plot_{time_stamp}.png"))

    return ece_loss_raw.item(), ece_loss_cali.item(), _best_cali_ece, test_acc_cali, _best_cali_acc, test_f_cali, test_gt, test_f_pred


class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, valid_loader):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        self.cuda()
        nll_criterion = nn.CrossEntropyLoss().cuda()
        ece_criterion = ECELoss().cuda()

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for indices, input, label in valid_loader:
                input = input.cuda()
                logits = self.model(input)
                logits_list.append(logits)
                labels_list.append(label)
            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        # print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)        # after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        # after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()

        # Calculate NLL and ECE after temperature scaling

        # print('Optimal temperature: %.3f' % self.temperature.item())
        # print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

        return self

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguement for SLDenoise")
    parser.add_argument("--seed", type=int, help="Random seed for the experiment", default=77)
    parser.add_argument("--gpus", type=str, help="Indices of GPUs to be used", default='0')
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
    for k, v in args._get_kwargs():
        exp_config[k] = v

    exp_config['ece_raw'] = []
    exp_config['ece_cali'] = []
    exp_config['best_ece_cali'] = []
    exp_config['test_acc_cali'] = []
    exp_config['best_acc_cali'] = []
    
    data_save_dict = {}
    
    dir_date = datetime.datetime.today().strftime("%Y%m%d")
    save_folder = os.path.join("./exp_logs/ts_"+dir_date)
    os.makedirs(save_folder, exist_ok=True)
    save_file_name = 'ts_' + datetime.date.today().strftime("%d_%m_%Y") + f"tinyimagenet.json"
    save_file_name = os.path.join(save_folder, save_file_name)
    print(save_file_name)
    
    data_save_folder = os.path.join("./rebuttal_data/ts_"+dir_date)
    os.makedirs(data_save_folder, exist_ok=True)
    data_save_file_name = 'ts_' + datetime.date.today().strftime("%d_%m_%Y") + f"tinyimagenet.pkl"
    data_save_file_name = os.path.join(data_save_folder, data_save_file_name)
    print(data_save_file_name)
    
    for seed in [77, 78, 79]:
        args.seed = seed
        ece_raw, ece_cali, best_ece_cali, test_acc_cali, best_acc_cali, test_f_cali, test_gt, test_f_pred = main(args)

        exp_config['ece_raw'].append(ece_raw)
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