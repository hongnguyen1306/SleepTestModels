import argparse
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataloader.dataloader_pytorch import data_generator
import models.pytorch_models.Attn_models.model as module_arch
from models.pytorch_models.TS_models.model import base_Model
from config_files.pytorch_configs.attn_configs import ConfigParser
from config_files.pytorch_configs.TCC_configs import Config as Configs
from tiny_test import predict as predict_tiny
from deepsleep_test import predict as predict_deepsleep

start_time = datetime.now()

device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
print("The model will be running on", device, "device\n") 


def model_evaluate(model, test_dl, device, method):
    model.eval()

    total_loss = []
    total_acc = []

    outs = np.array([])
    trgs = np.array([])

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data, labels, _, _ in test_dl:
            data, labels = data.float().to(device), labels.long().to(device)

            output = model(data)
            if method == 'TCC': 
                predictions, _ = output
            elif method == 'Attn':
                predictions = output
                
            loss = criterion(predictions, labels)
            total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())
            total_loss.append(loss.item())

            pred = predictions.max(1, keepdim=True)[1]  # get the index of the max log-probability
            outs = np.append(outs, pred.cpu().numpy())
            trgs = np.append(trgs, labels.data.cpu().numpy())

    total_loss = torch.tensor(total_loss).mean()
    total_acc = torch.tensor(total_acc).mean()

    print("Test loss: ", total_loss, "\t | \tTest Accuracy: ",total_acc)

    # Chuyển đổi các mảng thành kiểu dữ liệu integer
    outs = outs.astype(int)
    trgs = trgs.astype(int)

    # Tính tỉ lệ dự đoán đúng cho từng nhãn
    accuracy = {}
    for label in range(5):
        indices = np.where(trgs == label)[0]
        correct_predictions = np.sum(outs[indices] == trgs[indices])
        accuracy[label] = correct_predictions / len(indices)

    # In kết quả
    for label, acc in accuracy.items():
        print("Nhãn ", label, " Tỉ lệ dự đoán đúng = ",acc)

    return total_loss, total_acc, outs, trgs

def load_model_TCC(test_dl, base_path, method, act_func):
    # Load the model
    configs = Configs()
    model = base_Model(configs, activation_func=act_func).to(device)
    if act_func == 'ReLU':
        if method == 'TS':
            print("======         TS TCC Sleep         ======")
            load_from = "TestModels/input/mode_TS"
            checkpoint = torch.load(os.path.join(base_path, load_from, "model_epoch_25_ReLU.pt"), map_location=device)
        else:
            print("======         CA TCC Sleep         ======")
            load_from =  'TestModels/input/exp3CA/run_1/supervised_seed_123/saved_models/'
            checkpoint = torch.load(os.path.join(base_path, load_from, "model_epoch_19.pt"), map_location=device)
        
    if act_func == 'GELU':
        if method == 'TS':
            print("======         TS TCC Sleep         ======")
            load_from = "TestModels/input/mode_TS"
            checkpoint = torch.load(os.path.join(base_path, load_from, "model_epoch_17_GELU.pt"), map_location=device)
        else:
            print("======         CA TCC Sleep         ======")
            load_from =  'TestModels/input/exp5CAGELU/run_1/supervised_seed_123/saved_models/'
            checkpoint = torch.load(os.path.join(base_path, load_from, "model_epoch_22.pt"), map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    total_loss, total_acc, outs, trgs = model_evaluate(model, test_dl, 'cpu', 'TCC')
    
    return total_loss, total_acc, outs, trgs

def load_model_Attn(test_dl, base_path):
    # fix random seeds for reproducibility
    SEED = 123
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)

    config_path = str(os.path.join(base_path, "TestModels/models/pytorch_models/Attn_models/config.json"))

    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=config_path, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default="cpu", type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-f', '--fold_id', type=str,
                      help='fold_id')
    args.add_argument('-da', '--np_data_dir', type=str,
                      help='Directory containing numpy files')

    options = []
    fold_id = 1

    config = ConfigParser.from_args(args, fold_id, options)
    model = config.init_obj('arch', module_arch)

     # Load the saved checkpoint
    resume_path = str(os.path.join(base_path,"TestModels/input/exp3Attn/checkpoint-epoch96.pth"))
    checkpoint = torch.load(resume_path, map_location=device)

    print("Checkpoint loaded. Resume training from epoch {}".format(checkpoint['epoch']))

    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    print("======          ATTN Sleep         ======")
    total_loss, total_acc, outs, trgs = model_evaluate(model, test_dl, 'cpu', 'Attn')

    return total_loss, total_acc, outs, trgs

def load_model_Tiny(test_dl, base_path, act_func):

    if act_func == 'ReLU':
        model_path = "TestModels/input/attn81.9ReLU"
    else:
        model_path = "TestModels/input/BestModelGELU"
    predict_tiny(
        config_file= str(os.path.join(base_path, "TestModels/config_files/pytorch_configs/tiny_configs.py")),
        model_dir=str(os.path.join(base_path, model_path)),
        output_dir=str(os.path.join(base_path, model_path)),
        log_file='output.log',
        use_best=False,
        act_func = act_func
    )

def load_model_Deepsleep(test_dl, base_path):
    n_subjects = 1
    n_subjects_per_fold = 1
    predict_deepsleep(
        data_dir=str(os.path.join(base_path,"TestModels/data")),
        model_dir=str(os.path.join(base_path, "TestModels")),
        output_dir=str(os.path.join(base_path, "TestModels")),
        n_subjects=n_subjects,
        n_subjects_per_fold=n_subjects_per_fold,
        base_path=base_path
    )

if __name__ == "__main__":
    # root
    base_path = "/home/rosa"


    # Load datasets
    data_path = str(os.path.join(base_path,"TestModels/data"))
    test_dl = data_generator(str(os.path.join(base_path, "TestModels/data/test_4072.pt")))


    print("*****    ReLU    ******")
    total_loss, total_acc, outs, trgs = load_model_TCC(test_dl, base_path, method='TS', act_func='ReLU')
    total_loss, total_acc, outs, trgs = load_model_TCC(test_dl, base_path, method='CA', act_func='ReLU')
    
    print("*****    GELU    ******")
    total_loss, total_acc, outs, trgs = load_model_TCC(test_dl, base_path, method='TS', act_func='GELU')
    total_loss, total_acc, outs, trgs = load_model_TCC(test_dl, base_path, method='CA', act_func='GELU')
    
    total_loss, total_acc, outs, trgs = load_model_Attn(test_dl, base_path)
    load_model_Tiny(test_dl, base_path, act_func = 'ReLU')
    load_model_Tiny(test_dl, base_path, act_func = 'GELU')
    load_model_Deepsleep(test_dl, base_path)

