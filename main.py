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
from tiny_test import predict

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

    print(f"Test loss: {total_loss.item():.4f}\t | \tTest Accuracy: {total_acc.item():.4f}")

    # Chuyển đổi các mảng thành kiểu dữ liệu integer
    outs = outs.astype(int)
    trgs = trgs.astype(int)

    # Tính tỉ lệ dự đoán đúng cho từng nhãn
    accuracy = {}
    for label in range(5):
        # Lấy chỉ mục các mẫu thuộc nhãn label
        indices = np.where(trgs == label)[0]
        
        # Đếm số lượng dự đoán đúng cho nhãn label
        correct_predictions = np.sum(outs[indices] == trgs[indices])
        
        # Tính tỉ lệ dự đoán đúng
        accuracy[label] = correct_predictions / len(indices)

    # In kết quả
    for label, acc in accuracy.items():
        print(f"Nhãn {label}: Tỉ lệ dự đoán đúng = {acc}")

    return total_loss, total_acc, outs, trgs

def load_model_CA(test_dl, base_path):
    configs = Configs()

    # Load the model
    model = base_Model(configs).to(device)
    # E:\TestModels\input\exp5CAGELU\run_1\supervised_seed_123\saved_models\model_epoch_19.pt
    # Load the saved checkpoint
    load_from =  'TestModels/input/exp5CAGELU/run_1/supervised_seed_123/saved_models/'
    checkpoint = torch.load(os.path.join(base_path, load_from, "model_epoch_19.pt"), map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("======         TS TCC Sleep         ======")
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

def load_model_Tiny(test_dl, base_path):

    predict(
        config_file= str(os.path.join(base_path, "TestModels/config_files/pytorch_configs/tiny_configs.py")),
        model_dir=str(os.path.join(base_path, "input/81.9")),
        output_dir=str(os.path.join(base_path, "TestModels/input/81.9")),
        log_file='output.log',
        use_best=False,
    )

if __name__ == "__main__":
    # root
    base_path = "/home/rosa"


    # Load datasets
    data_path = str(os.path.join(base_path,"TestModels/data"))
    test_dl = data_generator(str(os.path.join(base_path, "TestModels/data/test_4072.pt")))


    total_loss, total_acc, outs, trgs = load_model_CA(test_dl, base_path)
    total_loss, total_acc, outs, trgs = load_model_Attn(test_dl, base_path)
    load_model_Tiny(test_dl, base_path)

