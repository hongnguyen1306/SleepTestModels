import argparse
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import models.pytorch_models.Attn_models.model as module_arch
from dataloader.dataloader_pytorch import data_generator
from models.pytorch_models.TS_models.model import base_Model
from config_files.pytorch_configs.attn_configs import ConfigParser

start_time = datetime.now()

parser = argparse.ArgumentParser()

######################## Model parameters ########################
home_dir = os.getcwd()
parser.add_argument('--experiment_description',     default='HAR_experiments',  type=str,   help='Experiment Description')
parser.add_argument('--run_description',            default='test1',            type=str,   help='Experiment Description')
parser.add_argument('--seed',                       default=0,                  type=int,   help='seed value')
parser.add_argument('--training_mode',              default='self_supervised',  type=str,
                    help='Modes of choice: random_init, supervised, self_supervised, SupCon, ft_1p, gen_pseudo_labels')

parser.add_argument('--selected_dataset',           default='HAR',              type=str,   help='Dataset of choice: EEG, HAR, Epilepsy, pFD')
parser.add_argument('--data_path',                  default='/Users/vdq1511/Downloads/CA-TCC/',           type=str,   help='Path containing dataset')

parser.add_argument('--logs_save_dir',              default='experiments_logs', type=str,   help='saving directory')
parser.add_argument('--device',                     default='cuda',           type=str,   help='cpu or cuda')
parser.add_argument('--home_path',                  default=home_dir,           type=str,   help='Project home directory')
args = parser.parse_args()

device = torch.device(args.device)
experiment_description = args.experiment_description
data_type = args.selected_dataset
training_mode = args.training_mode
run_description = args.run_description

logs_save_dir = args.logs_save_dir
os.makedirs(logs_save_dir, exist_ok=True)

exec(f'from config_files.pytorch_configs.TCC_configs import Config as Configs')



# ##### fix random seeds for reproducibility ########
SEED = args.seed
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
#####################################################


experiment_log_dir = os.path.join(logs_save_dir, experiment_description, run_description,
                                  training_mode + f"_seed_{SEED}")
os.makedirs(experiment_log_dir, exist_ok=True)

# Logging
log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")



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
    total_loss, total_acc, outs, trgs = model_evaluate(model, test_dl, device, 'TCC')
    

    return total_loss, total_acc, outs, trgs

def load_model_Attn(test_dl, base_path):
    # fix random seeds for reproducibility
    SEED = 123
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)

    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default="/TestModels/models/pytorch_models/Attn_models/config.json", type=str,
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

    print("======            ATTN Sleep           ======")
    total_loss, total_acc, outs, trgs = model_evaluate(model, test_dl, 'cpu', 'Attn')

    return total_loss, total_acc, outs, trgs


if __name__ == "__main__":
    # root
    base_path = "/"


    data_path = "test_4072.pt"
    # Load datasets
    test_dl = data_generator(data_path)

    total_loss, total_acc, outs, trgs = load_model_CA(test_dl, base_path)
    total_loss, total_acc, outs, trgs = load_model_Attn(test_dl, base_path)

