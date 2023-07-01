import argparse
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd


from dataloader.dataloader_pytorch import data_generator
import models.pytorch_models.Attn_models.model as module_arch
from models.pytorch_models.TS_models.model import base_Model
from config_files.pytorch_configs.attn_configs import ConfigParser
from config_files.pytorch_configs.TCC_configs import Config as Configs
from tiny_test import predict_tiny, predict_tiny_nolabels
from deepsleep_test import predict_deepsleep, predict_deepsleep_nolabels
from dataloader.dataloader_pytorch import data_generator
from dataloader.generate import generate_nolabels, generate_withlabels



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

    # Áp dụng phép làm tròn
    total_acc = round(total_acc.item() * 100, 2)
    total_loss = round(total_loss.item() * 100, 2)



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


def model_predict(model, test_dl, device, method):
    model.eval()

    outs = np.array([])

    with torch.no_grad():
        for data, _, _ in test_dl:
            data = data.float().to(device)
            output = model(data)
            if method == 'TCC': 
                predictions, _ = output
            elif method == 'Attn':
                predictions = output
                
            pred = predictions.max(1, keepdim=True)[1]  # get the index of the max log-probability
            outs = np.append(outs, pred.cpu().numpy())

    return outs

def load_model_TCC(test_dl, base_path, method, act_func, labels=True):
    # Load the model
    configs = Configs()
    model = base_Model(configs, activation_func=act_func).to(device)
    if act_func == 'ReLU':
        if method == 'TS':
            print("======         TS TCC Sleep   RELU      ======")
            load_from = "input/exp5TS/run_1/supervised_seed_123/saved_models"
            checkpoint = torch.load(os.path.join(base_path, load_from, "model_epoch_40.pt"), map_location=device)
        else:
            print("======         CA TCC Sleep         ======")
            load_from =  'input/exp3CA/run_1/supervised_seed_123/saved_models/'
            checkpoint = torch.load(os.path.join(base_path, load_from, "model_epoch_30.pt"), map_location=device)
        
    if act_func == 'GELU':
        if method == 'TS':
            print("======         TS TCC Sleep GELU        ======")
            load_from = "input/TS_GELU_exp16/run_1/supervised_seed_123/saved_models"
            checkpoint = torch.load(os.path.join(load_from, "model_epoch_40.pt"), map_location=device)
        else:
            print("======         CA TCC Sleep         ======")
            load_from =  'input/exp5CAGELU/run_1/supervised_seed_123/saved_models/'
            checkpoint = torch.load(os.path.join(base_path, load_from, "model_epoch_18.pt"), map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    total_loss = []
    total_acc = []

    outs = np.array([])
    trgs = np.array([])
    

    # # Convert numpy arrays to pandas dataframes
    # outs_df = pd.DataFrame(outs)
    # trgs_df = pd.DataFrame(trgs)

    # # Define the file paths for saving the Excel files
    # outs_file_path = 'outs.xlsx'
    # trgs_file_path = 'trgs.xlsx'

    # # Save the dataframes to Excel files
    # outs_df.to_excel(outs_file_path, index=False)
    # trgs_df.to_excel(trgs_file_path, index=False)

    # print("Data TCC has been exported to Excel files.")

    if labels==False:
        outs = model_predict(model, test_dl, 'cpu', 'TCC')
    else:
        total_loss, total_acc, outs, trgs = model_evaluate(model, test_dl, 'cpu', 'TCC')
    
    np.set_printoptions(threshold=np.inf)
    # print("TCC Nhãn dự đoán ", outs)
    # print("TCC Nhãn đúng ", trgs)

    return total_loss, total_acc, outs, trgs

def load_model_Attn(test_dl, base_path, labels=True):
    # fix random seeds for reproducibility
    SEED = 123
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    np.random.seed(SEED)

    config_path = str(os.path.join(base_path, "models/pytorch_models/Attn_models/config.json"))

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
    
    total_acc = []

    outs = np.array([])
    trgs = np.array([])

    config = ConfigParser.from_args(args, 1, [])
    model = config.init_obj('arch', module_arch)

     # Load the saved checkpoint
    resume_path = str(os.path.join(base_path,"input/exp3Attn/checkpoint-epoch96.pth"))
    checkpoint = torch.load(resume_path, map_location=device)

    print("Checkpoint loaded. Resume training from epoch {}".format(checkpoint['epoch']))

    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    print("======          ATTN Sleep         ======")
    if labels==True:
        total_loss, total_acc, outs, trgs = model_evaluate(model, test_dl, 'cpu', 'Attn')
    else:
        outs = model_predict(model, test_dl, 'cpu', 'Attn')
    
        # Convert numpy arrays to pandas dataframes


    return total_acc, outs, trgs

def load_model_Tiny(data_path, base_path, act_func, labels=True):
    f1_score = 0
    acc = 0
    preds = np.array([])
    print("Tiny load...1")

    if act_func == 'ReLU':
        model_path = "input/tiny81.9ReLU"
    else:
        model_path = "input/BestModelGELU"

    print("Tiny load...2")
    

    print("model_path ", model_path)
    print("Tiny load...3")

    if labels==True:
        acc, f1_score, preds = predict_tiny(
            config_file= str(os.path.join(base_path, "config_files/pytorch_configs/tiny_configs.py")),
            output_dir=str(os.path.join(base_path, model_path)),
            data_dir=str(os.path.join(base_path, data_path)),
            use_best=False,
            act_func = act_func
        )
    else:
        preds = predict_tiny_nolabels(
            config_file= str(os.path.join(base_path, "config_files/pytorch_configs/tiny_configs.py")),
            output_dir=str(os.path.join(base_path, model_path)),
            data_dir=str(os.path.join(base_path, data_path)),
            use_best=True,
            act_func = act_func
        )
    acc = round(acc * 100, 2)
    f1_score = round(f1_score * 100, 2)
    print("acc , f1 ", acc , " ", f1_score)
    return acc, f1_score, preds

def load_model_Deepsleep(data_path, base_path, labels=True):
    n_subjects = 1
    n_subjects_per_fold = 1
    f1 = 0
    acc = 0

    if labels==True:
        acc, f1, outs = predict_deepsleep(
            data_dir=str(os.path.join(base_path, data_path)),
            model_dir=str(os.path.join(base_path, "TestModels")),
            output_dir=str(os.path.join(base_path, "TestModels")),
            n_subjects=n_subjects,
            n_subjects_per_fold=n_subjects_per_fold,
            base_path=base_path
        )
    else:
        outs = predict_deepsleep_nolabels(
            data_dir=str(os.path.join(base_path, data_path)),
            model_dir=str(os.path.join(base_path, "TestModels")),
            output_dir=str(os.path.join(base_path, "TestModels")),
            n_subjects=n_subjects,
            n_subjects_per_fold=n_subjects_per_fold,
            base_path=base_path
        )

    acc = round(acc * 100, 2)
    f1 = round(f1 * 100, 2)
    print("acc , f1 ", acc , " ", f1)
    return acc, f1, outs

def main():
    # root
    base_path = "/home/rosa/TestModels"

    # Load datasets
    # data_path = str(os.path.join(base_path,"data"))
    # test_dl = data_generator(str(os.path.join(base_path, "data/test_data.pt")))


    val_folder = "/home/rosa/TestModels/Tel_test_npz"
    for test_npz_file in os.listdir(val_folder):
        if test_npz_file.endswith('.npz'):
            test_npz_path = os.path.join(val_folder, test_npz_file)
            generate_withlabels(base_path, test_npz_path)
            test_pt = data_generator(str(os.path.join(base_path, "test_data.pt")), labels=True)

            # print("\n*****    ReLU    ******")
            loss_TS, acc_TS, outs_TS, trues = load_model_TCC(test_pt, base_path, method='TS', act_func='ReLU')
            loss_CS, acc_CA, outs_CA, trgs = load_model_TCC(test_pt, base_path, method='CA', act_func='ReLU')

            # print("\n*****    GELU    ******")
            loss_TS_G, acc_TS_G, outs_TS_G, trgs_G  = load_model_TCC(test_pt, base_path, method='TS', act_func='GELU')
            loss_CA_G, acc_CA_G, outs_CA_G, trgs_G  = load_model_TCC(test_pt, base_path, method='CA', act_func='GELU')


            acc_Attn, outs_attn, trgs = load_model_Attn(test_pt, base_path, labels=True)
            acc_tiny_relu, f1_tiny_relu, outs_tiny_ReLU = load_model_Tiny(test_npz_path, base_path, act_func = 'ReLU', labels=True)
            acc_tiny_gelu, f1_tiny_gelu, outs_tiny_GELU = load_model_Tiny(test_npz_path, base_path, act_func = 'GELU', labels=True)
            acc_deepsleep, f1_deepsleep, outs_deepsleep = load_model_Deepsleep(test_npz_path, base_path, labels=True)

            # results = {
            #     'trgs': trgs,
            #     'outs_TS_G': outs_TS_G,
            #     'outs_TS': outs_TS,
            #     'outs_CA': outs_CA,
            #     'outs_CA_G': outs_CA_G,
            #     'outs_tiny_ReLU': outs_tiny_ReLU,
            #     'outs_tiny_GELU': outs_tiny_GELU
            # }

            # # Convert the dictionary to a dataframe
            # results_df = pd.DataFrame(results)
            # file_name = os.path.splitext(test_npz_file)[0]
            # # Define the file path for saving the Excel file    
            # excel_file_path = '/home/rosa/excel_test/' + str(file_name) + '_results.xlsx'

            # # Save the dataframe to the Excel file
            # results_df.to_excel(excel_file_path, index=False)

    # test_path = "data/test_data.npz"

    # generate_withlabels(base_path, test_path)
    # test_pt = data_generator(str(os.path.join(base_path, "data/test_data.pt")), labels=True)

    # # print("\n*****    ReLU    ******")
    # loss_TS, acc_TS, outs_TS, trues = load_model_TCC(test_pt, base_path, method='TS', act_func='ReLU')
    # loss_CS, acc_CA, outs_CA, trgs = load_model_TCC(test_pt, base_path, method='CA', act_func='ReLU')

    # # print("\n*****    GELU    ******")
    # loss_TS_G, acc_TS_G, outs_TS_G, trgs_G  = load_model_TCC(test_pt, base_path, method='TS', act_func='GELU')
    # loss_CA_G, acc_CA_G, outs_CA_G, trgs_G  = load_model_TCC(test_pt, base_path, method='CA', act_func='GELU')


    # acc_Attn, outs_attn, trgs = load_model_Attn(test_pt, base_path, labels=True)
    # acc_tiny_relu, f1_tiny_relu, outs_tiny_ReLU = load_model_Tiny(test_path, base_path, act_func = 'ReLU', labels=True)
    # acc_tiny_gelu, f1_tiny_gelu, outs_tiny_GELU = load_model_Tiny(test_path, base_path, act_func = 'GELU', labels=True)
    # acc_deepsleep, f1_deepsleep, outs_deepsleep = load_model_Deepsleep(test_path, base_path, labels=True)

        

if __name__ == '__main__':
    main()

