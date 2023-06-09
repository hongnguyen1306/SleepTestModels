from flask import Flask, request, render_template
# from flask_ngrok import run_with_ngrok
import os
import torch
import numpy as np
import pandas as pd
from datetime import datetime
import argparse
# import subprocess
import template
from dataloader.generate import generate
from main import load_model_TCC, load_model_Attn, load_model_Tiny, load_model_Deepsleep
from dataloader.dataloader_pytorch import data_generator




parser = argparse.ArgumentParser()

######################## Model parameters ########################
home_dir = os.getcwd()
parser.add_argument('--experiment_description', default='Exp1', type=str,
                    help='Experiment Description')
parser.add_argument('--run_description', default='run1', type=str,
                    help='Experiment Description')
parser.add_argument('--seed', default=0, type=int,
                    help='seed value')
parser.add_argument('--training_mode', default='supervised', type=str,
                    help='Modes of choice: random_init, supervised, self_supervised, fine_tune, train_linear')
parser.add_argument('--selected_dataset', default='Epilepsy', type=str,
                    help='Dataset of choice: sleepEDF, HAR, Epilepsy, pFD')
parser.add_argument('--logs_save_dir', default='experiments_logs', type=str,
                    help='saving directory')
parser.add_argument('--home_path', default=home_dir, type=str,
                    help='Project home directory')
args = parser.parse_args()



device=torch.device('cpu')
experiment_description = args.experiment_description
data_type = args.selected_dataset
method = 'TS-TCC'
training_mode = args.training_mode
run_description = args.run_description

logs_save_dir = args.logs_save_dir
os.makedirs(logs_save_dir, exist_ok=True)

app = Flask(__name__, template_folder='/home/rosa/TestModels/template')

@app.route('/')
def home():
    return render_template('index.html')

# Define route
@app.route('/test', methods=['POST'])
def test_model():
    # Load datasets
    uploaded_file = request.files['file']
   
    test_dl = generate(uploaded_file, '/home/rosa')
    base_path = "/home/rosa"

    # Load datasets
    data_path = str(os.path.join(base_path,"TestModels/data"))
    test_dl = data_generator(str(os.path.join(base_path, "TestModels/data/test_data.pt")))


    print("*****    ReLU    ******")
    total_acc_TS, outs, trgs = load_model_TCC(test_dl, base_path, method='TS', act_func='ReLU')
    total_acc_CA, outs, trgs = load_model_TCC(test_dl, base_path, method='CA', act_func='ReLU')
    
    print("*****    GELU    ******")
    total_acc_TS, outs, trgs = load_model_TCC(test_dl, base_path, method='TS', act_func='GELU')
    total_acc_CA, outs, trgs = load_model_TCC(test_dl, base_path, method='CA', act_func='GELU')
    
    total_acc_Attn, outs, trgs = load_model_Attn(test_dl, base_path)
    total_acc_tiny, total_f1_tiny, cm_tiny = load_model_Tiny(test_dl, base_path, act_func = 'ReLU')
    load_model_Tiny(test_dl, base_path, act_func = 'GELU')
    load_model_Deepsleep(test_dl, base_path)
    
    return render_template('index.html', result_CA='acc CA :{}'.format(total_acc_CA), result_TS='acc TS :{}'.format(total_acc_TS))

if __name__ == '__main__':
    app.run()
