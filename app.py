from flask import Flask, request, render_template
# from flask_ngrok import run_with_ngrok
import os
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from werkzeug.utils import secure_filename
# import subprocess
import template
from dataloader.generate import generate
from main import load_model_TCC, load_model_Attn, load_model_Tiny, load_model_Deepsleep
from dataloader.dataloader_pytorch import data_generator
from dataloader.edf_to_npz import EdfToNpz


app = Flask(__name__, template_folder='/home/rosa/TestModels/template')
base_path = "/home/rosa"
data_path = "TestModels/data"

@app.route('/')
def home():
    return render_template('index.html')

# Define route
@app.route('/test', methods=['POST'])
def test_model():
    # Load datasets
    uploaded_files = request.files.getlist('file')
    print("\n************* uploaded_files ", uploaded_files)
    for file in uploaded_files:
        filename = secure_filename(file.filename)
        print("file ", filename)
        file.save(os.path.join(base_path, data_path , filename))

    EdfToNpz(base_path, data_path)
    test_pt = generate(base_path, "TestModels/data/test_data.npz")
    # Load datasets
    test_dl = data_generator(str(os.path.join(base_path, "TestModels/data/test_data.pt")))


    print("*****    ReLU    ******")
    total_acc_TS, outs, trgs = load_model_TCC(test_dl, base_path, method='TS', act_func='ReLU')
    total_acc_CA, outs, trgs = load_model_TCC(test_dl, base_path, method='CA', act_func='ReLU')
    
    print("*****    GELU    ******")
    total_acc_TS, outs, trgs = load_model_TCC(test_dl, base_path, method='TS', act_func='GELU')
    total_acc_CA, outs, trgs = load_model_TCC(test_dl, base_path, method='CA', act_func='GELU')
    
    total_acc_Attn, outs, trgs = load_model_Attn(test_dl, base_path)
    total_acc_tiny, total_f1_tiny, cm_tiny = load_model_Tiny(test_dl, base_path, act_func = 'ReLU')
    total_acc_tiny, total_f1_tiny, cm_tiny = load_model_Tiny(test_dl, base_path, act_func = 'GELU')
    # total_acc_deepsleep, total_f1_deepsleep, cm_deepsleep = load_model_Deepsleep(test_dl, base_path)
    
    return render_template('index.html', result_CA='acc CA : {}'.format(total_acc_CA), result_TS='acc TS : {}'.format(total_acc_TS))

if __name__ == '__main__':
    app.run()
