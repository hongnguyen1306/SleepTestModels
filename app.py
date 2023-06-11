import glob
from flask import Flask, request, render_template, make_response
# from flask_ngrok import run_with_ngrok
import os
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import random
import string
import mne
# import subprocess
import template
from dataloader.generate import generate
from main import load_model_TCC, load_model_Attn, load_model_Tiny, load_model_Deepsleep
from dataloader.dataloader_pytorch import data_generator
from dataloader.edf_to_npz import EdfToNpz

app = Flask(__name__, template_folder='/home/rosa/TestModels/template')
base_path = "/home/rosa"
data_path = "TestModels/data"

def delete_files_with_extension(folder_path, extension):
    for file_name in os.listdir(folder_path):
        if file_name.endswith(extension):
            file_path = os.path.join(folder_path, file_name)
            os.remove(file_path)

@app.route('/')
def home():
    return render_template('index.html')

# Define route
@app.route('/test', methods=['POST', 'GET'])
def test_model():
    # Load datasets
    delete_files_with_extension("/home/rosa/TestModels/data", 'PSG.edf')
    delete_files_with_extension("/home/rosa/TestModels/data", 'Hypnogram.edf')
    delete_files_with_extension("/home/rosa/TestModels/static", '.png')

    uploaded_files = request.files.getlist('file')
    print("\n************* uploaded_files ", uploaded_files)
    for file in uploaded_files:
        filename = secure_filename(file.filename)
        print("file ", filename)
        file.save(os.path.join(base_path, data_path , filename))


    psg_file = glob.glob(os.path.join(base_path, data_path, "*PSG.edf"))
    ann_file = glob.glob(os.path.join(base_path, data_path, "*Hypnogram.edf"))
    raw = mne.io.read_raw_edf(psg_file[0])
    annotations = mne.read_annotations(ann_file[0])
    raw.set_annotations(annotations)
    channel_names = ["EEG Fpz-Cz"]
    two_meg_chans = raw[channel_names, 0:10000]
    y_offset = np.array([5e-11, 0])  # just enough to separate the channel traces
    x = two_meg_chans[1]
    y = two_meg_chans[0].T + y_offset
    lines = plt.plot(x, y)
    plt.legend(lines, channel_names)
    raw.plot()
    random_number = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
    filename = 'result_' + random_number + '-raw_edf.png'
    plt.savefig(os.path.join('/home/rosa/TestModels/static', filename))
    plt.close()

    EdfToNpz(base_path, data_path)
    generate(base_path, "TestModels/data/test_data.npz")
    test_dl = data_generator(str(os.path.join(base_path, "TestModels/data/test_data.pt")))


    print("\n*****    ReLU    ******")
    total_acc_TS, outs_TS, trgs_TS = load_model_TCC(test_dl, base_path, method='TS', act_func='ReLU')
    total_acc_CA, outs_CA, trgs_CA = load_model_TCC(test_dl, base_path, method='CA', act_func='ReLU')
    
    print("\n*****    GELU    ******")
    total_acc_TS_gelu, outs_TS_gelu, trgs_TS_gelu = load_model_TCC(test_dl, base_path, method='TS', act_func='GELU')
    total_acc_CA_gelu, outs_CA_gelu, trgs_CA_gelu = load_model_TCC(test_dl, base_path, method='CA', act_func='GELU')
    
    # total_acc_Attn, outs, trgs = load_model_Attn(test_dl, base_path)
    # total_acc_tiny, total_f1_tiny, cm_tiny = load_model_Tiny(test_dl, base_path, act_func = 'ReLU')
    # total_acc_tiny, total_f1_tiny, cm_tiny = load_model_Tiny(test_dl, base_path, act_func = 'GELU')
    # total_acc_deepsleep, total_f1_deepsleep, cm_deepsleep = load_model_Deepsleep(test_dl, base_path)
     # Tạo biểu đồ
    labels = ['ReLU', 'GELU']
    TS_scores = [total_acc_TS, total_acc_TS_gelu]
    CA_scores = [total_acc_CA, total_acc_CA_gelu]
    width = 0.2
    x = list(range(len(labels)))

    fig, ax = plt.subplots()
    rects1 = ax.bar([i - width/2 for i in x], TS_scores, width, label='TS', color = 'b')
    rects2 = ax.bar([i + width/2 for i in x], CA_scores, width, label='CA', color = 'r')

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy by Method and Activation Function')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{:.2f}%'.format(height),  # Định dạng hiển thị số phần trăm với 2 chữ số thập phân
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 2),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    random_number = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
    filename = 'result_' + random_number + '-acc.png'
    plt.savefig(os.path.join('/home/rosa/TestModels/static', filename)) 
    plt.close()


    # Thêm tiêu đề và nhãn cho trục x và y
    plt.plot(trgs_TS)
    plt.title('True Labels Line Chart')
    plt.xlabel('30-s Epoch (120 epochs = 1 hour)')
    plt.ylabel('Sleep stage')
    plt.yticks(range(5), ['W', 'N1', 'N2', 'N3', 'REM'])
    filename = 'result_' + random_number + '-true_line' + '.png'
    plt.savefig(os.path.join('/home/rosa/TestModels/static', filename)) 
    plt.close()

    plt.plot(outs_TS, color='green')
    plt.title('TS-TCC line chart')
    plt.xlabel('30-s Epoch (120 epochs = 1 hour)')
    plt.ylabel('Sleep stage')
    plt.yticks(range(5), ['W', 'N1', 'N2', 'N3', 'REM'])
    filename = 'result_' + random_number + '-TS_pre_line' + '.png'
    plt.savefig(os.path.join('/home/rosa/TestModels/static', filename)) 
    plt.close()

    plt.plot(outs_CA, color='green')
    plt.title('CA-TCC line chart')
    plt.xlabel('30-s Epoch (120 epochs = 1 hour)')
    plt.ylabel('Sleep stage')
    plt.yticks(range(5), ['W', 'N1', 'N2', 'N3', 'REM'])
    filename = 'result_' + random_number + '-_CA_pre_line' + '.png'
    plt.savefig(os.path.join('/home/rosa/TestModels/static', filename)) 
    plt.close()

    
    image_names = os.listdir('static/')
    image_names = [img for img in image_names if img.endswith('.png')]
    return render_template('result.html', image_names=image_names)

if __name__ == '__main__':
    app.run(threaded=True)

