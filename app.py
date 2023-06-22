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
from dataloader.generate import generate_nolabels, generate_withlabels
from main import load_model_TCC, load_model_Attn, load_model_Tiny, load_model_Deepsleep
from dataloader.dataloader_pytorch import data_generator
from dataloader.edf_to_npz import EdfToNpz, EdfToNpz_NoLabels

app = Flask(__name__, template_folder='template')
base_path = ""
data_path = "data"

def delete_files_with_extension(folder_path, extension):
    for file_name in os.listdir(folder_path):
        if file_name.endswith(extension):
            file_path = os.path.join(folder_path, file_name)
            os.remove(file_path)

def preds_chart(preds, name_chart):
    random_number = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
    filename = 'preds' + random_number + '-'+ name_chart +'.png'

    fig = plt.figure(figsize=(10, 2))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(preds, color='red')
    ax.set_title(name_chart)
    # ax.set_xlabel('Epochs')
    ax.set_xticklabels('Epochs', fontsize=10)
    ax.set_ylabel('Sleep stage')
    ax.set_yticks(range(2))
    ax.set_yticklabels(['W', 'N1', 'N2', 'N3', 'REM'], fontsize=10)
    plt.savefig(os.path.join('static', filename)) 
    plt.close()

@app.route('/')
def home():
    return render_template('index.html')

# Define route
@app.route('/predict', methods=['POST'])
def predict():
    # Load datasets
    delete_files_with_extension("/home/rosa/TestModels/data", 'PSG.edf')
    delete_files_with_extension("/home/rosa/TestModels/data", 'Hypnogram.edf')
    delete_files_with_extension("/home/rosa/TestModels/data", '.npz')
    delete_files_with_extension("/home/rosa/TestModels/data", '.pt')
    delete_files_with_extension("/home/rosa/TestModels/static", '.png')

    uploaded_files = request.files.getlist('file')
    print("\n************* uploaded_files ", uploaded_files)
    for file in uploaded_files:
        filename = secure_filename(file.filename)
        print("file ", filename)
        file.save(os.path.join(base_path, data_path , filename))


    EdfToNpz_NoLabels(base_path, data_path)
    generate_nolabels(base_path, "TestModels/data/test_data.npz")
    test_dl = data_generator(str(os.path.join(base_path, "TestModels/data/test_data.pt")), labels=False)

    total_loss, total_acc_TS, outs_TS, trgs = load_model_TCC(test_dl, base_path, method='TS', act_func='GELU', labels=False)
    total_loss, total_acc_CA, outs_CA, trgs = load_model_TCC(test_dl, base_path, method='CA', act_func='GELU', labels=False)
    
    acc_Attn, outs_attn, trgs = load_model_Attn(test_dl, base_path, labels=False)
    acc, f1_score, outs_tiny = load_model_Tiny(test_dl, base_path, act_func = 'GELU', labels=False)
    acc_deepsleep, f1_deepsleep, outs_deepsleep = load_model_Deepsleep(base_path, labels=False)

     # Tạo biểu đồ
    preds_chart(outs_TS, 'TS-TCC')
    preds_chart(outs_CA, 'CA-TCC')
    preds_chart(outs_attn, 'Attn')
    preds_chart(outs_tiny, 'TinySleepNet')
    preds_chart(outs_deepsleep, 'DeepSleepNet')

    image_names = os.listdir('static/')
    image_names = [img for img in image_names if img.endswith('.png')]
    return render_template('user.html', image_names=image_names)

@app.route('/evaluate', methods=['POST'])
def evaluate():
    # Load datasets
    # delete_files_with_extension("/home/rosa/TestModels/data", 'PSG.edf')
    # delete_files_with_extension("/home/rosa/TestModels/data", 'Hypnogram.edf')
    # delete_files_with_extension("/home/rosa/TestModels/data", '.npz')
    # delete_files_with_extension("/home/rosa/TestModels/data", '.pt')
    # delete_files_with_extension("static", '.png')

    # uploaded_files = request.files.getlist('file')
    # print("\n************* uploaded_files ", uploaded_files)
    # for file in uploaded_files:
    #     filename = secure_filename(file.filename)
    #     print("file ", filename)
    #     file.save(os.path.join(base_path, data_path , filename))


    # psg_file = glob.glob(os.path.join(base_path, data_path, "*PSG.edf"))
    # ann_file = glob.glob(os.path.join(base_path, data_path, "*Hypnogram.edf"))
    # raw = mne.io.read_raw_edf(psg_file[0])
    # annotations = mne.read_annotations(ann_file[0])
    # raw.set_annotations(annotations)
    # channel_names = ["EEG Fpz-Cz"]
    # two_meg_chans = raw[channel_names, 0:10000]
    # y_offset = np.array([5e-11, 0])  # just enough to separate the channel traces
    # x = two_meg_chans[1]
    # y = two_meg_chans[0].T + y_offset
    # lines = plt.plot(x, y)
    # plt.legend(lines, channel_names)
    # raw.plot()
    # random_number = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
    # filename = 'result_' + random_number + '-raw_edf.png'
    # plt.savefig(os.path.join('/home/rosa/TestModels/static', filename))
    # plt.close()

    # EdfToNpz(base_path, data_path)
    test_npz = "data/test_data_20.npz"
    generate_withlabels(base_path, test_npz)
    test_pt = data_generator(str(os.path.join(base_path, "test_data.pt")), labels=True)

    # print("\n*****    ReLU    ******")
    total_loss, total_acc_TS, outs_TS, trgs = load_model_TCC(test_pt, base_path, method='TS', act_func='ReLU')
    total_loss, total_acc_CA, outs_CA, trgs = load_model_TCC(test_pt, base_path, method='CA', act_func='ReLU')

    # print("\n*****    GELU    ******")
    total_loss, total_acc_TS, outs_TS, trgs  = load_model_TCC(test_pt, base_path, method='TS', act_func='GELU')
    total_loss, total_acc_CA, outs_CA, trgs  = load_model_TCC(test_pt, base_path, method='CA', act_func='GELU')
    
    total_acc_Attn, outs_attn, trgs = load_model_Attn(test_pt, base_path, labels=True)
    acc_tiny_relu, f1_tiny_relu, outs_tiny = load_model_Tiny(test_npz, base_path, act_func = 'ReLU', labels=True)
    acc_tiny_gelu, f1_tiny_gelu, outs_tiny = load_model_Tiny(test_npz, base_path, act_func = 'GELU', labels=True)
    total_acc_deepsleep, total_f1_deepsleep, outs_deepsleep = load_model_Deepsleep(test_npz, base_path, labels=True)
    
    print("acc_tiny_relu, total_acc_deepsleep ", acc_tiny_relu, total_acc_deepsleep)
    methods = ['TS-TCC', 'CA-TCC', 'Attn', 'Tinysleepnet', 'Deepsleepnet']
    accuracy = [total_acc_TS, total_acc_CA, total_acc_Attn, acc_tiny_relu, total_acc_deepsleep]
    
    random_number = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
    acc_chart = 'result_' + random_number + '-acc.png'

    # Tạo một danh sách số từ 0 đến chiều dài của methods
    x_values = list(range(len(methods)))

    # Vẽ biểu đồ
    plt.figure(figsize=(10, 4))
    plt.bar(x_values, accuracy)
    plt.xticks(x_values, methods)
    plt.xlabel('Methods')
    plt.ylabel('Accuracy')
    plt.title('Evaluation of Methods')
    plt.ylim(0, 100)
    plt.tight_layout()
    # plt.subplots_adjust(bottom=)
    plt.gca().set_yticklabels(['{:.0f}%'.format(x) for x in plt.gca().get_yticks()])  # Format y-axis tick labels as percentages
    plt.savefig(os.path.join(base_path, "static", acc_chart)) 
    plt.close()

    # True Labels Chart
    true_labels_chart = 'result_' + random_number + '-true_line.png'
    fig = plt.figure(figsize=(10, 2))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(trgs, color='red')
    ax.set_title('True Labels')
    ax.set_xlabel('30-s Epoch')
    ax.set_ylabel('Sleep stage')
    ax.set_yticks(range(5))
    ax.set_yticklabels(['W', 'N1', 'N2', 'N3', 'REM'])
    plt.savefig(os.path.join(os.path.join(base_path, "static", true_labels_chart)))
    plt.close()

    # Vẽ chú thích
    plt.figure(figsize=(10, 1))
    plt.axis('off')
    plt.text(0, 0.5, 'Danh sách nhãn đúng là: ' + str(trgs), fontsize=12, verticalalignment='center')
    plt.savefig(os.path.join(os.path.join(base_path, "static", 'true_labels_legend.png')))
    plt.close()


    def preds_chart_5model(outs_TS, outs_CA, outs_attn, outs_tiny, outs_deepsleep, name_chart):
        random_number = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
        filename = 'preds' + random_number + '-' + name_chart + '.png'

        fig = plt.figure(figsize=(10, 2))
        ax = fig.add_subplot(1, 1, 1)

        ax.plot(outs_TS, label='TS-TCC', color="green")
        ax.plot(outs_CA, label='CA-TCC', color="blue")
        ax.plot(outs_attn, label='Attn', color="chocolate")
        ax.plot(outs_tiny, label='TinySleepNet', color="gold")
        ax.plot(outs_deepsleep, label='DeepSleepNet', color="darkorchid")

        ax.set_title(name_chart)
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Sleep stage')

        # ax.set_xticklabels('Epochs', fontsize=10)
        ax.set_yticks(range(2))
        ax.set_yticklabels(['W', 'N1', 'N2', 'N3', 'REM'], fontsize=10)

        plt.savefig(os.path.join('static', filename))
        plt.close()


    preds_chart_5model(outs_TS, outs_CA, outs_attn, outs_tiny, outs_deepsleep, 'sum-result')
    # preds_chart(outs_CA, 'CA-TCC')
    # preds_chart(outs_attn, 'Attn')
    # preds_chart(outs_tiny, 'TinySleepNet')
    # preds_chart(outs_deepsleep, 'DeepSleepNet')

    # fig = plt.figure(figsize=(10, 5))
    # ax = fig.add_subplot(1, 1, 1)

    # # Vẽ đường cho mỗi biểu đồ cũ
    # ax.plot(outs_TS, label='TS-TCC')
    # ax.plot(outs_CA, label='CA-TCC')
    # ax.plot(outs_attn, label='Attn')
    # ax.plot(outs_tiny, label='TinySleepNet')
    # ax.plot(outs_deepsleep, label='DeepSleepNet')
    # ax.set_title('Biểu đồ của các mô hình học sâu')
    # ax.set_xlabel('Epochs')
    # ax.set_ylabel('Sleep stage')

    # # Hiển thị chú thích
    # ax.legend()

    # # Hiển thị biểu đồ
    # plt.show()

    
    image_names = os.listdir('static/')
    image_names = [img for img in image_names if img.endswith('.png')]
    return render_template('dev.html', image_names=image_names)

if __name__ == '__main__':
    app.run(port=1113)

