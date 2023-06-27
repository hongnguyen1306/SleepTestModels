import glob
from flask import Flask, request, render_template, make_response, jsonify
# from flask_socketio import SocketIO, emit
import json
import os
import numpy as np
import pandas as pd
from datetime import datetime
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import string
import mne
import threading
from dataloader.generate import generate_nolabels, generate_withlabels
from main import load_model_TCC, load_model_Attn, load_model_Tiny, load_model_Deepsleep
from dataloader.dataloader_pytorch import data_generator
from dataloader.edf_to_npz import EdfToNpz, EdfToNpz_NoLabels
from dataloader.edf_to_full_npz import EdfToFullNpz


initial_chart_data = {
        # "AttnSleep": 0,
        # "CA-TCC": 0,
        # "DeepSleepNet": 0,
        # "TS-TCC": 0,
        # "TinySleepNet": 0
    }

base_path = ""
data_path = "data"
app = Flask(__name__, template_folder=os.path.join(base_path,'template'))
# socketio = SocketIO(app)

def delete_files_with_extension(folder_path, extension):
    for file_name in os.listdir(folder_path):
        if file_name.endswith(extension):
            file_path = os.path.join(folder_path, file_name)
            os.remove(file_path)

def delete_files_async(folder_path, extension):
    thread = threading.Thread(target=delete_files_with_extension, args=(folder_path, extension))
    thread.start()

def preds_chart(preds, name_chart):
    random_number = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
    filename = 'preds' + random_number + '-'+ name_chart +'.png'
    fig = plt.figure(figsize=(10, 3))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(preds, color='green')
    ax.set_title(name_chart)
    ax.set_ylabel('Sleep Stage')
    ax.set_xlabel('Epoch')
    ax.set_yticks(range(5))
    ax.set_yticklabels(['W', 'N1', 'N2', 'N3', 'REM'])

    plt.savefig(os.path.join(base_path, "static", filename))
    plt.close()

def raw_chart(base_path, data_path):
    random_number = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
    filename = 'file' + str(random_number) + '-' + 'Fpz_Cz.png'
    psg_file = glob.glob(os.path.join(base_path, data_path, "*PSG.edf"))
    raw = mne.io.read_raw_edf(psg_file[0])
    channel_names = ["EEG Fpz-Cz"]
    start_time = 0 
    end_time = 30

    start_idx = int(start_time * raw.info['sfreq'])
    end_idx = int(end_time * raw.info['sfreq'])

    segment = raw[channel_names, start_idx:end_idx]
    y_offset = np.array([5e-11, 0])
    x = segment[1]
    y = segment[0].T + y_offset

    # Plot and save the image of the channel
    fig = plt.figure(figsize=(10, 3))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y)
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')
    ax.set_title('Channel: ' + channel_names[0] + ' (Segment: ' + str(start_time) + 's to ' + str(end_time) + 's)')
    plt.savefig(os.path.join(base_path, "static", filename))
    plt.close()


def acc_chart(methods, accuracy, chart_filename):
    x_values = list(range(len(methods)))

    plt.figure(figsize=(10, 4))
    plt.bar(x_values, accuracy, align='center')
    plt.xticks(x_values, methods, horizontalalignment='center')
    plt.xlabel('Methods')
    plt.ylabel('Accuracy')
    plt.title('Evaluation of Methods')
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.gca().set_yticklabels(['{:.0f}%'.format(x) for x in plt.gca().get_yticks()])  # Format y-axis tick labels as percentages
    plt.savefig(os.path.join(base_path, "static", chart_filename)) 
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
    ax.set_yticks(range(6))
    ax.set_yticklabels(['W', 'N1', 'N2', 'N3', 'REM'], fontsize=10)

    plt.savefig(os.path.join('static', filename))
    plt.close()


@app.route('/', endpoint='home_endpoint')
def home():
    pass
    return render_template('index.html')

# Define route
@app.route('/predict', methods=['POST'])
def predict():
    # Load datasets
    
    delete_files_async(os.path.join(base_path, "data"), 'PSG.edf')
    delete_files_async(os.path.join(base_path, "data"), 'Hypnogram.edf')
    delete_files_async(os.path.join(base_path, "data"), '.npz')
    delete_files_async(os.path.join(base_path, "data"), '.pt')
    delete_files_async(os.path.join(base_path, "static"), '.png')

    uploaded_files = request.files.getlist('file')
    print("\n************* uploaded_files ", uploaded_files)

    for file in uploaded_files:
        filename = secure_filename(file.filename)
        print("file ", filename)
        file.save(os.path.join(base_path, data_path , filename))

    raw_chart(base_path=base_path,data_path=data_path)

    EdfToNpz_NoLabels(base_path, data_path)
    test_npz = "data/test_data.npz"
    generate_nolabels(base_path, "data/test_data.npz")
    test_dl = data_generator(str(os.path.join(base_path, "data/test_data.pt")), labels=False)

    total_loss, total_acc_TS, outs_TS, trgs = load_model_TCC(test_dl, base_path, method='TS', act_func='GELU', labels=False)
    total_loss, total_acc_CA, outs_CA, trgs = load_model_TCC(test_dl, base_path, method='CA', act_func='GELU', labels=False)
    
    acc_Attn, outs_attn, trgs = load_model_Attn(test_dl, base_path, labels=False)
    acc, f1_score, outs_tiny = load_model_Tiny(test_npz, base_path, act_func = 'ReLU', labels=False)
    acc_deepsleep, f1_deepsleep, outs_deepsleep = load_model_Deepsleep(test_npz, base_path, labels=False)

    preds_chart_5model(outs_TS, outs_CA, outs_attn, outs_tiny, outs_deepsleep, 'sum-result')
    predicts = {
            'true_labels': [],
            'TS-TCC_gelu': [],
            'TS-TCC': outs_TS.tolist(),
            'CA-TCC': outs_CA.tolist(),
            'CA-TCC_gelu': [],
            'outs_attn': outs_attn.tolist(),
            'outs_tiny_ReLU': outs_tiny.tolist(),
            'outs_tiny_GELU': [],
            'outs_deepsleep': outs_deepsleep.tolist()
        }

    image_names = os.listdir('static/')
    image_names = [img for img in image_names if img.endswith('.png')]
    return render_template('user.html', image_names=image_names, predicts_json=json.dumps(predicts))

@app.route('/evaluate', methods=['POST'])
def evaluate():
    # Load datasets
    delete_files_async(os.path.join(base_path, "data"), '-PSG.edf')
    delete_files_async(os.path.join(base_path, "data"), '-Hypnogram.edf')
    delete_files_async(os.path.join(base_path, "data"), '.npz')
    delete_files_async(os.path.join(base_path, "data"), '.pt')
    delete_files_async(os.path.join(base_path, "static"), '.png')

    uploaded_files = request.files.getlist('file')

    for file in uploaded_files:
        filename = secure_filename(file.filename)
        print("file ", filename)
        file.save(os.path.join(base_path, data_path , filename))

    raw_chart(base_path=base_path,data_path=data_path)
    EdfToFullNpz(base_path=base_path, data_dir=data_path)

    test_npz = 'data/test_data.npz'
    generate_withlabels(base_path, test_npz)
    test_pt = data_generator(os.path.join(base_path, "data/test_data.pt"), labels=True)

    # print("\n*****    ReLU    ******")
    total_loss, total_acc_TS, outs_TS, true_labels = load_model_TCC(test_pt, base_path, method='TS', act_func='ReLU')
    total_loss, total_acc_CA, outs_CA, trgs = load_model_TCC(test_pt, base_path, method='CA', act_func='ReLU')

    # print("\n*****    GELU    ******")
    total_loss, gelu_acc_TS, outs_TS_G, trgs  = load_model_TCC(test_pt, base_path, method='TS', act_func='GELU')
    total_loss, gelu_acc_CA, outs_CA_G, trgs  = load_model_TCC(test_pt, base_path, method='CA', act_func='GELU')
    
    total_acc_Attn, outs_attn , trgs = load_model_Attn(test_pt, base_path, labels=True)
    acc_tiny_relu, f1_tiny_relu, outs_tiny_ReLU = load_model_Tiny(test_npz, base_path, act_func = 'ReLU', labels=True)
    acc_tiny_gelu, f1_tiny_gelu, outs_tiny_GELU = load_model_Tiny(test_npz, base_path, act_func = 'GELU', labels=True)
    total_acc_deepsleep, total_f1_deepsleep, outs_deepsleep = load_model_Deepsleep(test_npz, base_path, labels=True)
    
    data = np.load(os.path.join(base_path, "data/test_data.npz"))
    results = {}

    if len(data['y']) < 10:
        stage_mapping = {
            0: "Giai đoạn Thức",
            1: "Giai đoạn 1",
            2: "Giai đoạn 2",
            3: "Giai đoạn 3",
            4: "REM"
        }

        trgs_labels = [stage_mapping[trg] for trg in trgs]
        outs_TS_G_labels = [stage_mapping[out] for out in outs_TS_G]
        outs_TS_labels = [stage_mapping[out] for out in outs_TS]
        outs_CA_labels = [stage_mapping[out] for out in outs_CA]
        outs_CA_G_labels = [stage_mapping[out] for out in outs_CA_G]
        outs_attn_labels = [stage_mapping[out] for out in outs_attn]
        outs_tiny_ReLU_labels = [stage_mapping[out] for out in outs_tiny_ReLU]
        outs_tiny_GELU_labels = [stage_mapping[out] for out in outs_tiny_ReLU]
        outs_deepsleep_labels = [stage_mapping[out] for out in outs_deepsleep]

        results = {
            'true_label': trgs_labels,
            'TS-TCC_gelu': outs_TS_G_labels,
            'TS-TCC': outs_TS_labels,
            'CA-TCC': outs_CA_labels,
            'CA-TCC_gelu': outs_CA_G_labels,
            'outs_attn': outs_attn_labels,
            'outs_tiny_ReLU': outs_tiny_ReLU_labels,
            'outs_tiny_GELU': outs_tiny_GELU_labels,
            'outs_deepsleep': outs_deepsleep_labels
        }

        image_names = os.listdir('static/')
        image_names = [img for img in image_names if img.endswith('.png')]

        return render_template('predictOneLabel.html', results=results, image_names=image_names)
    else:
        random_number = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))

        methods_acc_relu = ['TS-TCC', 'CA-TCC', 'Attn', 'Tinysleepnet', 'Deepsleepnet']
        accuracy_relu = [total_acc_TS, total_acc_CA, total_acc_Attn, acc_tiny_relu, total_acc_deepsleep]
        acc_chart_relu = 'result_' + random_number + '-acc_relu.png'
        acc_chart(methods_acc_relu, accuracy_relu, acc_chart_relu)

        methods_acc_gelu = ['TS-TCC', 'CA-TCC', 'Tinysleepnet']
        accuracy_gelu = [gelu_acc_TS, gelu_acc_CA, acc_tiny_gelu]
        acc_chart_gelu = 'result_' + random_number + '-acc_gelu.png'
        acc_chart(methods_acc_gelu, accuracy_gelu, acc_chart_gelu)

        # True Labels Chart
        true_labels_chart = 'result_' + random_number + '-true_line.png'
        fig = plt.figure(figsize=(10, 2))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(trgs, color='red')
        ax.set_title('True Labels')
        ax.set_xlabel('30-s Epoch')
        ax.set_ylabel('Sleep stage')
        ax.set_yticks(range(6))
        ax.set_yticklabels(['W', 'N1', 'N2', 'N3', 'REM'])
        plt.savefig(os.path.join(os.path.join(base_path, "static", true_labels_chart)))
        plt.close()

        # Vẽ chú thích
        plt.figure(figsize=(10, 1))
        plt.axis('off')
        plt.text(0, 0.5, 'Danh sách nhãn đúng là: ' + str(true_labels), fontsize=12, verticalalignment='center')
        plt.savefig(os.path.join(os.path.join(base_path, "static", str(random_number) + '-true_labels_legend.png')))
        plt.close()
        preds_chart_5model(outs_TS, outs_CA, outs_attn, outs_tiny_ReLU, outs_deepsleep, 'sum-result')

        predicts = {
            'true_labels': true_labels.tolist(),
            'TS-TCC_gelu': outs_TS_G.tolist(),
            'TS-TCC': outs_TS.tolist(),
            'CA-TCC': outs_CA.tolist(),
            'CA-TCC_gelu': outs_CA_G.tolist(),
            'outs_attn': outs_attn.tolist(),
            'outs_tiny_ReLU': outs_tiny_ReLU.tolist(),
            'outs_tiny_GELU': outs_tiny_GELU.tolist(),
            'outs_deepsleep': outs_deepsleep.tolist()
        }

            ###==================================================================================
    image_names = os.listdir('static/')
    image_names = [img for img in image_names if img.endswith('.png')]
    return render_template('evaluate.html', image_names=image_names,  predicts_json=json.dumps(predicts))
    # # Route cho trang HTML
    # @app.route('/evaluate2')
    # def index():
    #     return render_template('evaluate.html')

    # # Route để cung cấp dữ liệu biểu đồ ban đầu
    # @app.route('/initial-chart-data', methods=['GET'])
    # def get_initial_chart_data():
    #     return jsonify(initial_chart_data)

@app.route('/update-chart', methods=['POST'])
def update_chart():
    getData = request.get_json()
    predicts = getData['predicts']
    selected_values = getData['selectedValues']

    true_label = predicts['true_labels']
    outs_TS_G = predicts['TS-TCC_gelu']
    outs_TS = predicts['TS-TCC']
    outs_CA = predicts['CA-TCC']
    outs_CA_G = predicts['CA-TCC_gelu']
    outs_attn = predicts['outs_attn']
    outs_tiny_ReLU = predicts['outs_tiny_ReLU']
    outs_tiny_GELU = predicts['outs_tiny_GELU']
    outs_deepsleep = predicts['outs_deepsleep']

    labels = ['W', 'N1', 'N2', 'N3', 'REM']

    # plt.figure(figsize=(25, 5))
    initial_chart_data = {}

    # Update the initial_chart_data based on selected checkboxes
    for value in selected_values:
        if value == "0":
            temp = true_label
            for index in range(0,len(temp)):
                if temp[index]==0:
                    temp[index]='W'
                elif temp[index]==1:
                    temp[index]='N1'
                elif temp[index]==2:
                    temp[index]='N2'
                elif temp[index]==3:
                    temp[index]='N3'
                elif temp[index]==4:
                    temp[index]='REM'
            initial_chart_data["Nhãn đúng"] = temp
        elif value == "1":
            temp = outs_attn
            for index in range(0,len(temp)):
                if temp[index]==0:
                    temp[index]='W'
                elif temp[index]==1:
                    temp[index]='N1'
                elif temp[index]==2:
                    temp[index]='N2'
                elif temp[index]==3:
                    temp[index]='N3'
                elif temp[index]==4:
                    temp[index]='REM'

            initial_chart_data["AttnSleep"] = temp
        elif value == "2":
            temp = outs_CA
            for index in range(0,len(temp)):
                if temp[index]==0:
                    temp[index]='W'
                elif temp[index]==1:
                    temp[index]='N1'
                elif temp[index]==2:
                    temp[index]='N2'
                elif temp[index]==3:
                    temp[index]='N3'
                elif temp[index]==4:
                    temp[index]='REM'

            initial_chart_data["CA-TCC"] = temp
        elif value == "3":
            temp = outs_deepsleep
            for index in range(0,len(temp)):
                if temp[index]==0:
                    temp[index]='W'
                elif temp[index]==1:
                    temp[index]='N1'
                elif temp[index]==2:
                    temp[index]='N2'
                elif temp[index]==3:
                    temp[index]='N3'
                elif temp[index]==4:
                    temp[index]='REM'
            initial_chart_data["DeepSleepNet"] = temp
        elif value == "4":
            temp = outs_TS
            for index in range(0,len(temp)):
                if temp[index]==0:
                    temp[index]='W'
                elif temp[index]==1:
                    temp[index]='N1'
                elif temp[index]==2:
                    temp[index]='N2'
                elif temp[index]==3:
                    temp[index]='N3'
                elif temp[index]==4:
                    temp[index]='REM'
            initial_chart_data["TS-TCC"] = temp
        elif value == "5":
            temp = outs_tiny_ReLU
            for index in range(0,len(temp)):
                if temp[index]==0:
                    temp[index]='W'
                elif temp[index]==1:
                    temp[index]='N1'
                elif temp[index]==2:
                    temp[index]='N2'
                elif temp[index]==3:
                    temp[index]='N3'
                elif temp[index]==4:
                    temp[index]='REM'
            initial_chart_data["TinySleepNet"] = temp
        elif value == "6":
            temp = outs_TS_G
            for index in range(0,len(temp)):
                if temp[index]==0:
                    temp[index]='W'
                elif temp[index]==1:
                    temp[index]='N1'
                elif temp[index]==2:
                    temp[index]='N2'
                elif temp[index]==3:
                    temp[index]='N3'
                elif temp[index]==4:
                    temp[index]='REM'
            initial_chart_data["TS-TCC GELU"] = temp
        elif value == "7":
            temp = outs_CA_G
            for index in range(0,len(temp)):
                if temp[index]==0:
                    temp[index]='W'
                elif temp[index]==1:
                    temp[index]='N1'
                elif temp[index]==2:
                    temp[index]='N2'
                elif temp[index]==3:
                    temp[index]='N3'
                elif temp[index]==4:
                    temp[index]='REM'
            initial_chart_data["CA-TCC GELU"] = temp
        elif value == "8":
            temp = outs_tiny_GELU
            for index in range(0,len(temp)):
                if temp[index]==0:
                    temp[index]='W'
                elif temp[index]==1:
                    temp[index]='N1'
                elif temp[index]==2:
                    temp[index]='N2'
                elif temp[index]==3:
                    temp[index]='N3'
                elif temp[index]==4:
                    temp[index]='REM'
            initial_chart_data["TinySleepNet GELU"] = temp

    return jsonify({'labels': labels, 'data': initial_chart_data})


if __name__ == '__main__':
    app.run(port=8080)

