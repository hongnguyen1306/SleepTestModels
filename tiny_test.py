import argparse
import glob
import importlib
import os
import numpy as np
import shutil
import sklearn.metrics as skmetrics
import torch
from models.pytorch_models.Tiny_models.model import Model
from models.pytorch_models.Tiny_models.network import TinySleepNet

from dataloader.dataloader_tiny import load_data, get_subject_files
from models.pytorch_models.Tiny_models.minibatching import (iterate_batch_multiple_seq_minibatches)
from config_files.pytorch_configs.tiny_configs import predict


def compute_performance(cm):
    """Computer performance metrics from confusion matrix.

    It computers performance metrics from confusion matrix.
    It returns:
        - Total number of samples
        - Number of samples in each class
        - Accuracy
        - Macro-F1 score
        - Per-class precision
        - Per-class recall
        - Per-class f1-score
    """

    tp = np.diagonal(cm).astype(np.float64)
    tpfp = np.sum(cm, axis=0).astype(np.float64) # sum of each col
    tpfn = np.sum(cm, axis=1).astype(np.float64) # sum of each row
    acc = np.sum(tp) / np.sum(cm)
    precision = tp / tpfp
    recall = tp / tpfn
    f1 = (2 * precision * recall) / (precision + recall)
    mf1 = np.mean(f1)

    total = np.sum(cm)
    n_each_class = tpfn

    return total, n_each_class, acc, mf1, precision, recall, f1


def predict(
    config_file,
    model_dir,
    output_dir,
    log_file,
    use_best=True,
):

    spec = importlib.util.spec_from_file_location("*", config_file)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    config = config.predict

    # Create output directory for the specified fold_idx
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    subject_files = glob.glob(os.path.join(config["data_dir"], "*.npz"))

    # Add dummy class weights
    config["class_weights"] = np.ones(config["n_classes"], dtype=np.float32)

    trues = []
    preds = []
    s_trues = []
    s_preds = []

    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    model = Model(
        config=config,
        output_dir=output_dir,
        use_rnn=True,
        testing=True,
        use_best=use_best,
        device=device
    )


    test_x, test_y, _ = load_data(subject_files)
    for night_idx, night_data in enumerate(zip(test_x, test_y)):
                # Create minibatches for testing
                night_x, night_y = night_data
                test_minibatch_fn = iterate_batch_multiple_seq_minibatches(
                    [night_x],
                    [night_y],
                    batch_size=config["batch_size"],
                    seq_length=config["seq_length"],
                    shuffle_idx=None,
                    augment_seq=False,
                )
                # Evaluate
                test_outs = model.evaluate_with_dataloader(test_minibatch_fn)  # 预测入口在这里

                trues.extend(test_outs["test/trues"])
                preds.extend(test_outs["test/preds"])

    # Get corresponding
    acc = skmetrics.accuracy_score(y_true=trues, y_pred=preds)
    f1_score = skmetrics.f1_score(y_true=trues, y_pred=preds, average="weighted")
    cm = skmetrics.confusion_matrix(y_true=trues, y_pred=preds, labels=[0,1,2,3,4])
    
    preds = np.array(preds)
    trues = np.array(trues)
    preds = preds.astype(int)
    trues = trues.astype(int)
    # Tính tỉ lệ dự đoán đúng cho từng nhãn
    accuracy = {}
    for label in range(5):
        # Lấy chỉ mục các mẫu thuộc nhãn label
        indices = np.where(trues == label)[0]
        
        # Đếm số lượng dự đoán đúng cho nhãn label
        correct_predictions = np.sum(preds[indices.astype(int)] == trues[indices.astype(int)])
        
        # Tính tỉ lệ dự đoán đúng
        accuracy[label] = correct_predictions / len(indices)

    # In kết quả
    print("=====         TinySleepNet        =====")
    for label, acc in accuracy.items():
        print(f"Nhãn {label}: Tỉ lệ dự đoán đúng = {acc}")

    # print("")
    # print("=====         Tinysleep        =====")
    # print("n={}, acc={:.1f}, mf1={:.1f}".format(len(preds), acc*100.0, f1_score*100.0))

    # print(">> Confusion Matrix")
    # print(cm)

    # metrics = compute_performance(cm=cm)
    # print("Total: {}".format(metrics[0]))
    # print("Number of samples from each class: {}".format(metrics[1]))
    # print("Accuracy: {:.1f}".format(metrics[2]*100.0))
    # print("Macro F1-Score: {:.1f}".format(metrics[3]*100.0))
    # print("Per-class Precision: " + " ".join(["{:.1f}".format(m*100.0) for m in metrics[4]]))
    # print("Per-class Recall: " + " ".join(["{:.1f}".format(m*100.0) for m in metrics[5]]))
    # print("Per-class F1-Score: " + " ".join(["{:.1f}".format(m*100.0) for m in metrics[6]])
