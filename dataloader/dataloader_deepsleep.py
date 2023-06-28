import os
import glob
import numpy as np

from models.Deepsleep_models.sleep_stage import print_n_samples_each_class
from models.Deepsleep_models.utils import get_balance_class_oversample

import re

class SeqDataLoader(object):

    def __init__(self, data_dir, train_dir, val_dir, n_folds, fold_idx):
            self.data_dir = data_dir
            self.train_dir = train_dir
            self.val_dir = val_dir
            self.n_folds = n_folds
            self.fold_idx = fold_idx

    def _load_npz_file(self, npz_file):
        """Load data and labels from a npz file."""
        with np.load(npz_file) as f:
            data = f["x"]
            labels = f["y"]
            sampling_rate = f["fs"]
        return data, labels, sampling_rate

    def _load_npz_list_files(self, npz_files):
        """Load data and labels from list of npz files."""
        data = []
        labels = []
        fs = None
        for npz_f in npz_files:
            tmp_data, tmp_labels, sampling_rate = self._load_npz_file(npz_f)
            if fs is None:
                fs = sampling_rate
            elif fs != sampling_rate:
                raise Exception("Found mismatch in sampling rate.")

            # Reshape the data to match the input of the model - conv2d
            # tmp_data = np.squeeze(tmp_data)
            tmp_data = np.squeeze(tmp_data, axis=(2,))  # Remove the axis=2 dimension
            tmp_data = tmp_data[:, :, np.newaxis, np.newaxis]
            
            # # Reshape the data to match the input of the model - conv1d
            # tmp_data = tmp_data[:, :, np.newaxis]

            # Casting
            tmp_data = tmp_data.astype(np.float32)
            tmp_labels = tmp_labels.astype(np.int32)

            data.append(tmp_data)
            labels.append(tmp_labels)

        return data, labels

    def _load_cv_data(self, list_files):
        """Load sequence training and cross-validation sets."""
        # Split files for training and validation sets
        val_files = np.array_split(list_files, self.n_folds)
        train_files = np.setdiff1d(list_files, val_files[self.fold_idx])

        # Load a npz file
        print("Load training set:")
        data_train, label_train = self._load_npz_list_files(train_files)
        print(" ")
        print("Load validation set:")
        data_val, label_val = self._load_npz_list_files(val_files[self.fold_idx])
        print(" ")

        return data_train, label_train, data_val, label_val

    @staticmethod
    def load_subject_data(data_dir, subject_idx):
        # Remove non-mat files, and perform ascending sort
        subject_files = []
        subject_files = [data_dir]
        # subject_files.append(npz_file)
        
        if len(subject_files) == 0 or len(subject_files) > 2:
            raise Exception("Invalid file pattern")

        def load_npz_file(npz_file):
            """Load data and labels from a npz file."""
            with np.load(npz_file) as f:
                data = f["x"]
                labels = f["y"]
                sampling_rate = f["fs"]
            return data, labels, sampling_rate

        def load_npz_list_files(npz_files):
            """Load data and labels from list of npz files."""
            data = []
            labels = []
            fs = None
            for npz_f in npz_files:
                tmp_data, tmp_labels, sampling_rate = load_npz_file(npz_f)
                if fs is None:
                    fs = sampling_rate
                elif fs != sampling_rate:
                    raise Exception("Found mismatch in sampling rate.")

                # Reshape the data to match the input of the model - conv2d
                # tmp_data = np.squeeze(tmp_data)
                tmp_data = np.squeeze(tmp_data, axis=(2,))  # Remove the axis=2 dimension
                tmp_data = tmp_data[:, :, np.newaxis, np.newaxis]
                
                # # Reshape the data to match the input of the model - conv1d
                # tmp_data = tmp_data[:, :, np.newaxis]

                # Casting
                tmp_data = tmp_data.astype(np.float32)
                tmp_labels = tmp_labels.astype(np.int32)

                data.append(tmp_data)
                labels.append(tmp_labels)

            return data, labels

        data, labels = load_npz_list_files(subject_files)

        return data, labels
    
    @staticmethod
    def load_subject_nolabels(data_dir, subject_idx):
        # Remove non-mat files, and perform ascending sort
        subject_files = []
        # subject_files = glob.glob(data_dir + ".npz")
        subject_files = [data_dir]

        # subject_files.append(npz_file)
        print("subject_files Deepsleep  ", subject_files)
        
        if len(subject_files) == 0 or len(subject_files) > 2:
            raise Exception("Invalid file pattern")

        def load_npz_file(npz_file):
            """Load data and labels from a npz file."""
            with np.load(npz_file) as f:
                data = f["x"]
                sampling_rate = f["fs"]
            return data, sampling_rate

        def load_npz_list_files(npz_files):
            """Load data and labels from list of npz files."""
            data = []
            fs = None
            for npz_f in npz_files:
                tmp_data, sampling_rate = load_npz_file(npz_f)
                if fs is None:
                    fs = sampling_rate
                elif fs != sampling_rate:
                    raise Exception("Found mismatch in sampling rate.")

                # Reshape the data to match the input of the model - conv2d
                tmp_data = np.squeeze(tmp_data, axis=(2,))
                tmp_data = tmp_data[:, :, np.newaxis, np.newaxis]
                
                # # Reshape the data to match the input of the model - conv1d
                # tmp_data = tmp_data[:, :, np.newaxis]

                # Casting
                tmp_data = tmp_data.astype(np.float32)

                data.append(tmp_data)

            return data

        data = load_npz_list_files(subject_files)

        return data