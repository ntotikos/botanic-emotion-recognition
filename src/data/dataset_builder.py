""" Build custom datasets. """
import pickle

from src.utils.constants import DATASETS_DIR, EKMAN_NEUTRAL_TO_INT_DICT, INT_TO_EKMAN_NEUTRAL_DICT
from torch.utils.data import DataLoader, random_split, TensorDataset
import torch
import numpy as np
import pandas as pd
import warnings


class EkmanDataset:
    def __init__(self, data_path):
        """
        :param data_path: Path to pickle file with data point dictionaries.
        """
        with open(data_path, 'rb') as file:
            raw_data = pickle.load(file)

        self.raw_data = raw_data
        self.dataset = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.batch_size = 0

    def __len__(self):
        return len(self.raw_data)

    def get_data_and_labels(self):
        """
        """
        wav_slices = []
        labels = []
        for segment in self.raw_data:
            wav_slices.append(segment["wav_slice"])
            labels.append(self.map_label_to_int(segment["label"]))

        wav_slices = torch.tensor(np.array(wav_slices), dtype=torch.float32)
        labels = torch.tensor(np.array(labels), dtype=torch.long)
        self.dataset = TensorDataset(wav_slices, labels)

    def get_data_and_labels_without_neutral(self):
        """
        TODO: Create new class EkmanNeutral to account for the Neutral class
        """
        wav_slices = []
        labels = []
        for segment in self.raw_data:
            if segment["label"] != "Neutral":
                wav_slices.append(segment["wav_slice"])
                labels.append(self.map_label_to_int(segment["label"]))

        wav_slices = torch.tensor(np.array(wav_slices), dtype=torch.float32)
        labels = torch.tensor(np.array(labels), dtype=torch.long)
        self.dataset = TensorDataset(wav_slices, labels)

    def get_labels(self):
        pass

    def normalize_samples(self, normalization="per-sample"):
        # TODO: Test calculation and broadcast of mean and std_dev
        if normalization == "per-sample":
            data_tensor, labels_tensor = self.dataset.tensors
            mean = torch.mean(data_tensor, dim=1, keepdim=True)
            std_dev = torch.std(data_tensor, dim=1, keepdim=True)
            standardized_data = (data_tensor - mean) / (std_dev + 1e-8)  # smoothing term to prevent zero division
        elif normalization == "per-feature":  # i.e. per column
            data_tensor, labels_tensor = self.dataset.tensors
            mean = torch.mean(data_tensor, dim=0)
            std_dev = torch.std(data_tensor, dim=0)
            standardized_data = (data_tensor - mean) / (std_dev + 1e-8)  # smoothing term to prevent zero division
        elif normalization == "global":
            data_tensor, labels_tensor = self.dataset.tensors
            mean = torch.mean(data_tensor)
            std_dev = torch.std(data_tensor)
            standardized_data = (data_tensor - mean) / (std_dev + 1e-8)  # smoothing term to prevent zero division
        elif normalization == "min-max-scaling":
            data_tensor, labels_tensor = self.dataset.tensors
            minima, _ = torch.min(data_tensor, dim=0)
            maxima, _ = torch.max(data_tensor, dim=0)
            standardized_data = (data_tensor - minima) / (maxima - minima + 1e-8)

        self.dataset = TensorDataset(standardized_data, labels_tensor)

    @staticmethod
    def map_label_to_int(emotion: str):
        return EKMAN_NEUTRAL_TO_INT_DICT[emotion]

    def split_dataset_into_train_val_test(self, train_split: float = 0.8, val_split: float = 0.1, stratify=False):
        # TODO: This should be an abstract method in a super class.
        # TODO 2: call this method `random_split_dataset_...` and create other `split` function that makes sure that
        # split is done in a way that one set is not getting all rare samples (that are not a lot in the dataset)
        train_size = int(train_split * len(self.dataset))
        val_size = int(val_split * len(self.dataset))
        test_size = len(self.dataset) - train_size - val_size

        self.train_data, self.val_data, self.test_data = random_split(self.dataset, [train_size, val_size, test_size])

        # TODO: implement.
        if not stratify:  # randomly split dataset
            pass
        elif stratify:  # stratified split, i.e. all subsets have similar class distribution
            pass

    @staticmethod
    def get_label_distribution(dataloader):
        # TODO: dataloader_labels = get_labels(dataloader)
        dataloader_labels = []
        for batch_data, batch_labels in dataloader:
            dataloader_labels.extend([int(i) for i in batch_labels])
        df = pd.DataFrame([map_int_to_label(j) for j in dataloader_labels], columns=['Class'])
        class_counts = df['Class'].value_counts()
        return class_counts

    def create_data_loader(self, batch_size=32):
        # TODO: How to handle case when there are only 29/32 samples left? Padding or doesn't matter?
        train_loader = DataLoader(self.train_data, batch_size)
        val_loader = DataLoader(self.val_data, batch_size)
        test_loader = DataLoader(self.test_data, batch_size)

        return train_loader, val_loader, test_loader


def map_int_to_label(emotion: int):
    return INT_TO_EKMAN_NEUTRAL_DICT[emotion]


if __name__ == "__main__":
    path_to_pickle = DATASETS_DIR / "sdm_2023-01-10_team_01_8333_9490.pkl"
    dataset = EkmanDataset(path_to_pickle)
    dataset.get_data_and_labels()
    dataset.split_dataset_into_train_val_test()

    train_dataloader, val_dataloader, test_dataloader = dataset.create_data_loader()

    output = dataset.get_label_distribution(train_dataloader)
    print("OUTPUT:", output)


    #for i, j in train_dataloader:
    #    print(i)
