""" Build custom datasets. """
from abc import ABC, abstractmethod
import pickle

from src.utils.constants import DATASETS_DIR, EKMAN_NEUTRAL_TO_INT_DICT
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
import torch
import numpy as np


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
            print(segment)
            wav_slices.append(segment["wav_slice"])
            labels.append(self.map_label_to_int(segment["label"]))

        wav_slices = torch.tensor(np.array(wav_slices))  # Converting list into tensor faster as numpy array.
        labels = torch.tensor(np.array(labels))
        print("Labels:", labels)
        self.dataset = TensorDataset(wav_slices, labels)

    def get_labels(self):
        pass

    def normalize_samples(self):
        # TODO: implement normalization. Mean. Std.
        pass

    @staticmethod
    def map_label_to_int(emotion: str):
        return EKMAN_NEUTRAL_TO_INT_DICT[emotion]

    def split_dataset_into_train_val_test(self, train_split: float = 0.8, val_split: float = 0.1):
        # TODO: This should be an abstract method in a super class.
        # TODO 2: call this method `random_split_dataset_...` and create other `split` function that makes sure that
        # split is done in a way that one set is not getting all rare samples (that are not a lot in the dataset)
        train_size = int(train_split * len(self.dataset))
        val_size = int(val_split * len(self.dataset))
        test_size = len(self.dataset) - train_size - val_size

        self.train_data, self.val_data, self.test_data = random_split(self.dataset, [train_size, val_size, test_size])

    @staticmethod
    def get_label_distribution():
        # TODO: implement method to plot number of samples per class
        pass

    def create_data_loader(self, batch_size=32):
        # TODO: How to handle case when there are only 29/32 samples left? Padding or doesn't matter?
        train_loader= DataLoader(self.train_data, batch_size)
        val_loader = DataLoader(self.val_data, batch_size)
        test_loader = DataLoader(self.test_data, batch_size)

        return train_loader, val_loader, test_loader


if __name__ == "__main__":
    path_to_pickle = DATASETS_DIR / "sdm_2023-01-10_team_01_8333_9490.pkl"
    dataset = EkmanDataset(path_to_pickle)
    dataset.get_data_and_labels()
    dataset.split_dataset_into_train_val_test()

    train_dataloader, val_dataloader, test_dataloader = dataset.create_data_loader()



