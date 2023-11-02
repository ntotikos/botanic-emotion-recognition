""" Build custom datasets. """
import gc  # garbage collection
import pickle
import joblib, sqlite3
from matplotlib import pyplot as plt

from src.features.feature_factory import FeatureFactory
from src.utils.constants import DATASETS_DIR, EKMAN_NEUTRAL_TO_INT_DICT, INT_TO_EKMAN_NEUTRAL_DICT
from torch.utils.data import DataLoader, random_split, TensorDataset, WeightedRandomSampler
import torch
import os
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

from src.utils.reproducibility import set_seed
#set_seed(42)


class EkmanDataset:
    def __init__(self, data_path, feature_type="passthrough"):
        """
        :param data_path: Path to pickle file with data point dictionaries.
        """
        with open(data_path, 'rb') as file:
            raw_data = pickle.load(file)

        self.raw_data = raw_data
        self.feature_extractor = FeatureFactory.get_extractor(feature_type)

        self.dataset = None
        self.features_dataset = None

        self.train_data = None
        self.val_data = None
        self.test_data = None

        self.batch_size = 32

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, idx):
        sample = self.dataset.tensors[0][idx]
        label = self.dataset.tensors[1][idx]
        return sample, label

    def load_dataset(self):
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
        print("Ciaoooooooo", self.dataset[0])

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

    def split_dataset_into_train_val_test(self, train_split: float = 0.7, val_split: float = 0.15, stratify=False):
        # TODO: This should be an abstract method in a super class.
        # TODO 2: call this method `random_split_dataset_...` and create other `split` function that makes sure that
        # split is done in a way that one set is not getting all rare samples (that are not a lot in the dataset)

        if not stratify:  # randomly split dataset
            train_size = int(train_split * len(self.dataset))
            val_size = int(val_split * len(self.dataset))
            test_size = len(self.dataset) - train_size - val_size

            self.train_data, self.val_data, self.test_data = random_split(self.dataset,
                                                                          [train_size, val_size, test_size])
        elif stratify:  # stratified split, i.e. all subsets have similar class distribution
            y = self.dataset.tensors[1]
            indices = list(range(len(self.dataset)))
            train_indices, test_val_indices = train_test_split(indices, test_size=0.3, stratify=y, random_state=42)
            test_indices, val_indices = train_test_split(test_val_indices, test_size=0.5,
                                                         stratify=[y[i] for i in test_val_indices], random_state=42)

            #print(y)
            #print(indices)
            #print(len(train_indices))
            #print(len(test_indices))
            #print(len(val_indices))

            self.train_data = Subset(self.dataset, train_indices)
            self.test_data = Subset(self.dataset, test_indices)
            self.val_data = Subset(self.dataset, val_indices)

            #print(Subset(dataset, train_indices))


    @staticmethod
    def get_label_distribution(dataloader):
        # TODO: dataloader_labels = get_labels(dataloader)
        dataloader_labels = []
        for batch_data, batch_labels in dataloader:
            dataloader_labels.extend([int(i) for i in batch_labels])
        df = pd.DataFrame([map_int_to_label(j) for j in dataloader_labels], columns=['Class'])
        class_counts = df['Class'].value_counts()
        return class_counts

    def create_data_loader(self, batch_size=32, upsampling="none"):
        # Upsampling the training data.
        if upsampling == "none":
            # TODO: How to handle case when there are only 29/32 samples left? Padding or doesn't matter?
            train_loader = DataLoader(self.train_data, batch_size)
        elif upsampling == "naive":  # naive upsampling
            # Get samples and class distribution
            y = torch.tensor([sample[1] for sample in self.train_data])
            n_class_samples = torch.tensor([(y == t).sum() for t in torch.unique(y, sorted=True)])

            # Compute weights for weighted sampler
            weights = 1. / n_class_samples.float()
            samples_weights = torch.tensor([weights[t] for t in y])
            sampler = WeightedRandomSampler(samples_weights, len(samples_weights))

            # Create DataLoader for training
            train_loader = DataLoader(self.train_data, batch_size=batch_size, sampler=sampler)
        elif upsampling == "smote":
            pass

        val_loader = DataLoader(self.val_data, batch_size)
        test_loader = DataLoader(self.test_data, batch_size)

        return train_loader, val_loader, test_loader

    def class_decomposition(self, save_path=DATASETS_DIR, method="ovo"):
        """
        Implementation of class decomposition "one-vs-ovo" and "one-vs-all" for mitigating class imbalance and
        overlapping classes. It generates binary classification datasets that are stored in the memory. In the case of
        ovo with 7 classes we get 21 binary classifiers, for ova it is 7 binary classifiers.

        Theoretically, can be applied to balanced and imbalanced datasets.

        :param save_path: Path to save generated binary datasets.
        :param method: Specify method ovo or ova.
        """
        unique_labels = torch.unique(torch.tensor([sample[1] for sample in self.dataset]))

        path = os.path.join(save_path, method+"-datasets")
        if not os.path.exists(path):
            os.mkdir(path)
            print(f"Directory created: {path}")

        for idx, label1 in enumerate(unique_labels):
            for label2 in unique_labels[idx + 1:]:
                # Generate binary dataset for all combinations.
                binary_data = [(data.float().numpy(), int(target)) for data, target in self.dataset if target == label1
                               or target == label2]

                file_name = f"binary_{map_int_to_label(label1)}_{label1}_vs_{map_int_to_label(label2)}_{label2}.pkl"
                full_path = os.path.join(path, file_name)

                if not os.path.exists(full_path):
                    with open(full_path, 'wb') as f:
                        pickle.dump(binary_data, f)

                #del binary_data  # delete reference to object
                #gc.collect()

    # def class_decomposition(self, save_path=DATASETS_DIR, method="ovo"):
    #     """
    #     Implementation of class decomposition "one-vs-ovo" and "one-vs-all" for mitigating class imbalance and
    #     overlapping classes. It generates binary classification datasets that are stored in the memory. In the case of
    #     ovo with 7 classes we get 21 binary classifiers, for ova it is 7 binary classifiers.
    #
    #     Theoretically, can be applied to balanced and imbalanced datasets.
    #
    #     :param save_path: Path to save generated binary datasets.
    #     :param method: Specify method ovo or ova.
    #     """
    #     unique_labels = torch.unique(torch.tensor([sample[1] for sample in self.dataset]))
    #
    #     path = os.path.join(save_path, method+"-datasets")
    #     if not os.path.exists(path):
    #         os.mkdir(path)
    #         print(f"Directory created: {path}")
    #
    #     for idx, label1 in enumerate(unique_labels):
    #         for label2 in unique_labels[idx + 1:]:
    #             # Generate binary dataset for all combinations.
    #             binary_data = [(data.float().numpy(), int(target)) for data, target in self.dataset if target == label1 or target == label2]
    #
    #             file_name = f"binary_{map_int_to_label(label1)}{label1}_vs_{map_int_to_label(label2)}{label2}.joblib"
    #             full_path = os.path.join(path, file_name)
    #
    #             if not os.path.exists(full_path):
    #                 joblib.dump(binary_data, full_path)
    #             print(f"#Samples: {len(binary_data)}")
    #             print(f"#Samples: {binary_data[0:3]}")
    #             del binary_data  # delete reference to object
    #             gc.collect()

    def remove_neutral(self):
        # TODO: implement removal of neutral class from loaded dataset.
        pass

    def extract_features(self):
        """
        Make sure to normalize the time series before computing the MFCC features.
        """
        features = []
        labels = []

        for wav_slice, label in tqdm(self.dataset):
            sample_features = self.feature_extractor.extract(wav_slice)
            features.append(sample_features)  # sample_features should be torch.Tensor torch.float32!
            labels.append(label)

        features_tensor = torch.tensor(np.array(features), dtype=torch.float32)
        labels_tensor = torch.tensor(np.array(labels), dtype=torch.long)
        self.dataset = TensorDataset(features_tensor, labels_tensor)

        print("Hiiiiiiiiiiiiiiii", self.dataset[0][0])
        print("Size", self.dataset[0][0].size())

        plt.imshow(self.dataset[0][0], cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.show()


def map_int_to_label(emotion: int):
    return INT_TO_EKMAN_NEUTRAL_DICT[int(emotion)]


if __name__ == "__main__":
    path_to_pickle = DATASETS_DIR / "sdm_2023-01_all_valid_files_version_1.pkl"

    """
    Plot class distribution balanced vs. imbalanced. 
    """
    dataset = EkmanDataset(path_to_pickle, feature_type="spectral")
    dataset.load_dataset()
    dataset.normalize_samples()
    dataset.extract_features()

    dataset.split_dataset_into_train_val_test(stratify=True)

    train_dl, _, _ = dataset.create_data_loader(upsampling="none")
    train_dataloader, val_dataloader, test_dataloader = dataset.create_data_loader(upsampling="naive")

    output = dataset.get_label_distribution(train_dl)
    output_2 = dataset.get_label_distribution(train_dataloader)

    print("OUTPUT:", output)
    print("OUTPUT 2:", output_2)

    """
    Class decomposition: split multi-class dataset into 21 binary datasets.
    """
    #dataset = EkmanDataset(path_to_pickle)
    #dataset.load_dataset()
    #dataset.class_decomposition()
