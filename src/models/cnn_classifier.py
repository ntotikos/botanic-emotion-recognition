""" Implementation of Convolutional Neural Network (CNN) classifier plus relevant methods. """
from src.data.dataset_builder import EkmanDataset
from src.models.deeplearning_classifier import DLClassifier
import torch

from src.utils.constants import DATASETS_DIR, MODELS_DIR, LOGS_DIR, EKMAN_EMOTIONS_NEUTRAL

import os
import pickle

from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, precision_score, recall_score, \
    classification_report

import optuna
from optuna.trial import TrialState
import wandb


class CNNClassifier(DLClassifier):
    """
    Class for CNN + MFCC classifier.
    """
    def __init__(self, params):
        super().__init__(params)
        self.params = params

    def __call__(self, x):
        """
        Objects that are derived from nn.Module it is treated as a callable because nn.Module defines a
        __call__ method, which in turn calls the forward method of the object. Hence, the __call__ method is
        implemented so that it calls the forward method.
        """
        return self.forward(x)

    def setup_model(self):
        input_dim = 10000
        output_dim = 6

        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # torch.nn.Dropout(self.dropout_rate),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # torch.nn.Dropout(self.dropout_rate),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 7 * 7, 64),
            # torch.nn.Dropout(self.dropout_rate),
            torch.nn.ReLU(),
            torch.nn.Linear(64, output_dim)
        )

    def forward(self, x):
        x = self.model(x)
        return x


def objective(trial, save=False):
    "---------- Adjust to image dataloader-----------"
    # Get the TS dataset.
    path_to_pickle = DATASETS_DIR / "sdm_2023-01_all_valid_files_version_1.pkl"
    dataset = EkmanDataset(path_to_pickle)
    #dataset.load_dataset()
    dataset.load_data_and_labels_without_neutral()
    dataset.normalize_samples(normalization="per-sample")
    #dataset.load_dataset()
    dataset.split_dataset_into_train_val_test(stratify=True)

    train_dataloader, val_dataloader, test_dataloader = dataset.create_data_loader(upsampling="none")

    dataset.get_label_distribution(train_dataloader)
    "---------- Adjust to image dataloader-----------"

    # Generate the model.
    model = CNNClassifier("params")
    model.setup_model()
    model.setup_training()

    # HP search space.
    lr = trial.suggest_categorical('lr', [0.0001, 0.001, 0.01])
    conv_filters_1 = trial.suggest_categorical('conv_filters_1', [2 ** i for i in range(6, 8)])
    conv_filters_2 = trial.suggest_categorical('conv_filters_2', [2 ** i for i in range(5, 7)])
    conv_kernel_size = trial.suggest_categorical('conv_kernel_size', [3, 5, 7])
    dropout_rate = trial.suggest_categorical('dropout_rate', [0, 0.1, 0.2])

    model.learning_rate = lr
    print("model.learning_rate", model.learning_rate)
    model.n_conv_fil_1 = conv_filters_1
    model.n_con_fil_2 = conv_filters_2
    model.conv_ker_size = conv_kernel_size
    model.dropout_rate = dropout_rate

    # remove model.setup_training() and run:
    #criterion = torch.nn.CrossEntropyLoss()
    #optimizer = torch.optim.Adam(model.parameters(), lr=model.learning_rate)

    name_core = "cnn-multi-class_6_rgb_81k"
    name_experiment = (f"{trial.number}_{name_core}_lr-{lr}_convf1-{conv_filters_1}_convf2-"
                       f"{conv_filters_2}_convker-{conv_kernel_size}_dr-{dropout_rate}")

    config_dict = {
        "lr": lr,
        "conv_filters_1": conv_filters_1,
        "conv_filters_2": conv_filters_2,
        "conv_kernel_size": conv_kernel_size,
        "dropout_rate": dropout_rate,
        "epochs": 35
    }

    wandb.init(
        project="model_" + name_core + "-hpo",
        dir=LOGS_DIR,
        name=name_experiment,
        config=config_dict
    )

    # Number of epochs
    epochs = 35  # instead of 40; values are rather constant after 15 epochs. Probably due to imbalance in data

    # Training loop
    for epoch in range(epochs):
        model.model.train()
        for batch_data, batch_labels in train_dataloader:
            print("Batch_data[0]: ", batch_data[0])
            # Zero gradients
            model.optimizer.zero_grad()

            # Forward pass
            outputs = model(batch_data)  # Without implemented __call__ method: model.forward(data)

            # Compute loss
            loss = model.criterion(outputs, batch_labels)

            # Backward pass and optimize
            loss.backward()
            model.optimizer.step()

        all_preds = []
        all_labels = []

        model.model.eval()
        with torch.no_grad():
            for batch_data, batch_labels in val_dataloader:
                output = model(batch_data)  # 32 x 7

                predicted = output.argmax(dim=1)
                all_preds.extend(predicted.numpy())
                all_labels.extend(batch_labels.numpy())

        # TODO: explicitely state in written thesis that zero_division=0.0
        balanced_accuracy = balanced_accuracy_score(all_labels, all_preds)
        accuracy = accuracy_score(all_labels, all_preds)
        f1_class = f1_score(all_labels, all_preds, average=None, zero_division=0.0)
        f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0.0)
        recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0.0)
        precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0.0)
        report = classification_report(
            all_labels,
            all_preds,
            target_names=["Angry 0", "Disgust 1", "Happy 2", "Sad 3", "Surprise 4", "Fear 5"])

        trial.report(balanced_accuracy, epoch)

        trial.set_user_attr("f1_class", f1_class.tolist())
        trial.set_user_attr("f1_weighted", f1_weighted)
        trial.set_user_attr("accuracy", accuracy)
        trial.set_user_attr("precision", precision)
        trial.set_user_attr("recall", recall)
        trial.set_user_attr("classification_report", report)

        metrics = {
            "balanced_accuracy": balanced_accuracy,
            "accuracy": accuracy,
            "f1_weighted": f1_weighted,
            "recall": recall,
            "precision": precision
        }

        wandb_input = metrics
        wandb.log(wandb_input)

    wandb.finish()

    return balanced_accuracy
