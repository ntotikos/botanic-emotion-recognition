""" Implementation of Fully Connected Neural Network (FC-NN) classifier plus relevant methods. """
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


class DenseClassifier(DLClassifier):
    """
    ...
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
        #input_dim = 10000 # raw time-series dimension
        #input_dim = 1287  # flattened mfcc dimension
        #input_dim = 10004  # dwt-1
        input_dim = 10013

        output_dim = 6  # For Ekman neutral: 6
        #output_dim = 7  # For Ekman neutral: 7

        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, self.n_hidden_1),
            torch.nn.ReLU(),
            torch.nn.Linear(self.n_hidden_1, self.n_hidden_2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.n_hidden_2, output_dim),
            # torch.nn.Softmax(dim=1) -> not needed because torch.nn.CrossEntropyLoss inherently applies softmax
        )

    def forward(self, x):
        x = self.model(x)
        return x


def objective(trial, save=False):
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

    # Generate the model.
    model = DenseClassifier("params")
    model.setup_model()
    model.setup_training()

    # HP search space.
    lr = trial.suggest_categorical('lr', [0.0001, 0.001, 0.01])
    hidden_dim_1 = trial.suggest_categorical('hidden_dim_1', [2 ** i for i in range(4, 8)])
    hidden_dim_2 = trial.suggest_categorical('hidden_dim_2', [2 ** i for i in range(4, 7)])
    dropout_rate = trial.suggest_categorical('dropout_rate', [0, 0.1, 0.2])

    model.learning_rate = lr
    model.n_hidden_1 = hidden_dim_1
    model.n_hidden_2 = hidden_dim_2
    model.dropout_rate = dropout_rate

    #name_core = "fc-multi-class_6_normalized_191k"
    name_core = "fc-multi-class_6_normalized_81k"
    name_experiment = (f"{trial.number}_{name_core}_lr-{lr}_hd1-{hidden_dim_1}_hd2-"
                       f"{hidden_dim_2}_dr-{dropout_rate}")
    experiment_notes = """
    This experiment uses a multi-class FC model for hyperparameter optimization AND importantly normalization.
    The dataset has imbalances among the classes. 7-class and 191k samples with neutral.
    Precision, recall, and F1 metrics are computed with zero_division set to 0.0 to handle potential edge cases.
    Optuna is used for hyperparameter optimization.
    """

    config_dict = {
        "lr": lr,
        "hidden_dim_1": hidden_dim_1,
        "hidden_dim_2": hidden_dim_2,
        "dropout_rate": dropout_rate,
        "epochs": 35
    }

    wandb.init(
        project="baseline_" + name_core + "-hpo",
        dir=LOGS_DIR,
        name=name_experiment,
        notes=experiment_notes,
        config=config_dict
    )

    # Number of epochs
    epochs = 25  # instead of 40; values are rather constant after 15 epochs. Probably due to imbalance in data

    # Training loop
    for epoch in range(epochs):
        model.model.train()
        for batch_data, batch_labels in train_dataloader:
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
                #pred = output.argmax(dim=1, keepdims=True)  # 32 x 1

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
        #auc = roc_auc_score(all_labels, all_preds, multi_class='ovr')  # TODO: document this.
        report = classification_report(
            all_labels,
            all_preds,
            target_names=["Angry 0", "Disgust 1", "Happy 2", "Sad 3", "Surprise 4", "Fear 5"])
            #target_names=["Angry 0", "Disgust 1", "Happy 2", "Sad 3", "Surprise 4", "Fear 5", "Neutral 6"])

        trial.report(balanced_accuracy, epoch)

        #if trial.should_prune():
        #    print(f"{trial.number}_fc-multi-class_6_normalized_lr-{lr}_hd1-{hidden_dim_1}_hd2-"
        #          f"{hidden_dim_2}_dr-{dropout_rate}")
        #    raise optuna.exceptions.TrialPruned()

        trial.set_user_attr("f1_class", f1_class.tolist())
        trial.set_user_attr("f1_weighted", f1_weighted)
        trial.set_user_attr("accuracy", accuracy)
        trial.set_user_attr("precision", precision)
        trial.set_user_attr("recall", recall)
        #trial.set_user_attr("roc_auc", a uc)
        trial.set_user_attr("classification_report", report)

        # USE THIS ONLY WHEN 7 classes.
        #f1_per_class_dict = {}
        #for idx, class_name in enumerate(EKMAN_EMOTIONS_NEUTRAL):
        #   f1_per_class_dict[f"f1_{class_name.lower()}"] = f1_class[idx]

        metrics = {
            "balanced_accuracy": balanced_accuracy,
            "accuracy": accuracy,
            "f1_weighted": f1_weighted,
            "recall": recall,
            "precision": precision
        }

        wandb_input = metrics
        #wandb_input = metrics | f1_per_class_dict
        wandb.log(wandb_input)

    wandb.finish()

    return balanced_accuracy


def objective_spectral(trial, save=False):
    method = "dwt-3"

    # Get the TS dataset.
    path_to_pickle = DATASETS_DIR / "sdm_2023-01_all_valid_files_version_iter2.pkl"
    dataset = EkmanDataset(path_to_pickle, feature_type="spectral", method_type=method)
    #dataset.load_dataset()
    dataset.load_data_and_labels_without_neutral()

    #dataset.normalize_samples(normalization="per-sample")
    #dataset.extract_features(flatten=True) # For MFCC
    dataset.extract_features(flatten=False)
    dataset.split_dataset_into_train_val_test(stratify=True)

    train_dataloader, val_dataloader, test_dataloader = dataset.create_data_loader(upsampling="none")

    dataset.get_label_distribution(train_dataloader)

    # Generate the model.
    model = DenseClassifier("params")
    model.setup_model()
    model.setup_training()

    # HP search space.
    lr = trial.suggest_categorical('lr', [0.0001, 0.001, 0.01])
    hidden_dim_1 = trial.suggest_categorical('hidden_dim_1', [2 ** i for i in range(4, 8)])
    hidden_dim_2 = trial.suggest_categorical('hidden_dim_2', [2 ** i for i in range(4, 7)])
    dropout_rate = trial.suggest_categorical('dropout_rate', [0, 0.1, 0.2])

    model.learning_rate = lr
    model.n_hidden_1 = hidden_dim_1
    model.n_hidden_2 = hidden_dim_2
    model.dropout_rate = dropout_rate

    # MFCC
    #name_core = "mfcc-fc-multi-class_7_normalized_191k"
    #name_core = "mfcc-fc-multi-class_6_normalized_81k"

    # Wavelet DWT-1
    name_core = "dwt3-fc-multi-class_6_normalized-after-dwt_191k"
    #name_core = "dwt1-fc-multi-class_6_normalized_81k"

    name_experiment = (f"{trial.number}_{name_core}_lr-{lr}_hd1-{hidden_dim_1}_hd2-"
                       f"{hidden_dim_2}_dr-{dropout_rate}")
    experiment_notes = """
    This experiment uses a multi-class FC model for hyperparameter optimization AND importantly normalization.
    The dataset has imbalances among the classes. 7-class and 191k samples with neutral.
    Precision, recall, and F1 metrics are computed with zero_division set to 0.0 to handle potential edge cases.
    Optuna is used for hyperparameter optimization.
    """

    config_dict = {
        "lr": lr,
        "hidden_dim_1": hidden_dim_1,
        "hidden_dim_2": hidden_dim_2,
        "dropout_rate": dropout_rate,
        "epochs": 20
    }

    wandb.init(
        project= name_core + "-hpo",
        dir=LOGS_DIR,
        name=name_experiment,
        notes=experiment_notes,
        config=config_dict
    )

    # Number of epochs
    epochs = 20  # instead of 40; values are rather constant after 15 epochs. Probably due to imbalance in data

    # Training loop
    for epoch in range(epochs):
        model.model.train()
        for batch_data, batch_labels in train_dataloader:
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
                #pred = output.argmax(dim=1, keepdims=True)  # 32 x 1

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
        #auc = roc_auc_score(all_labels, all_preds, multi_class='ovr')  # TODO: document this.
        report = classification_report(
            all_labels,
            all_preds,
            target_names=["Angry 0", "Disgust 1", "Happy 2", "Sad 3", "Surprise 4", "Fear 5"])
            #target_names=["Angry 0", "Disgust 1", "Happy 2", "Sad 3", "Surprise 4", "Fear 5", "Neutral 6"])

        trial.report(balanced_accuracy, epoch)

        #if trial.should_prune():
        #    print(f"{trial.number}_fc-multi-class_6_normalized_lr-{lr}_hd1-{hidden_dim_1}_hd2-"
        #          f"{hidden_dim_2}_dr-{dropout_rate}")
        #    raise optuna.exceptions.TrialPruned()

        trial.set_user_attr("f1_class", f1_class.tolist())
        trial.set_user_attr("f1_weighted", f1_weighted)
        trial.set_user_attr("accuracy", accuracy)
        trial.set_user_attr("precision", precision)
        trial.set_user_attr("recall", recall)
        #trial.set_user_attr("roc_auc", a uc)
        trial.set_user_attr("classification_report", report)

        # USE THIS ONLY WHEN 7 classes.
        #f1_per_class_dict = {}
        #for idx, class_name in enumerate(EKMAN_EMOTIONS_NEUTRAL):
        #   f1_per_class_dict[f"f1_{class_name.lower()}"] = f1_class[idx]

        metrics = {
            "balanced_accuracy": balanced_accuracy,
            "accuracy": accuracy,
            "f1_weighted": f1_weighted,
            "recall": recall,
            "precision": precision
        }

        wandb_input = metrics
        #wandb_input = metrics | f1_per_class_dict
        wandb.log(wandb_input)

    wandb.finish()

    return balanced_accuracy


def main_hp_optimization():
    search_space = {
        'lr': [0.0001, 0.001, 0.01],
        'hidden_dim_1': [2 ** i for i in range(4, 8)],
        'hidden_dim_2': [2 ** i for i in range(4, 7)],
        "dropout_rate": [0, 0.1, 0.2]
    }

    #name_core = "fc_baseline_6_normalized_191k"
    name_core = "fc_baseline_6_normalized_81k"

    sampler = optuna.samplers.GridSampler(search_space)  # Grid Search
    study = optuna.create_study(sampler=sampler, study_name=name_core, storage="sqlite:///hpo_" + name_core + ".db",
                                direction="maximize", load_if_exists=True)
    study.optimize(objective, n_trials=108)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial_ = study.best_trial

    print("  Value: ", trial_.value)

    print("  Params: ")
    for key, value in trial_.params.items():
        print("    {}: {}".format(key, value))


def _main(save=True):
    #set_seed(42)

    # Get the TS dataset.
    path_to_pickle = DATASETS_DIR / "sdm_2023-01_all_valid_files_version_1.pkl"
    dataset = EkmanDataset(path_to_pickle)
    dataset.load_dataset()
    dataset.split_dataset_into_train_val_test()

    train_dataloader, val_dataloader, test_dataloader = dataset.create_data_loader(upsampling="none")

    dataset.get_label_distribution(train_dataloader)

    # Generate the model.
    model = DenseClassifier("params")
    model.setup_model()
    model.setup_training()

    # Hyperparameter suggestion.
    lr = 0.1
    hidden_dim_1 = 16
    hidden_dim_2 = 64

    model.learning_rate = lr
    model.n_hidden_1 = hidden_dim_1
    model.n_hidden_2 = hidden_dim_2

    # Number of epochs
    epochs = 50

    # Training loop
    for epoch in range(epochs):
        for batch_data, batch_labels in train_dataloader:
            # Zero gradients
            model.optimizer.zero_grad()

            # Forward pass
            outputs = model(batch_data)  # Without implemented __call__ method: model.forward(data)

            # Compute loss
            loss = model.criterion(outputs, batch_labels)

            # Backward pass and optimize
            loss.backward()
            model.optimizer.step()

            # logging.info(f"[{epoch + 1}/{epochs}]:training loss: {loss.item}")

        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    if save:
        with open(os.path.join(MODELS_DIR, 'fc_classifier_v1.pkl'), 'wb') as pkl:
            pickle.dump(model, pkl)


def main_hp_optimization_spectral():
    search_space = {
        'lr': [0.0001, 0.001, 0.01],
        'hidden_dim_1': [2 ** i for i in range(4, 8)],
        'hidden_dim_2': [2 ** i for i in range(4, 7)],
        "dropout_rate": [0, 0.1, 0.2]
    }

    # MFCC
    #name_core = "fc_mfcc_7_normalized_191k"
    #name_core = "fc_mfcc_6_normalized_81k"

    # Wavelet
    name_core = "fc_dwt3_6_normalized-after-dwt_191k"
    #name_core = "fc_dwt1_6_normalized_81k"

    sampler = optuna.samplers.GridSampler(search_space)  # Grid Search
    study = optuna.create_study(sampler=sampler, study_name=name_core, storage="sqlite:///hpo_" + name_core + ".db",
                                direction="maximize", load_if_exists=True)
    study.optimize(objective_spectral, n_trials=108)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial_ = study.best_trial

    print("  Value: ", trial_.value)

    print("  Params: ")
    for key, value in trial_.params.items():
        print("    {}: {}".format(key, value))


if __name__ == "__main__":
    # main_hp_optimization()  # raw TS
    #_main(False)
    main_hp_optimization_spectral()  # MFCCs + Wavelet (level 1 & 3)


