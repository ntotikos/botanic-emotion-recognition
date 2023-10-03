""" Implementation of Fully Connected Neural Network (FC-NN) classifier plus relevant methods. """
from src.data.dataset_builder import EkmanDataset
from src.models.deeplearning_classifier import DLClassifier
import torch
import logging

from src.utils.constants import DATASETS_DIR, MODELS_DIR, LOGS_DIR
from src.utils.reproducibility import set_seed

import os
import pickle #TODO: Instead of pickle maybe use ...lib

from sklearn.metrics import f1_score, accuracy_score

import optuna
from optuna.trial import TrialState

logging.basicConfig(filename=LOGS_DIR / 'training/fc_classifier_try.log',
                    filemode='w',
                    level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')


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
        input_dim = 10000
        output_dim = 7  # For Ekman neutral: 7

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
    set_seed(42)

    # Get the TS dataset.
    path_to_pickle = DATASETS_DIR / "sdm_2023-01_all_valid_files_version_1.pkl"
    dataset = EkmanDataset(path_to_pickle)
    dataset.get_data_and_labels()
    dataset.split_dataset_into_train_val_test()

    train_dataloader, val_dataloader, test_dataloader = dataset.create_data_loader()

    dataset.get_label_distribution(train_dataloader)

    # Generate the model.
    model = DenseClassifier("params")
    model.setup_model()
    model.setup_training()

    # Hyperparameter suggestion.
    lr = trial.suggest_categorical('lr', [0.0001, 0.001, 0.01, 0.1])
    hidden_dim_1 = trial.suggest_categorical('hidden_dim_1', [2 ** i for i in range(3, 9)])
    hidden_dim_2 = trial.suggest_categorical('hidden_dim_2', [2 ** i for i in range(3, 8)])

    model.learning_rate = lr
    model.n_hidden_1 = hidden_dim_1
    model.n_hidden_2 = hidden_dim_2

    print(model.learning_rate)
    print(model.n_hidden_1)
    print(model.n_hidden_2)

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

            #logging.info(f"Epoch [{epoch + 1}/{epochs}]: training loss: {loss.item():.4f}")

        all_preds = []
        all_labels = []
        f1_class_values = []
        f1_weighted_values = []

        model.model.eval()
        correct = 0
        with torch.no_grad():
            for batch_data, batch_labels in val_dataloader:
                output = model(batch_data)  # 32 x 7
                pred = output.argmax(dim=1, keepdims=True)  # 32 x 1
                correct += pred.eq(batch_labels.view_as(pred)).sum().item()  # assure same dimensions

                predicted = output.argmax(dim=1)
                all_preds.extend(predicted.numpy())
                all_labels.extend(batch_labels.numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        f1_class = f1_score(all_labels, all_preds, average=None)
        f1_weighted = f1_score(all_labels, all_preds, average="weighted")

        f1_class_values.append(f1_class.tolist())  # convert to list because of conversion to db object
        f1_weighted_values.append(f1_weighted)

        #print(f"Accuracy: {accuracy}")
        #print(f"F1 class: {f1_class}")
        #print(f"F1 weighted: {f1_weighted}")

        # Log additional metrics
        trial.report(accuracy, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        #print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    # Log additional eval metrics
    trial.set_user_attr("f1_class", f1_class_values)
    trial.set_user_attr("f1_weighted", f1_weighted_values)

    if save:
        with open(os.path.join(MODELS_DIR, 'fc_classifier_v1.pkl'), 'wb') as pkl:
            pickle.dump(model, pkl)

    return accuracy


def main_hp_optimization():
    search_space = {
        'lr': [0.0001, 0.001, 0.01, 0.1],
        'hidden_dim_1': [2 ** i for i in range(3, 9)],
        'hidden_dim_2': [2 ** i for i in range(3, 7)]
    }

    sampler = optuna.samplers.GridSampler(search_space)  # Grid Search
    study = optuna.create_study(sampler=sampler, study_name="fc_study", storage="sqlite:///fc_hyperparam_opt.db",
                                direction="maximize", load_if_exists=True)
    study.optimize(objective, n_trials=96)

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
    set_seed(42)

    # Get the TS dataset.
    path_to_pickle = DATASETS_DIR / "sdm_2023-01_all_valid_files_version_1.pkl"
    dataset = EkmanDataset(path_to_pickle)
    dataset.get_data_and_labels()
    dataset.split_dataset_into_train_val_test()

    train_dataloader, val_dataloader, test_dataloader = dataset.create_data_loader()

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

            logging.info(f"[{epoch + 1}/{epochs}]:training loss: {loss.item}")

        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    if save:
        with open(os.path.join(MODELS_DIR, 'fc_classifier_v1.pkl'), 'wb') as pkl:
            pickle.dump(model, pkl)


if __name__ == "__main__":
    # main_hp_optimization()
    _main(False)
