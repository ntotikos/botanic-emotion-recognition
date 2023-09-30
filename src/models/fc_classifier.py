""" Implementation of Fully Connected Neural Network (FC-NN) classifier plus relevant methods. """
from src.data.dataset_builder import EkmanDataset
from src.models.deeplearning_classifier import DLClassifier
import torch
import logging

from src.utils.constants import DATASETS_DIR, MODELS_DIR, LOGS_DIR
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
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, output_dim),
            # torch.nn.Softmax(dim=1) -> not needed because torch.nn.CrossEntropyLoss inherently applies softmax
        )

    def forward(self, x):
        x = self.model(x)
        return x


def objective(trial, save=False):
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
    # TODO: Pass LR and Hidden_dim so that it is being used!
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    hidden_dim = trial.suggest_int("hidden_dim", 32, 256, log=True)

    # Number of epochs
    epochs = 2

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

            logging.info(f"Epoch [{epoch + 1}/{epochs}]: training loss: {loss.item():.4f}")

        all_preds = []
        all_labels = []
        f1_class_values = []
        f1_micro_values = []
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
        f1_micro = f1_score(all_labels, all_preds, average="micro")
        f1_weighted = f1_score(all_labels, all_preds, average="weighted")

        f1_class_values.append(f1_class.tolist())  # convert to list because of conversion to db object
        f1_micro_values.append(f1_micro)
        f1_weighted_values.append(f1_weighted)

        print(f"Accuracy: {accuracy}")
        print(f"F1 class: {f1_class}")
        print(f"F1 micro: {f1_micro}")
        print(f"F1 weighted: {f1_weighted}")

        # Log additional metrics
        trial.report(accuracy, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    # Log additional eval metrics
    trial.set_user_attr("f1_class", f1_class_values)
    trial.set_user_attr("f1_micro", f1_micro_values)
    trial.set_user_attr("f1_weighted", f1_weighted_values)

    if save:
        with open(os.path.join(MODELS_DIR, 'fc_classifier_v1.pkl'), 'wb') as pkl:
            pickle.dump(model, pkl)

    return accuracy


if __name__ == "__main__":
    study = optuna.create_study(study_name="fc_study", storage="sqlite:///fc_hyperparam_opt.db", direction="maximize")
    study.optimize(objective, n_trials=2, timeout=600)

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
