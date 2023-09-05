""" Implementation of Fully Connected Neural Network (FC-NN) classifier plus relevant methods. """
from src.data.dataset_builder import EkmanDataset
from src.models.deeplearning_classifier import DLClassifier
import torch
from tqdm import tqdm

from src.utils.constants import DATASETS_DIR


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
        input_dim = 3
        output_dim = 2  # For Ekman neutral: 7

        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, output_dim),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.model(x)
        return x


def _main_pseudo_training():
    # Test run
    data = torch.tensor([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5], [0.4, 0.5, 0.6]], dtype=torch.float32)
    target = torch.tensor([1, 0, 0, 1], dtype=torch.long)

    # Model
    model = DenseClassifier("params")
    model.setup_model()
    model.setup_training()

    # Number of epochs
    epochs = 100

    # Training loop
    for epoch in range(epochs):
        # Forward pass
        outputs = model(data)  # Without implemented __call__ method: model.forward(data)

        # Compute loss
        loss = model.criterion(outputs, target)

        # Zero gradients
        model.optimizer.zero_grad()

        # Backward pass and optimize
        loss.backward()
        model.optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")


def _main_one_file():
    path_to_pickle = DATASETS_DIR / "sdm_2023-01-10_team_01_8333_9490.pkl"
    dataset = EkmanDataset(path_to_pickle)
    dataset.get_data_and_labels()
    dataset.split_dataset_into_train_val_test()

    train_dataloader, val_dataloader, test_dataloader = dataset.create_data_loader()

    # TODO: use correct input and output dimensions
    input_dim = 10000
    output_dim = 7
    # Model
    model = DenseClassifier("params")
    model.setup_model()
    model.model = torch.nn.Sequential(
        torch.nn.Linear(input_dim, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, output_dim)
        # torch.nn.Softmax(dim=1) -> not needed because torch.nn.CrossEntropyLoss inherently applies softmax
        )
    model.setup_training()

    # Number of epochs
    epochs = 200

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

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")


def _main():
    path_to_pickle = DATASETS_DIR / "sdm_2023-01_all_valid_files_version_1.pkl"
    dataset = EkmanDataset(path_to_pickle)
    dataset.get_data_and_labels()
    dataset.split_dataset_into_train_val_test()

    train_dataloader, val_dataloader, test_dataloader = dataset.create_data_loader()

    dataset.get_label_distribution(train_dataloader)

    # TODO: use correct input and output dimensions
    input_dim = 10000
    output_dim = 7
    # Model
    model = DenseClassifier("params")
    model.setup_model()
    model.model = torch.nn.Sequential(
        torch.nn.Linear(input_dim, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, output_dim)
        # torch.nn.Softmax(dim=1) -> not needed because torch.nn.CrossEntropyLoss inherently applies softmax
        )
    model.setup_training()

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

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")


if __name__ == "__main__":
    # _main_pseudo_training()
    _main()
