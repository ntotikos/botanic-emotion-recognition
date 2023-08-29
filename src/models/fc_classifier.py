""" Implementation of Fully Connected Neural Network (FCNN) classifier plus relevant methods. """


from src.models.base_classifier import PlantClassifier
import torch


class DenseClassifier(PlantClassifier):
    def __init__(self, params):
        super().__init__(params)
        self.params = params

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

    def __call__(self, x):
        """
        Objects that are derived from nn.Module it is treated as a callable becasue nn.Module defines a
        __call__ method, which in turn calls the forward method of the object. Hence, the __call__ method is
        implemented so that it calls the forward method.
        """
        return self.forward(x)

    def get_something(self):
        pass

    def parameters(self):
        """
        Forwards the call to self.model.parameters because DenseClassifier does not inherit from nn.Module. Instead,
        composition is used because DenseClassifier already inherits from an abstract class and multi-inheritage
        shall be avoided.
        """
        return self.model.parameters()

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == "__main__":
    # Test run
    data = torch.tensor([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5], [0.4, 0.5, 0.6]], dtype=torch.float32)
    target = torch.tensor([1, 0, 0, 1], dtype=torch.long)

    # Model
    model = DenseClassifier("params")

    # Loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Number of epochs
    epochs = 100

    # Training loop
    for epoch in range(epochs):
        # Forward pass
        outputs = model(data)  # Without implemented __call__ method: model.forward(data)
        print(outputs)

        # Compute loss
        loss = criterion(outputs, target)

        # Zero gradients
        optimizer.zero_grad()

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

