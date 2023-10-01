""" Base class and methods for deep learning based classifier. """
from abc import abstractmethod

from base_classifier import PlantClassifier
import torch


class DLClassifier(PlantClassifier):
    """
    ...
    """
    def __init__(self, params):
        super().__init__(params)
        # self.params = params
        self.criterion = None
        self.optimizer = None
        self.model = None
        self.learning_rate = 0.0001
        self.n_hidden_1 = 128
        self.n_hidden_2 = 64


    def __call__(self, **kwargs):
        """
        Objects that are derived from nn.Module it is treated as a callable because nn.Module defines a
        __call__ method, which in turn calls the forward method of the object. Hence, the __call__ method is
        implemented so that it calls the forward method.
        """
        raise NotImplementedError("The ´__call__´ method is not implemented in this subclass of "
                                  "the abstract ´DLClassifier´ class.")

    @abstractmethod
    def setup_model(self):
        raise NotImplementedError("The ´setup_model´ method is not implemented in this subclass of "
                                  "the abstract ´DLClassifier´ class.")

    @abstractmethod
    def forward(self, **kwargs):
        raise NotImplementedError("The ´forward´ method is not implemented in this subclass of "
                                  "the abstract ´DLClassifier´ class.")

    def setup_training(self):
        """
        TODO: If binary classification use BCELoss(). If multi-class classification use CrossEntropyLoss().
        In this method!
        """
        # Loss function & optimizer
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def parameters(self):
        """
        Forwards the call to self.model.parameters because DenseClassifier does not inherit from nn.Module. Instead,
        composition is used because DenseClassifier already inherits from an abstract class and multi-inheritance
        shall be avoided.
        """
        return self.model.parameters()
