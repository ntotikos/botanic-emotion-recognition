""" This file provides the base class for all emotion classifiers. """
from abc import ABC, abstractmethod


class PlantClassifier(ABC):
    """
    Base class for all emotion classifiers with useful methods. For deep learning based classifiers and classical
    machine learning techniques.
    """
    def __init__(self, params: str):
        """
        Initialize.......
        """
        self.params = params

    @abstractmethod
    def setup_training(self):
        """
        Set up the training environment.
        """
        raise NotImplementedError("The ´setup_training´ method is not implemented in this subclass of "
                                  "the abstract ´PlantClassifier´ class.")

    @abstractmethod
    def setup_model(self):
        """
        Set up the model parameters.
        """
        raise NotImplementedError("The ´setup_model´ method is not implemented in this subclass of "
                                  "the abstract ´PlantClassifier´ class.")
