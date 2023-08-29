""" This file provides the base class for all emotion classifiers. """
from abc import ABC, abstractmethod


class PlantClassifier(ABC):
    """
    Base class for all emotion classifiers with useful methods.
    """
    def __init__(self, params: None):
        self.params = params

    @abstractmethod
    def get_something(self):
        #TODO: implement methods
        return 1



