"""
This file implements the one-vs-one class decomposition to mitigate
class imbalance and class overlap.
"""


class OvOEnsemble:
    def __init__(self, input_size, hidden_neurons):
        self.models = {}
        combinations = [(i, j) for i in range(7) for j in range(i+1, 7)]