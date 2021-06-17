import numpy as np
import abc


class Model(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def Evaluate(self, x):  # Take an input variable x and return an output variable y
        pass

    @abc.abstractmethod
    def Distance(self, y1, y2):  # Compute the distance between two output variables
        pass

    @abc.abstractmethod
    def Create(self, xy_tuples, **kwargs):  # Create a model with the given (x, y) observations
        pass

    @abc.abstractmethod
    def MinimumNumberOfDataToDefineModel(self):  # The minimum number or (x, y) observations to define the model
        pass


class Modeler():
    def __init__(self, number_of_trials, acceptable_error):
        self.number_of_trials = number_of_trials
        self.acceptable_error = acceptable_error

    def ConsensusModel(self, xy_tuples, **kwargs):
        pass
