import torch
from abc import ABC, abstractmethod


class Unlearner(ABC):
    """
    Abstract class to implement custom unlearning methods for a given ml model.
    Parameters
    ----------
    mlmodel: xai-bench.models.MLModel
        Classifier we wish to unlearn a subset of points from.
    Methods
    -------
    get_updated_model:
        Generate updated model for given forget set, retain set and validation set.
    Returns
    -------
    None
    """

    def __init__(self, mlmodel):
        self.model = mlmodel

    @abstractmethod
    def get_updated_model(self, retain_set, forget_set, validation_set):
        """
        get_updated_model:
            Generate updated model for given forget set, retain set and validation set.
        Parameters
        ----------
        inputs: torch.tensor
            Input in two-dimensional shape (m, n)
        label: torch.tensor
            Label
        Returns
        -------
        model
            updated pytorch model
        """
        pass
