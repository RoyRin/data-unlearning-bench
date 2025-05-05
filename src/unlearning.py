# stdlib dependencies
from abc import ABC, abstractmethod
from copy import deepcopy

# third party deps
import torch
from torch.utils.data import DataLoader 

class UnlearningMethod(ABC):
    @abstractmethod
    def unlearn(self, model: torch.nn.Module, forget_loader: DataLoader, retain_loader: DataLoader, **kwargs) -> torch.nn.Module:
        pass

class DoNothing(UnlearningMethod):
    def unlearn(self, model, forget_set, retain_set, **kwargs):
        return deepcopy(model)

UNLEARNING_METHODS = {
        "do_nothing": DoNothing,
}
