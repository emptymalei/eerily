import copy
from dataclasses import dataclass
from typing import Dict, Iterator

import numpy as np


@dataclass(frozen=True)
class ARModelParams:
    """Parameters of our AR model,

    $$s(t+1) = \phi_0 + \phi_1 s(t) + \epsilon.$$

    :param delta_t: step size of time in each iteration
    :param phi0: pho_0 in the AR model
    :param phi1: pho_1 in the AR model
    :param epsilon: noise iterator, e.g., Gaussian noise
    :param initial_state: a dictionary of the initial state, e.g., `{"s": 1}`
    """

    delta_t: float
    phi0: float
    phi1: float
    epsilon: Iterator
    initial_state: Dict[str, float]


class AR1Stepper:
    """Stepper that calculates the next step in time in an AR model

    :param model_params: parameters for the AR model
    """

    def __init__(self, model_params):
        self.model_params = model_params
        self.current_state = copy.deepcopy(self.model_params.initial_state)

    def __iter__(self):
        return self

    def __next__(self):
        phi0 = self.model_params.phi0
        phi1 = self.model_params.phi1
        epsilon = next(self.model_params.epsilon)

        next_s = self.model_params.phi0 + self.model_params.phi1 * self.current_state["s"] + epsilon
        self.current_state = {"s": next_s}

        return copy.deepcopy(self.current_state)
