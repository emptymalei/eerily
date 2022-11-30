import copy
from dataclasses import dataclass
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class VAR1ModelParams:
    """Parameters of our VAR model,

    $$\begin{pmatrix}s^{(1)}(t+1) \\ s^{(2)}(t+1) \end{pmatrix} = \begin{pmatrix} \phi^{(1)}_0 \\ \phi^{(2)}_0 \end{pmatrix} +  \begin{pmatrix}\phi_{1, 11} & \phi_{1, 12}\\ \phi_{1, 21} & \phi_{1, 22} \end{pmatrix} \begin{pmatrix}s^{(1)}(t) \\ s^{(2)}(t) \end{pmatrix} + \begin{pmatrix}\epsilon^{(1)} \\ \epsilon^{(2)} \end{pmatrix}.$$

    :param delta_t: step size of time in each iteration
    :param phi0: pho_0 in the AR model
    :param phi1: pho_1 in the AR model
    :param epsilon: noise iterator, e.g., Gaussian noise
    :param initial_state: a dictionary of the initial state, e.g., `{"s": 1}`
    """

    delta_t: float
    phi0: np.ndarray
    phi1: np.ndarray
    epsilon: Iterator
    initial_state: np.ndarray


class VAR1Stepper:
    """Calculate the next values using VAR(1) model.

    :param model_params: the parameters of the VAR(1) model, e.g.,
        [`VAR1ModelParams`][eerily.data.generators.var.VAR1ModelParams]
    """

    def __init__(self, model_params):
        self.model_params = model_params
        self.current_state = copy.deepcopy(self.model_params.initial_state)

    def __iter__(self):
        return self

    def __next__(self):

        epsilon = next(self.model_params.epsilon)
        phi0 = self.model_params.phi0
        phi1 = self.model_params.phi1

        self.current_state = phi0 + np.matmul(phi1, self.current_state) + epsilon

        return copy.deepcopy(self.current_state)
