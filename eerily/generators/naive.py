import copy
from dataclasses import dataclass
from typing import Any, List

from eerily.generators.utils.stepper import BaseStepper, StepperParams


@dataclass(frozen=True)
class ConstStepperParams(StepperParams):
    """
    Parameters for constant model.

    ```python
    cp = ConstStepperParams(
        initial_state=["A"],
        variable_names=["name"],
    )
    ```
    """

    initial_state: List[Any]
    variable_names: List[Any]


class ConstantStepper(BaseStepper):
    """Generates constant values for an given initial condition.

    ```python
    csp = ConstStepperParams(initial_state=[1], variable_names=["y"])
    cs = ConstantStepper(model_params=csp)

    next(cs)
    ```
    """

    def compute_step(self):
        return dict(
            zip(self.model_params.variable_names, self.model_params.initial_state)
        )

    def __repr__(self) -> str:
        return (
            "ConstantStepper: \n"
            f"parameters: {self.model_params}\n"
            f"current_state: {self.current_state}"
        )


@dataclass(frozen=True)
class SequenceStepperParams(StepperParams):
    """
    Parameters for constant model.

    ```python
    cp = SequenceStepperParams(
        initial_state=[1],
        variable_names=["step"],
        step_sizes=[1]
    )
    ```
    """

    initial_state: List[Any]
    variable_names: List[Any]
    step_sizes: List[Any]


class SequenceStepper(BaseStepper):
    """Generates sequence values for an given initial condition.

    ```python
    csp = SequenceStepperParams(
        initial_state=[1],
        variable_names=["y"],
        step_sizes=[1]
    )
    cs = SequenceStepper(model_params=csp)

    next(cs)
    ```
    """

    def compute_step(self):
        new_state = []
        for c, i in zip(self.current_state, self.model_params.step_sizes):
            new_state.append(c + i)

        self.current_state = copy.deepcopy(new_state)
        return dict(zip(self.model_params.variable_names, self.current_state))

    def __repr__(self) -> str:
        return (
            "SequenceStepper: \n"
            f"parameters: {self.model_params}\n"
            f"current_state: {self.current_state}"
        )
