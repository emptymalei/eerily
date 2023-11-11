from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Optional, Union

from loguru import logger


class StepperOperator:
    """Allowing `&` and `+` for the steppers."""

    def __add__(
        self,
        another_stepper: Union[
            StepperOperator, BaseStepper, SequentialStepper, MergedStepper
        ],
    ) -> SequentialStepper:
        return SequentialStepper([self, another_stepper])

    def __and__(
        self,
        another_stepper: Union[
            StepperOperator, BaseStepper, SequentialStepper, MergedStepper
        ],
    ) -> MergedStepper:
        return MergedStepper([self, another_stepper])


@dataclass(frozen=True)
class StepperParams:
    """Base Parameters for Stepper

    :param initial_state: the initial state, e.g., `np.array([1])`
    :param variable_name: variable names of the time series provided as a list.
    """

    initial_state: Any
    variable_names: List[Any]


class BaseStepper(ABC, StepperOperator):
    """A framework to evolve a DGP to the next step"""

    def __init__(
        self, model_params: StepperParams, length: Optional[int] = None
    ) -> None:
        self.model_params = model_params
        self.current_state = copy.deepcopy(self.model_params.initial_state)
        self.length = length
        self._counter = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.length is None:
            logger.warning("length is not set")
            self._counter += 1
            return self.compute_step()
        else:
            if self._counter < self.length:
                self._counter += 1
                return self.compute_step()
            else:
                raise StopIteration

    def __len__(self):
        return self.length

    def get_iterator(self):
        return self.__iter__()

    def __str__(self) -> str:
        return f"Model Parameters: {self.model_params}"

    @abstractmethod
    def compute_step(self):
        pass


class SequentialStepper(StepperOperator):
    def __init__(
        self,
        iterators: List[
            Union[StepperOperator, BaseStepper, SequentialStepper, MergedStepper]
        ],
    ):
        self.iterators: List[
            Union[StepperOperator, BaseStepper, SequentialStepper, MergedStepper]
        ] = []
        self._length = 0
        for stepper in iterators:
            if isinstance(stepper, SequentialStepper):
                self.iterators.extend(stepper.iterators)
            elif isinstance(stepper, BaseStepper):
                self.iterators.append(stepper)
            elif isinstance(stepper, MergedStepper):
                self.iterators.append(stepper)
            else:
                raise TypeError("Please provide a list of steppers")

    def __iter__(self):
        for stepper in self.iterators:
            for _ in range(stepper.length):
                yield from stepper

    @property
    def length(self):
        self._length = sum([stepper.length for stepper in self.iterators])
        return self._length

    def __len__(self):
        return self.length


class MergedStepper(StepperOperator):
    def __init__(
        self,
        iterators: List[
            Union[StepperOperator, BaseStepper, SequentialStepper, MergedStepper]
        ],
    ):
        self.iterators: List[
            Union[StepperOperator, BaseStepper, SequentialStepper, MergedStepper]
        ] = []
        self._length = 0
        self._counter = 0
        for stepper in iterators:
            if isinstance(stepper, MergedStepper):
                self.iterators.extend(stepper.iterators)
            elif isinstance(stepper, BaseStepper):
                self.iterators.append(stepper)
            elif isinstance(stepper, SequentialStepper):
                self.iterators.append(stepper)
            else:
                raise TypeError("Please provide a list of steppers")

    def __iter__(self):
        if all([stepper.length is not None for stepper in self.iterators]):
            length = min([stepper.length for stepper in self.iterators])
        else:
            raise ValueError("length is not set")

        for vals in zip(range(length), *self.iterators):
            idx = vals[0]
            iter_values = vals[1:]
            combined = {}
            for val in iter_values:
                if isinstance(val, dict):
                    combined.update(val)
                else:
                    raise NotImplementedError(
                        "Please implement __and__ for your steppers"
                    )
            yield combined
            self._counter = idx

    @property
    def length(self):
        self._length = min([stepper.length for stepper in self.iterators])
        return self._length

    def __len__(self):
        return self.length
