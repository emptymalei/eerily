import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import chain
from typing import Any, Iterator, List, Optional

from loguru import logger


@dataclass(frozen=True)
class StepperParams:
    """Base Parameters for Stepper

    :param initial_state: the initial state, e.g., `np.array([1])`
    :param variable_name: variable names of the time series provided as a list.
    """

    initial_state: Any
    variable_names: List[Any]


class BaseStepper(ABC):
    """A framework to evolve a DGP to the next step"""

    def __init__(self, model_params: StepperParams, length: Optional[int] = None) -> None:
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

    def __add__(self, another_stepper) -> Iterator:
        return SequentialStepper([self, another_stepper])

    def __and__(self, another_stepper) -> Iterator:
        return MergedStepper([self, another_stepper])


class SequentialStepper(BaseStepper):
    def __init__(
        self,
        iterators: List[BaseStepper],
    ):
        self.iterators: List[BaseStepper] = []
        self._length = 0
        self._counter = 0
        for stepper in iterators:
            if isinstance(stepper, SequentialStepper):
                self.iterators.extend(stepper.iterators)
            elif isinstance(stepper, BaseStepper):
                self.iterators.append(stepper)
            else:
                raise TypeError("Please provide a list of steppers")

    def compute_step(self):
        for stepper in self.iterators:
            for _ in range(stepper.length):
                yield next(stepper)

    def __next__(self):
        yield from self.compute_step()

    @property
    def length(self):
        self._length = sum([stepper.length for stepper in self.iterators])
        return self._length

    def __len__(self):
        return self.length


class MergedStepper(BaseStepper):
    def __init__(
        self,
        iterators: List[BaseStepper],
    ):
        self.iterators: List[BaseStepper] = []
        self._length = 0
        self._counter = 0
        for stepper in iterators:
            if isinstance(stepper, MergedStepper):
                self.iterators.extend(stepper.iterators)
            elif isinstance(stepper, BaseStepper):
                self.iterators.append(stepper)
            else:
                raise TypeError("Please provide a list of steppers")

    def compute_step(self):
        if all([stepper.length is not None for stepper in self.iterators]):
            length = min([stepper.length for stepper in self.iterators])
        else:
            raise ValueError("length is not set")

        for _, vals in zip(range(length), self.iterators):
            combined = {}
            for val in vals:
                if isinstance(val, dict):
                    combined.update(val)
                else:
                    raise NotImplementedError("Please implement __and__ for your steppers")
            yield combined

    def __next__(self):
        yield from self.compute_step()

    @property
    def length(self):
        self._length = min([stepper.length for stepper in self.iterators])
        return self._length

    def __len__(self):
        return self.length
