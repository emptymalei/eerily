import random
import copy
from dataclasses import dataclass
from collections import namedtuple
from typing import NamedTuple, Tuple
from tessa.data.generators.stepper import Stepper
from typing import Dict, Iterator, Iterable, Union, Optional
import numpy as np
from loguru import logger


@dataclass
class ElasticityData:
    elasticity: list
    prices: list
    sales: list

    def append(self, new_data: Dict[str, float]) -> None:
        """Append new data point to the existing data

        :param new_data:
        """
        self.elasticity.append(new_data["elasticity"])
        self.prices.append(new_data["prices"])
        self.sales.append(new_data["sales"])


class ElasticityStepper(Stepper):
    """Generates the next time step for an given initial condition.

    We use the following formula to generate the data

    $$
    \ln Q' = \ln Q + \epsilon (\ln P' - \ln P)
    $$

    Define new log transformed variables to make this a linear relation

    $$
    y' = y + \epsilon (x' - x).
    $$

    For example, with initial condition

    ```
    initial_condition = {"price": 0, "sales": 10}
    ```

    For a deterministic model, we have

    ```python
    elasticity = [-3] * (length -1)
    initial_condition = {"price: 0.5, "sale": 3}
    prices = range(10)

    es = ElasticityStepper(
        initial_condition=initial_condition,
        elasticity=elasticity,
        prices=prices
    )

    next(es)
    ```

    !!! warning "Initial Condition"
        Initial condition is a dictionary with at least two keys `sale` and `price`.

        Note that the initial condition is NOT returned in the iterator.

    """

    def __init__(
        self,
        initial_condition: Dict[str, float],
        elasticity: Union[Iterator, Iterable],
        prices: Union[Iterator, Iterable],
        length: Optional[int] = None,
    ):
        self.initial_condition = copy.deepcopy(initial_condition)
        self.current_state = copy.deepcopy(initial_condition)
        self.elasticity = elasticity
        self.prices = prices

        self.length = length

        if not isinstance(self.prices, Iterator):
            self.length = len(self.prices)
            self.prices = iter(self.prices)

        if not isinstance(self.elasticity, Iterator):
            elasticity_length = len(self.elasticity)
            self.elasticity = iter(self.elasticity)
            if (self.length is not None) and (elasticity_length != self.length):
                logger.warning(
                    f"elasticity length {elasticity_length} is different "
                    f"from prices length {self.length}; "
                    "Setting length to the min of the two."
                )
            self.length = min(self.length, elasticity_length)

    def __iter__(self):
        return self

    def __next__(self):

        price = next(self.prices)
        elasticity = next(self.elasticity)
        sale = self.current_state["sale"] + elasticity * (
            price - self.current_state["price"]
        )

        self.current_state["sale"] = sale
        self.current_state["price"] = price
        self.current_state["elasticity"] = elasticity

        return copy.deepcopy(self.current_state)

    def __repr__(self) -> str:
        return (
            "ElasticityStepper: \n"
            f"initial_condition: {self.initial_condition}\n"
            f"current_state: {self.current_state}"
        )
