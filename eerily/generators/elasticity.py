import copy
from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Sequence, Union

import pandas as pd
from loguru import logger

from eerily.generators.utils.stepper import BaseStepper, StepperParams


@dataclass(frozen=True)
class LinearElasticityParams(StepperParams):
    """
    Parameters for constant elasticity model.

    ```python
    length = 10

    lep = LinearElasticityParams(
        initial_state={"log_demand": 3, "log_price": 0.5, "elasticity": None},
        log_prices=iter(range(length)),
        elasticity=iter([-3] * length),
        variable_names=["log_demand", "log_price", "elasticity"],
    )
    ```

    !!! warning "Initial Condition"
        Initial condition is a dictionary with at least two keys `sale` and `price`.

        Note that the initial condition is NOT returned in the iterator.


    :param elasticity: an iterator that generates the elasticity to be used for each step
    :param log_prices: an iterator that generates the log prices in each step
    """

    elasticity: Iterator
    log_prices: Iterator
    log_base_demand: Optional[Iterator] = None

    def __post_init__(self):
        if self.initial_state is None:
            self.initial_state = {"log_demand": 1, "log_price": 10, "elasticity": None}
        if self.variable_names is None:
            self.variable_names = ["log_demand", "log_price", "elasticity"]


class ElasticityStepper(BaseStepper):
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
    initial_condition = {"log_price": 1, "log_sales": 10, "elasticity": None}
    ```

    For a deterministic model, we have

    ```python
    length = 10
    elasticity = iter([-3] * length)
    log_prices = iter(range(length))

    initial_condition = {"log_demand": 3, "log_price": 0.5, "elasticity": None}

    lep = LinearElasticityParams(
        initial_state=initial_condition,
        log_prices=log_prices,
        elasticity=elasticity,
        variable_names=["log_demand", "log_price", "elasticity"],
    )

    es = ElasticityStepper(model_params=lep)

    next(es)
    ```

    We have utils in [`eerily.generators.utils`][eerily.generators.utils]
    to help the user creating elasticty and log prices generators.
    For example, we can create a constant iterator using
    [`ConstantIterator`][eerily.generators.utils.base.ConstantIterator]


    """

    def compute_step(self):
        elasticity = next(self.model_params.elasticity)

        current_log_price = self.current_state["log_price"]
        current_log_demand = self.current_state["log_demand"]

        next_log_price = next(self.model_params.log_prices)

        if self.model_params.log_base_demand is None:
            next_log_demand = current_log_demand + elasticity * (
                next_log_price - current_log_price
            )
        else:
            next_log_base_demand = next(self.model_params.log_base_demand)
            next_log_demand = next_log_base_demand + elasticity * next_log_price
            self.current_state["log_base_demand"] = next_log_base_demand

        self.current_state["log_demand"] = next_log_demand
        self.current_state["log_price"] = next_log_price
        self.current_state["elasticity"] = elasticity

        return copy.deepcopy(self.current_state)

    def __repr__(self) -> str:
        return (
            "ElasticityStepper: \n"
            f"parameters: {self.model_params}\n"
            f"current_state: {self.current_state}"
        )
