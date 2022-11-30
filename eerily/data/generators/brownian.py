import copy
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Sequence, Union

import numpy as np

from eerily.data.generators.stepper import BaseStepper


@dataclass
class BrownianMotionParams:
    """
    Parameters for Brownian motion

    :param gamma: the damping factor $\gamma$ of the Brownian motion.
    :param delta_t: the minimum time step $\Delta t$.
    :param force_densities: the stochastic force densities, e.g. [`GaussianNoise`][eerily.data.generators.noise.GaussianNoise].
    :param initial_state: the initial velocity $v(0)$.
    """

    gamma: float
    delta_t: float
    force_densities: Iterator
    initial_state: Dict[str, float]


class BrownianMotionStepper(BaseStepper):
    """Calculates the next step in a brownian motion.

    ??? note "Brownian Motion"

        Macroscopically, Brownian Motion can be described by the notion of random forces on the particles,

        $$\\frac{d}{dt} v(t) + \gamma v(t) = R(t),$$

        where $v(t)$ is the velocity at time $t$ and $R(t)$ is the stochastic force density from the reservoir particles.

        To simulate it numerically, we rewrite

        $$\\frac{d}{dt} v(t) + \gamma v(t) = R(t),$$

        as

        $$\Delta v (t+1) = R(t) \Delta t - \gamma v(t) \Delta t$$


    !!! example "Example Code"

        ```python
        guassian_force = GaussianForce(mu=0, std=1, seed=seed)
        bm_params = BrownianMotionParams(
            gamma=0, delta_t=0.1, force_densities=guassian_force, initial_state={"v": 0}
        )

        bms = BrownianMotionStepper(
            model_params = bm_params
        )

        next(bms)
        ```

    :param model_params: a dataclass that contains the necessary parameters for the model.
        e.g., [`BrownianMotionParams`][eerily.data.generators.brownian.BrownianMotionParams]
    """

    def __init__(self, model_params: Any):
        self.model_params = model_params
        self.current_state = copy.deepcopy(self.model_params.initial_state)
        self.forece_densities = self.model_params.force_densities

    def __iter__(self):
        return self

    def __next__(self) -> Dict[str, float]:

        force_density = next(self.forece_densities)
        v_current = self.current_state["v"]

        v_next = (
            v_current
            + force_density * self.model_params.delta_t
            - self.model_params.gamma * v_current * self.model_params.delta_t
        )

        self.current_state["force_density"] = force_density
        self.current_state["v"] = v_next

        return copy.deepcopy(self.current_state)
