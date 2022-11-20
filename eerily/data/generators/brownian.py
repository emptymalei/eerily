import copy
from typing import Dict, Iterator, Optional, Sequence, Union

import numpy as np

from eerily.data.generators.stepper import BaseStepper


class GaussianForce:
    """A Gaussian stochastic force iterator.
    Each iteration returns a single sample from the corresponding
    Gaussian distribution.

    :param mu: mean of the Gaussian distribution
    :param std: standard deviation of the Gaussian distribution
    :param seed: seed for the random generator
    """

    def __init__(self, mu: float, std: float, seed: Optional[float] = None):
        self.mu = mu
        self.std = std
        self.rng = np.random.default_rng(seed=seed)

    def __next__(self) -> float:
        return self.rng.normal(self.mu, self.std)


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

        bms = BrownianMotionStepper(
            gamma=0, delta_t=0.1, force_densities=guassian_force, initial_state={"v": 0}
        )

        next(bms)
        ```

    :param gamma: the damping factor $\gamma$ of the Brownian motion.
    :param delta_t: the minimum time step $\Delta t$.
    :param force_densities: the stochastic force densities, e.g. [`GaussianForce`][eerily.data.generators.brownian.GaussianForce].
    :param initial_state: the initial velocity $v(0)$.
    """

    def __init__(
        self,
        gamma: float,
        delta_t: float,
        force_densities: Iterator,
        initial_state: Dict[str, float],
    ):
        self.gamma = gamma
        self.delta_t = delta_t
        self.forece_densities = copy.deepcopy(force_densities)
        self.current_state = copy.deepcopy(initial_state)

    def __iter__(self):
        return self

    def __next__(self) -> Dict[str, float]:

        force_density = next(self.forece_densities)
        v_current = self.current_state["v"]

        v_next = v_current + force_density * self.delta_t - self.gamma * v_current * self.delta_t

        self.current_state["force_density"] = force_density
        self.current_state["v"] = v_next

        return copy.deepcopy(self.current_state)
