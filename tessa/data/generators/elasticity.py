import random
from collections import namedtuple
from typing import NamedTuple, Tuple

import numpy as np


class ElasticityModel:
    """A elasticty data generator

    We use the following formula to generate the data

    $$
    \ln Q' = \ln Q + \epsilon (\ln P' - \ln P)
    $$

    Define new log transformed variables to make this a linear relation

    $$
    y' = y + \epsilon (x' - x).
    $$

    ```python

    noise_std = 0.01
    noise_mu = 0.05
    true_elasticity=-3

    em = ElasticityModel(
        elasticity=true_elasticity,
        noise_mu=noise_mu,
        noise_std=noise_std,
    )

    initial_condition = (0.5, 3)
    steps = 1000

    em(initial_condition=initial_condition, steps=steps)
    ```
    """

    def __init__(
        self,
        elasticity: float,
        noise_mu: float,
        noise_std: float,
        covariate_levels: np.ndarray = np.linspace(0, 0.7, 8),
        rng: np.random.Generator = np.random.default_rng(),
    ):

        self.elasticity = elasticity
        self.rng = rng
        self.covariate_levels = covariate_levels
        self.noise_std = noise_std
        self.noise_mu = noise_mu

        Meta = namedtuple(
            "Meta",
            [
                "elasticity",
                "covariate_levels",
                "noise_mu",
                "noise_std",
            ],
        )

        self.meta = Meta(
            elasticity=elasticity,
            covariate_levels=covariate_levels,
            noise_mu=noise_mu,
            noise_std=noise_std,
        )

    def generate(
        self,
        initial_condition: Tuple[float, float],
        steps: int,
    ) -> NamedTuple:

        x_series = [initial_condition[0]]
        y_series = [initial_condition[1]]

        epsilon_noise = self.rng.normal(self.noise_mu, self.noise_std, steps)
        for i in range(steps - 1):
            xp = random.choice(self.covariate_levels)
            yp = y_series[-1] + self.elasticity * (1 + epsilon_noise[i]) * (xp - x_series[-1])
            x_series.append(xp)
            y_series.append(yp)

        Data = namedtuple("Data", ["x", "y"])

        return Data(x=np.array(x_series), y=np.array(y_series))

    def __call__(
        self,
        initial_condition: Tuple[float, float],
        steps: int,
    ) -> NamedTuple:
        return self.generate(initial_condition, steps)
