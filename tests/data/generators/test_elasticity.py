from re import X
import pytest
import numpy as np
import random

from tessa.data.generators.elasticity import ElasticityModel

SEED = 42


@pytest.fixture
def elasticity_model():

    noise_std = 0.01
    noise_mu = 0.05
    true_elasticity = -3

    return ElasticityModel(
        elasticity=true_elasticity,
        noise_mu=noise_mu,
        noise_std=noise_std,
        rng=np.random.default_rng(seed=SEED)
    )


def test_elasticity_model(elasticity_model):
    initial_condition = (0.5, 3)
    steps = 10

    random.seed(SEED)
    x, y = elasticity_model.generate(initial_condition=initial_condition, steps=steps)

    x_expected = np.array([0.5, 0.1, 0. , 0.4, 0.3, 0.3, 0.2, 0.1, 0.1, 0.6])
    y_expected = np.array([3.        , 4.2636566 , 4.57553665, 3.30653124, 3.62435293, 3.62435293, 3.93544639, 4.25082992, 4.25082992, 2.67608193])

    assert np.allclose(x, x_expected, rtol=1e-2, atol=1e-2)
    assert np.allclose(y, y_expected)
