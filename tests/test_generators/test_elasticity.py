import numpy as np
import pandas as pd
import pytest

from eerily.generators.elasticity import ElasticityStepper, LinearElasticityParams
from eerily.generators.utils.base import ConstantIterator
from eerily.generators.utils.noises import GaussianNoise


@pytest.fixture
def seed():
    return 42


@pytest.fixture
def length():
    return 10


@pytest.fixture
def constant_elasticity():
    return ConstantIterator(constant=-3)


@pytest.fixture
def log_prices(length):
    return iter(range(length))


@pytest.fixture
def log_base_demand(length):
    return iter(range(length))


@pytest.fixture
def deterministic_elasticity_stepper(constant_elasticity, log_prices):
    initial_condition = {"log_demand": 3, "log_price": 0.5, "elasticity": None}

    lep = LinearElasticityParams(
        initial_state=initial_condition,
        log_prices=log_prices,
        elasticity=constant_elasticity,
        variable_names=["log_demand", "log_price", "elasticity"],
    )

    return ElasticityStepper(model_params=lep)


@pytest.fixture
def deterministic_base_demand_elasticity_stepper(
    constant_elasticity, log_prices, log_base_demand
):
    initial_condition = {"log_demand": 3, "log_price": 0.5, "elasticity": None}

    lep = LinearElasticityParams(
        initial_state=initial_condition,
        log_prices=log_prices,
        elasticity=constant_elasticity,
        log_base_demand=log_base_demand,
        variable_names=["log_demand", "log_price", "elasticity"],
    )

    return ElasticityStepper(model_params=lep)


def test_deterministic_elasticity_stepper(deterministic_elasticity_stepper, length):
    container = []
    for _ in range(length):
        container.append(next(deterministic_elasticity_stepper))

    container_truth = [
        {"log_price": 0, "log_demand": 4.5, "elasticity": -3},
        {"log_price": 1, "log_demand": 1.5, "elasticity": -3},
        {"log_price": 2, "log_demand": -1.5, "elasticity": -3},
        {"log_price": 3, "log_demand": -4.5, "elasticity": -3},
        {"log_price": 4, "log_demand": -7.5, "elasticity": -3},
        {"log_price": 5, "log_demand": -10.5, "elasticity": -3},
        {"log_price": 6, "log_demand": -13.5, "elasticity": -3},
        {"log_price": 7, "log_demand": -16.5, "elasticity": -3},
        {"log_price": 8, "log_demand": -19.5, "elasticity": -3},
        {"log_price": 9, "log_demand": -22.5, "elasticity": -3},
    ]

    assert container == container_truth


def test_deterministic_base_demand_elasticity_stepper(
    deterministic_base_demand_elasticity_stepper, length
):
    container = []
    for _ in range(length):
        container.append(next(deterministic_base_demand_elasticity_stepper))

    container_truth = [
        {"log_demand": 0, "log_price": 0, "elasticity": -3, "log_base_demand": 0},
        {"log_demand": -2, "log_price": 1, "elasticity": -3, "log_base_demand": 1},
        {"log_demand": -4, "log_price": 2, "elasticity": -3, "log_base_demand": 2},
        {"log_demand": -6, "log_price": 3, "elasticity": -3, "log_base_demand": 3},
        {"log_demand": -8, "log_price": 4, "elasticity": -3, "log_base_demand": 4},
        {"log_demand": -10, "log_price": 5, "elasticity": -3, "log_base_demand": 5},
        {"log_demand": -12, "log_price": 6, "elasticity": -3, "log_base_demand": 6},
        {"log_demand": -14, "log_price": 7, "elasticity": -3, "log_base_demand": 7},
        {"log_demand": -16, "log_price": 8, "elasticity": -3, "log_base_demand": 8},
        {"log_demand": -18, "log_price": 9, "elasticity": -3, "log_base_demand": 9},
    ]

    assert container == container_truth


@pytest.fixture
def gaussian_elasticity(seed):
    elasticity_mean = -3
    elasticity_std = 0.5
    return GaussianNoise(mu=elasticity_mean, std=elasticity_std, seed=seed)


@pytest.fixture
def stochastic_elasticity_stepper(gaussian_elasticity, log_prices):
    initial_condition = {"log_demand": 3, "log_price": 0.5, "elasticity": None}

    lep = LinearElasticityParams(
        initial_state=initial_condition,
        log_prices=log_prices,
        elasticity=gaussian_elasticity,
        variable_names=["log_demand", "log_price", "elasticity"],
    )

    return ElasticityStepper(model_params=lep)


def test_stochastic_elasticity_stepper(stochastic_elasticity_stepper, length):
    container = []
    for _ in range(length):
        container.append(next(stochastic_elasticity_stepper))

    container_truth = [
        {
            "log_price": 0,
            "log_demand": 4.423820730061392,
            "elasticity": -2.847641460122784,
        },
        {
            "log_price": 1,
            "log_demand": 0.9038286769411439,
            "elasticity": -3.519992053120248,
        },
        {
            "log_price": 2,
            "log_demand": -1.7209457251556275,
            "elasticity": -2.6247744020967714,
        },
        {
            "log_price": 3,
            "log_demand": -4.250663366960021,
            "elasticity": -2.529717641804393,
        },
        {
            "log_price": 4,
            "log_demand": -8.22618096128694,
            "elasticity": -3.9755175943269183,
        },
        {
            "log_price": 5,
            "log_demand": -11.877270714718097,
            "elasticity": -3.651089753431159,
        },
        {
            "log_price": 6,
            "log_demand": -14.813350513134456,
            "elasticity": -2.9360797984163574,
        },
        {
            "log_price": 7,
            "log_demand": -17.971471809306248,
            "elasticity": -3.158121296171791,
        },
        {
            "log_price": 8,
            "log_demand": -20.979872388058393,
            "elasticity": -3.0084005787521444,
        },
        {
            "log_price": 9,
            "log_demand": -24.406394351845183,
            "elasticity": -3.42652196378679,
        },
    ]

    pd.testing.assert_frame_equal(
        pd.DataFrame(container), pd.DataFrame(container_truth), check_like=True
    )
