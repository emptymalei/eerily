import pytest

from tessa.data.generators.elasticity import ElasticityModel


@pytest.fixture
def elasticity_model():

    noise_std = 0.01
    noise_mu = 0.05
    true_elasticity = -3

    return ElasticityModel(
        elasticity=true_elasticity,
        noise_mu=noise_mu,
        noise_std=noise_std,
    )


def test_elasticity_model(elasticity_model):
    initial_condition = (0.5, 3)
    steps = 1000

    x, y = elasticity_model.generate(initial_condition=initial_condition, steps=steps)
