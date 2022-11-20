import pandas as pd
import pytest

from eerily.data.generators.brownian import BrownianMotionStepper, GaussianForce


@pytest.fixture
def seed():
    return 42


@pytest.fixture
def length():
    return 10


@pytest.fixture
def guassian_force(seed):
    return GaussianForce(mu=0, std=1, seed=seed)


@pytest.fixture
def brownian_motion_stepper(guassian_force):

    return BrownianMotionStepper(gamma=0, delta_t=0.1, force_densities=guassian_force, initial_state={"v": 0})


def test_brownian_motion_stepper(brownian_motion_stepper, length):

    container = []
    for _ in range(length):
        container.append(next(brownian_motion_stepper))

    container_truth = [
        {"v": 0.030471707975443137, "force_density": 0.30471707975443135},
        {"v": -0.07352670264860642, "force_density": -1.0399841062404955},
        {"v": 0.0015184169320393154, "force_density": 0.7504511958064572},
        {"v": 0.09557488857116071, "force_density": 0.9405647163912139},
        {"v": -0.09952863029422294, "force_density": -1.9510351886538364},
        {"v": -0.22974658098045475, "force_density": -1.302179506862318},
        {"v": -0.21696254066372622, "force_density": 0.12784040316728537},
        {"v": -0.24858679989808444, "force_density": -0.3162425923435822},
        {"v": -0.25026691564851333, "force_density": -0.016801157504288795},
        {"v": -0.33557130840587135, "force_density": -0.85304392757358},
    ]

    pd.testing.assert_frame_equal(pd.DataFrame(container), pd.DataFrame(container_truth), check_like=True)
