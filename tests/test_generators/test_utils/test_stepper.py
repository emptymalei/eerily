import pytest

from eerily.generators.utils.stepper import BaseStepper, StepperParams


class DummyStepperParams(StepperParams):
    pass


class DummyStepper(BaseStepper):
    def compute_step(self):
        return dict(
            zip(self.model_params.variable_names, self.model_params.initial_state)
        )


def test_base_stepper():
    stepper_params = DummyStepperParams(initial_state=[1], variable_names=["y"])
    stepper = DummyStepper(model_params=stepper_params)
    assert next(stepper) == {"y": 1}


@pytest.mark.parametrize(
    "length",
    [
        pytest.param(1, id="length_1"),
        pytest.param(5, id="length_5"),
        pytest.param(10, id="length_10"),
    ],
)
def test_base_stepper_with_length(length):
    stepper_params = DummyStepperParams(initial_state=[1], variable_names=["y"])
    stepper = DummyStepper(model_params=stepper_params, length=length)
    assert list(stepper) == [{"y": 1}] * length


def test_base_stepper_add():
    stepper_params_1 = DummyStepperParams(initial_state=[1], variable_names=["y1"])
    stepper_1 = DummyStepper(model_params=stepper_params_1, length=3)

    stepper_params_2 = DummyStepperParams(initial_state=[2], variable_names=["y2"])
    stepper_2 = DummyStepper(model_params=stepper_params_2, length=2)

    generator = stepper_1 + stepper_2

    assert list(generator) == [{"y1": 1}] * 3 + [{"y2": 2}] * 2


def test_base_stepper_add_3():
    stepper_params_1 = DummyStepperParams(initial_state=[1], variable_names=["y"])
    stepper_1 = DummyStepper(model_params=stepper_params_1, length=3)

    stepper_params_2 = DummyStepperParams(initial_state=[2], variable_names=["y"])
    stepper_2 = DummyStepper(model_params=stepper_params_2, length=2)

    stepper_params_3 = DummyStepperParams(initial_state=[3], variable_names=["y"])
    stepper_3 = DummyStepper(model_params=stepper_params_3, length=2)

    generator = stepper_1 + stepper_2 + stepper_3

    assert list(generator) == [{"y": 1}] * 3 + [{"y": 2}] * 2 + [{"y": 3}] * 2


def test_base_stepper_and():
    stepper_params_1 = DummyStepperParams(initial_state=[1], variable_names=["y1"])
    stepper_1 = DummyStepper(model_params=stepper_params_1, length=5)

    stepper_params_2 = DummyStepperParams(initial_state=[2], variable_names=["y2"])
    stepper_2 = DummyStepper(model_params=stepper_params_2, length=4)

    generator = stepper_1 & stepper_2

    assert list(generator) == [{"y1": 1, "y2": 2}] * 4


def test_base_stepper_and_3():
    stepper_params_1 = DummyStepperParams(initial_state=[1], variable_names=["y1"])
    stepper_1 = DummyStepper(model_params=stepper_params_1, length=5)

    stepper_params_2 = DummyStepperParams(initial_state=[2], variable_names=["y2"])
    stepper_2 = DummyStepper(model_params=stepper_params_2, length=4)

    stepper_params_3 = DummyStepperParams(initial_state=[3], variable_names=["y3"])
    stepper_3 = DummyStepper(model_params=stepper_params_3, length=4)

    generator = stepper_1 & stepper_2 & stepper_3

    assert list(generator) == [{"y1": 1, "y2": 2, "y3": 3}] * 4


def test_base_stepper_and_add():
    stepper_params_1 = DummyStepperParams(initial_state=[1], variable_names=["y1"])
    stepper_1 = DummyStepper(model_params=stepper_params_1, length=5)

    stepper_params_2 = DummyStepperParams(initial_state=[2], variable_names=["y2"])
    stepper_2 = DummyStepper(model_params=stepper_params_2, length=4)

    stepper_params_3 = DummyStepperParams(
        initial_state=[3, 4], variable_names=["y3", "y4"]
    )
    stepper_3 = DummyStepper(model_params=stepper_params_3, length=2)

    generator = (stepper_1 & stepper_2) + stepper_3

    assert list(generator) == [{"y1": 1, "y2": 2}] * 4 + [{"y3": 3, "y4": 4}] * 2
