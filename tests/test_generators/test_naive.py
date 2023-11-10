from eerily.generators.naive import (
    ConstantStepper,
    ConstStepperParams,
    SequenceStepper,
    SequenceStepperParams,
)


def test_constant_stepper():
    csp = ConstStepperParams(initial_state=[1], variable_names=["y"])
    cs = ConstantStepper(model_params=csp)

    assert next(cs) == {"y": 1}


def test_constant_stepper_length():
    csp = ConstStepperParams(initial_state=[1], variable_names=["y"])
    cs = ConstantStepper(model_params=csp, length=3)

    assert list(cs) == [{"y": 1}] * 3


def test_sequence_stepper():
    ssp = SequenceStepperParams(initial_state=[1], variable_names=["y"], step_sizes=[1])
    ss = SequenceStepper(model_params=ssp)

    assert next(ss) == {"y": 2}


def test_sequence_stepper_length():
    ssp = SequenceStepperParams(initial_state=[1], variable_names=["y"], step_sizes=[1])
    ss = SequenceStepper(model_params=ssp, length=5)

    assert list(ss) == [{"y": i} for i in range(2, 7)]
