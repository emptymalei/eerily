# Tutorials

## Generating Data and `Stepper`

Generating data using EERILY requires a predefined Stepper. A Stepper is an iterator that species what should happen based on the current states. Given a Stepper, we can iterate over it to get the time series. Some pseudo code like the following may help.

```python
stepper = MyStepper()
next(stepper)
```

EERILY ships a base class for a Stepper (i.e., [`eerily.generators.utils.stepper.BaseStepper`](../references/generators/utils/stepper/#eerily.generators.utils.stepper.BaseStepper)). This base class takes in `model_params` as its argument.

`model_params` should be a dataclass that contains the essential information for the stepper to evolve in time. EERILY provide a base class called [`StepperModelParams`](../references/generators/utils/stepper/#eerily.generators.utils.stepper.StepperModelParams).
