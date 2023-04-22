from typing import Any


class ConstantIterator:
    """An iterator that emits constant values.

    ```python
    rate = 0.1
    pe = PoissonEvents(rate=rate)
    next(pe)
    ```

    :param constant: the constant value to be emmited.
    """

    def __init__(self, constant: Any):
        self.constant = constant

    def __next__(self) -> Any:
        return self.constant
