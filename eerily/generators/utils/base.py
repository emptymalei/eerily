from typing import Any


class ConstantIterator:
    """An iterator that emits constant values.

    ```python
    pe = ConstantIterator(constant=1)
    next(pe)
    ```

    :param constant: the constant value to be emmited.
    """

    def __init__(self, constant: Any):
        self.constant = constant

    def __iter__(self):
        return self

    def __next__(self) -> Any:
        return self.constant
