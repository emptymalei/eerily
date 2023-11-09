from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np


class Choices:
    """Generating data by choosing from some given values.

    ```python
    elements = [0,1]
    c = Choices(elements=elements)
    next(c)
    ```

    :param elements: the elements to choose from
    :param seed: seed of the RNG for reproducibility
    """

    def __init__(
        self,
        elements: Sequence[Any],
        seed: Optional[float] = None,
    ):
        self.elements = elements
        self.indices = range(len(elements))

        self.rng = np.random.default_rng(seed=seed)

    def __next__(self) -> Any:
        idx = self.rng.choice(self.indices)
        return self.elements[idx]
