import numpy as np
import pytest

from eerily.generators.utils.base import ConstantIterator


@pytest.mark.parametrize(
    "constant, expected_val",
    [
        pytest.param(10, 10, id="constant_10"),
        pytest.param("a", "a", id="constant_a"),
        pytest.param([1, 2], [1, 2], id="constant_list_1_2"),
        pytest.param(np.array([1, 2]), np.array([1, 2]), id="constant_array_1_2"),
    ],
)
def test_constant_iterator(constant, expected_val):
    length = 10

    constant_iterator = ConstantIterator(constant=constant)

    results = [next(constant_iterator) for _ in range(length)]

    assert len(results) == length

    if isinstance(expected_val, np.ndarray):
        assert all([np.array_equal(i, expected_val) for i in results])
    else:
        assert all([i == expected_val for i in results])
