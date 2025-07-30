import pytest

from decoupledbo.modules.utils import is_power_of_2


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (8, True),
        (5, False),
        (4, True),
        (3, False),
        (2, True),
        (1, True),
        (0, False),
        (-3, False),
        (-4, False),
        (-5, False),
        (1.0, TypeError),
    ],
)
def test_is_power_of_2(n, expected):
    if isinstance(expected, type) and issubclass(expected, Exception):
        with pytest.raises(expected):
            is_power_of_2(n)
    else:
        assert is_power_of_2(n) is expected
