import pytest
from botorch import manual_seed


@pytest.fixture(autouse=True)
def torch_manual_seed():
    seed = 1234
    with manual_seed(seed):
        yield seed
