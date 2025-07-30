import logging
import os
import random
import time
from contextlib import contextmanager
from functools import wraps
from typing import TypeVar

import numpy as np
import torch

logger = logging.getLogger(__name__)

T = TypeVar("T")


@contextmanager
def log_duration(logger, msg=None, level=logging.INFO):
    if msg is not None:
        logger.log(level, "Starting: '%s'", msg)
    t0 = time.monotonic()
    yield
    t1 = time.monotonic()
    logger.log(level, "Finished: '%s' (elapsed: %.2gs)", msg, t1 - t0)


def log_node(func=None, /, *, level=logging.INFO):
    def wrap(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            logger = logging.getLogger("log_node")

            logger.log(level, "Starting node: '%s'", f.__name__)
            t0 = time.monotonic()
            out = f(*args, **kwargs)
            t1 = time.monotonic()

            if logger.isEnabledFor(level):
                elapsed = t1 - t0
                mins, secs = divmod(elapsed, 60)
                if mins:
                    logger.log(
                        level,
                        "Finished node: '%s' (elapsed: %dm %.0fs)",
                        f.__name__,
                        int(mins),
                        secs,
                    )
                else:
                    logger.log(
                        level, "Finished node: '%s' (elapsed: %.2gs)", f.__name__, secs
                    )

            return out

        return wrapped

    if func is not None:
        return wrap(func)
    else:
        return wrap


def set_random_seed(seed: int):
    """Set the pytorch, numpy and python random seeds simultaneously"""
    logger.info(f"Setting global seeds to {seed}")
    if not (isinstance(seed, int) and 1 <= seed <= 4294967295):
        raise ValueError(
            f"Expected seed to be an integer between 1 and 4294967295. Got {seed!r}."
        )

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if "PYTHONHASHSEED" not in os.environ:
        os.environ["PYTHONHASHSEED"] = str(seed)


def make_torch_std_grid(n_points_per_axis, n_dimensions, tkwargs=None):
    """Return a `nxd` matrix of points on a standard [0, 1]^d grid

    Example:
        >>> make_torch_std_grid(3, 2)
        tensor([[0.0000, 0.0000],
                [0.0000, 0.5000],
                [0.0000, 1.0000],
                [0.5000, 0.0000],
                [0.5000, 0.5000],
                [0.5000, 1.0000],
                [1.0000, 0.0000],
                [1.0000, 0.5000],
                [1.0000, 1.0000]])
    """
    tkwargs = tkwargs or {}

    if n_dimensions <= 0:
        raise ValueError(f"Expected n_dimensions >= 1. Got {n_dimensions}.")

    points = []
    for d in range(n_dimensions):
        for i in range(len(points)):
            points[i] = torch.tile(points[i], (n_points_per_axis,))
        newaxis = torch.linspace(0, 1, n_points_per_axis, **tkwargs)
        newaxis = torch.repeat_interleave(newaxis, n_points_per_axis ** len(points))
        points.append(newaxis)

    return torch.stack(points[::-1]).T


def is_power_of_2(n):
    if not isinstance(n, int):
        raise TypeError(f"Expected n to be an int. Got {type(n)}.")

    return (n & (n - 1) == 0) and n != 0


def as_tensor_or_none(x, tkwargs=None):
    if x is None:
        return None
    else:
        tkwargs = tkwargs or {}
        return torch.as_tensor(x, **tkwargs)
