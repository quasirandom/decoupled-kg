import os

import torch

LOG_FORMAT = "%(asctime)s: %(levelname)-8s - %(name)s - line %(lineno)3d - %(message)s"

# Note that the default dtype is also set to torch.double in main.py
TKWARGS = {"dtype": torch.double, "device": "cpu"}

SMOKE_TEST = bool(int(os.environ.get("SMOKE_TEST", 0)))
