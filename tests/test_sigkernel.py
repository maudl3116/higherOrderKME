import sys

sys.path = sys.path[1:]  # A hack so that we always import the installed library.

import math
import numpy as np
import numpy.random as npr
import torch
import pytest

torch.manual_seed(1147481649)
torch.set_default_dtype(torch.float64)


devices = [cpu, gpu] = [torch.device('cpu'), torch.device('cuda')]