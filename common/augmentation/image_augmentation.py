import numpy as np
import torch
import random
from typing import Optional, Tuple

class NoiseInjection:
    def __init__(self, noise_level: float=1e-7):
        self.noise_level = noise_level

    def __call__(self, x):
        return x + torch.randn(x.shape).to(x.device)*self.noise_level

