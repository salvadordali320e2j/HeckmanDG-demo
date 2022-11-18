
import random
import numpy
import torch


def fix_random_seed(s: int):
    random.seed(s)
    numpy.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
