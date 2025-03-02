import random
import numpy as np
import torch


def set_seed(seed):
    """
    Set the random seed for reproducibility.
    :param seed: The random seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False