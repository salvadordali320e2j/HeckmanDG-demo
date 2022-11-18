
import typing

import torch.optim as optim

from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


def configure_optimizer(params: typing.Iterable,
                        name: str,
                        lr: float,
                        weight_decay: typing.Optional[float] = 0.,
                        **kwargs) -> Optimizer:
    """
    Configure pytorch optimizer.
    Arguments:
        params: module parameters.
        name: str.
        lr: float.
        weight_decay: float.
    """

    name: str = name.lower()  # e.g., SGD -> sgd
    if name == 'sgd':
        return optim.SGD(
            params=params,
            lr=lr,
            weight_decay=weight_decay,
            momentum=kwargs.get('momentum', 0.9),
        )
    elif name == 'adam':
        return optim.Adam(
            params=params,
            lr=lr,
            weight_decay=weight_decay,
            betas=kwargs.get('betas', (0.9, 0.999)),
            amsgrad=False,
        )
    elif name == 'adamw':
        return optim.AdamW(
            params=params,
            lr=lr,
            weight_decay=weight_decay,
            betas=kwargs.get('betas', (0.9, 0.999)),
        )
    elif (name == 'lbfgs') or (name == 'l-bfgs'):
        return optim.LBFGS(
            params=params,
            lr=lr,
            line_search_fn='strong_wolfe',
            history_size=kwargs.get('history_size', 10),
            max_iter=kwargs.get('max_iter', 5),
        )
    else:
        raise NotImplementedError(
            "Currently only supports one of [sgd, adam, adamw, lbfgs (or l-bfgs)]."
        )
