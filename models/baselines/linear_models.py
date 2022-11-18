
import os
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F

from rich.progress import track
from torch.utils.data import DataLoader

from models.base import _LinearModelBase
from utils.optimization import configure_optimizer


class _LinearRegressionBase(_LinearModelBase):
    
    _keys_in_batch: typing.List[str] = ['x', 'y']
    
    def __init__(self, input_features: int, device: str = 'cpu') -> None:
        super(_LinearRegressionBase, self).__init__()
    
        self.input_features: int = input_features
        self.device: str = device
        self.hparams: dict = {}

    def compile(self):
        raise NotImplementedError

    def _init_model(self) -> None:
        self.model = nn.Linear(self.input_features, 1, bias=True)
        self.model.bias.data.fill_(1.)

    def _init_cuda(self) -> None:
        self.model.to(self.device)
    
    def _init_optimizer(self) -> None:
        cfg = {
            'params': self.model.parameters(),
            'name': self.hparams['optimizer'],
            'lr': self.hparams['lr'],
            'weight_decay': self.hparams.get('weight_decay', 0.)
        }

        if cfg['name'] in ('lbfgs', 'l-bfgs'):
            _ = cfg.pop('weight_decay')  # not supported by `torch.optim.LBFGS`
            cfg['max_iter'] = self.hparams.get('lbfgs_max_iter', 5)
            cfg['history_size'] = self.hparams.get('lbfgs_history_size', 100)

        self.optimizer = configure_optimizer(**cfg)

    def _is_valid_batch(self, batch: dict) -> bool:
        return all([c in batch.keys() for c in self._keys_in_batch])

    @torch.no_grad()
    def predict(self, x: torch.FloatTensor) -> torch.FloatTensor:
        """Helper function for model inference."""
        self.model.eval();
        return self.model(x).squeeze(1)  # (B,  ) <- (B, 1)

    @torch.no_grad()
    def evaluate(self,
                 loader: torch.utils.data.DataLoader,
                 metrics: typing.Dict[str, callable] = None,
                 **kwargs, ) -> typing.Dict[str, torch.FloatTensor]:
        """Helper function for model evaluation."""
        self.model.eval();

        # buffer
        results: typing.Dict[str, torch.Tensor] = {}

        # accumulate
        y_true, y_pred = list(), list()
        for i, batch in enumerate(loader):
            assert self._is_valid_batch(batch)
            y_true += [batch['y'].to(self.device)]
            y_pred += [self.predict(batch['x'].to(self.device))]
        y_true, y_pred = torch.cat(y_true, dim=0), torch.cat(y_pred, dim=0)
        assert (y_true.shape == y_pred.shape) and (y_pred.ndim == 1)

        # compute loss
        results['loss'] = F.mse_loss(y_pred, y_true, reduction='mean')

        # compute metrics (optional)
        if isinstance(metrics, dict):
            for metric_name, metric_fn in metrics.items():
                metric_val = metric_fn(y_pred, y_true)
                results[metric_name] = metric_val

        return results  # dict

    @property
    def coef_(self) -> torch.FloatTensor:
        return self.model.weight.data[0]

    @property
    def intercept_(self) -> torch.FloatTensor:
        return self.model.bias.data


class LinearRegression(_LinearRegressionBase):
    def __init__(self, input_features: int, device: str = 'cpu') -> None:
        super(LinearRegression, self).__init__(input_features=input_features,
                                               device=device)

    def compile(self,
                optimizer: str = 'lbfgs',
                lr: float = 0.1,
                weight_decay: float = 0.,
                lbfgs_max_iter: int = 5,
                lbfgs_history_size: int = 100, ) -> None:
        """
        Helper function for:
            1. creating model
            2. creating optimizers
            3. configuring the correct device.
        """

        self.hparams = dict(
            optimizer=optimizer,
            lr=lr,
            weight_decay=weight_decay,
            lbfgs_max_iter=lbfgs_max_iter,
            lbfgs_history_size=lbfgs_history_size
        )

        self._init_model()      # model
        self._init_optimizer()  # optimizer
        self._init_cuda()       # device setting

    def fit(self,
            train_set: torch.utils.data.Dataset,
            validation_set: torch.utils.data.Dataset,
            epochs: int,
            batch_size: int = None,
            description: str = 'Linear Regression',
            **kwargs, ) -> typing.Dict[str, torch.Tensor]:
        """
        Fit model. The model is evaluated every epoch (or iteration if full batch).
        """

        self.model.train()  # train mode

        # Use full batch if `batch_size` is not specified. In this case, the total number of
        # iterations will be equal to the number of training epochs.
        batch_size: int = len(train_set) if batch_size is None else batch_size
        assert isinstance(batch_size, int)

        # Create train loader
        train_loader = DataLoader(dataset=train_set,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=True)

        # Create validation loader
        validation_loader = DataLoader(dataset=validation_set,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       drop_last=False)

        results: typing.Dict[str, torch.Tensor] = {
            'train/loss': torch.zeros(epochs, device=self.device),
            'validation/loss': torch.zeros(epochs, device=self.device),
        }  # averaged later
        
        _desc: str = ":star_of_david: " + description
        for epoch in track(range(epochs), total=epochs, description=_desc):
            
            # train
            train_loss = self.train(train_loader)
            results['train/loss'][epoch] = train_loss

            # evaluate
            validation_loss = self.evaluate(validation_loader)['loss']
            results['validation/loss'][epoch] = validation_loss

        return results  # full training trajectory

    def train(self, loader: torch.utils.data.DataLoader) -> torch.FloatTensor:
        """Train single epoch."""

        losses = torch.zeros(len(loader), device=self.device)
        for i, batch in enumerate(loader):
            losses[i] = self.train_batch(batch)

        return losses.mean()

    def train_batch(self, batch: dict) -> torch.FloatTensor:
        """Fit model for a single batch."""
        
        assert self._is_valid_batch(batch)  # validate batch

        # fetch data
        x = batch['x'].to(self.device)
        y = batch['y'].to(self.device)

        def _closure():
            self.optimizer.zero_grad()
            y_pred = self.model(x).squeeze(1)
            loss = F.mse_loss(y_pred, y, reduction='mean')
            if isinstance(self.optimizer, torch.optim.LBFGS):
                # manual $l_2$ penalty (since torch.optim.LBFGS does not natively support it.)
                loss += self._l2_penalty() * self.hparams['weight_decay']
            loss.backward()
            return loss

        loss = self.optimizer.step(_closure)
        
        return loss.detach().clone()

    def _l2_penalty(self):
        l2_reg = 0.
        for param in self.model.parameters():
            l2_reg += torch.pow(param, 2).sum()
        
        return l2_reg