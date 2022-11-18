
import typing

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from rich.progress import track
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy, auroc
try:
    from torchmetrics.functional import f1
except ImportError:
    from torchmetrics.functional import f1_score as f1

from utils.optimization import configure_optimizer


class _HeckmanRegressionBase(object):
    _keys_in_batch: list = ['x', 'y']
    def __init__(self, input_features: int, device: str = 'cpu', ) -> None:
        super(_HeckmanRegressionBase, self).__init__()

        self.input_features: int = input_features
        self.device: str = device
        self.hparams: dict = {}

    def compile(self):
        raise NotImplementedError

    def _init_model(self) -> None:
        raise NotImplementedError

    def _init_cuda(self) -> None:
        raise NotImplementedError

    def _init_optimizer(self) -> None:
        raise NotImplementedError

    def fit(self):
        raise NotImplementedError

    @staticmethod
    def initialize_valid_rho(*size, device: str = 'cpu'):
        """Initialize correlation parameter."""
        return torch.rand(*size, device=device).uniform_(-0.9, 0.9)

    def _is_valid_batch(self, batch: dict, exclude: list = []) -> bool:
        """"Validates batch."""
        for c in self._keys_in_batch:
            if c in exclude:
                continue
            if c not in batch.keys():
                return False
        return True

    @torch.no_grad()
    def predict(self, x: torch.FloatTensor) -> torch.FloatTensor:
        self.outcome_model.eval();
        return self.outcome_model(x).squeeze(1)

    @torch.no_grad()
    def evaluate(self,
                 loader: torch.utils.data.DataLoader,
                 metrics: typing.Dict[str, callable] = None,
                 **kwargs, ) -> typing.Dict[str, torch.FloatTensor]:

        self.outcome_model.eval();

        # buffer
        results: typing.Dict[str, torch.Tensor] = {}

        # accumulate
        y_true, y_pred = list(), list()
        for i, batch in enumerate(loader):
            assert self._is_valid_batch(batch)
            y_true += [batch['y'].to(self.device)]
            y_pred += [self.predict(batch['x'].to(self.device))]
        y_true, y_pred = torch.cat(y_true, dim=0), torch.cat(y_pred, dim=0)

        # compute loss
        results['loss'] = F.mse_loss(y_pred, y_true, reduction='mean')

        # compute metrics
        if isinstance(metrics, dict):
            for metric_name, metric_fn in metrics.items():
                metric_val = metric_fn(y_pred, y_true)
                results[metric_name] = metric_val

        return results

    @property
    def rho(self):
        return torch.tanh(self._rho)

    @property
    def sigma(self):
        return self._sigma

    @property
    def coef_(self):
        return self.outcome_model.weight.data[0]

    @property
    def intercept_(self):
        return self.outcome_model.bias.data


class _HeckmanDGRegressionBase(_HeckmanRegressionBase):
    _keys_in_batch: str = ['x', 'y', 'domain']
    def __init__(self,
                 input_features: int,
                 num_train_domains: int = 2,
                 device: str = 'cpu', ) -> None:
        super(_HeckmanDGRegressionBase, self).__init__(input_features=input_features,
                                                       device=device)
        
        if num_train_domains < 2:
            raise ValueError("Argument `num_train_domains` must be greater or equal to 2.")
        self.num_train_domains: int = num_train_domains

    def compile(self):
        raise NotImplementedError

    def _init_model(self) -> None:
        K: int = self.num_train_domains
        self.selection_model = nn.Linear(self.input_features, K, bias=True)  # K probits
        self.outcome_model = nn.Linear(self.input_features, 1, bias=True)    # 1 linear
        self.outcome_model.bias.data.fill_(0.)
        self._rho = nn.Parameter(
            data=torch.atanh(self.initialize_valid_rho(K, device=self.device)),
            requires_grad=True,
        )
        self._sigma = nn.Parameter(
            data=torch.ones(1, device=self.device),
            requires_grad=True,
        )

    def _init_cuda(self) -> None:
        self.selection_model.to(self.device)
        self.outcome_model.to(self.device)

    def _init_optimizer(self) -> None:
        raise NotImplementedError

    def evaluate_selection_model(self, data_loader: torch.utils.data.DataLoader) -> typing.Dict[str, torch.Tensor]:
        """Helper function for evaluating selection model performance."""
        
        # buffers
        s_pred, s_true = list(), list()
        
        # accumulate
        for _, batch in enumerate(data_loader):
            assert self._is_valid_batch(batch, exclude=['y'])
            s_pred += [self.selection_model(batch['x'].to(self.device))]
            s_true += [batch['domain'].to(self.device)]
        
        s_pred = torch.cat(s_pred, dim=0)         # (N, K)
        s_true = torch.cat(s_true, dim=0)         # (N, K)
        assert s_pred.shape == s_true.shape

        # probits to probabilities
        s_pred = torch.distributions.Normal(loc=0., scale=1.).cdf(s_pred)
        
        # compute metrics
        results = dict()
        results['accuracy'] = accuracy(s_pred, s_true, threshold=.5, num_classes=2, multiclass=False)
        results['f1'] = f1(s_pred, s_true, threshold=.5, num_classes=2, multiclass=False, average='macro')

        return results  # dictionary

    @staticmethod
    def joint_loss(y_pred: torch.FloatTensor,
                   y_true: torch.FloatTensor,
                   s_pred: torch.FloatTensor,
                   s_true: typing.Union[torch.FloatTensor, torch.LongTensor],
                   rho: torch.FloatTensor,
                   sigma: torch.FloatTensor,
                   **kwargs, ) -> torch.FloatTensor:
        """
        Joint loss function.
        This function is used in the following cases:
            1) joint optimization of selection and outcome models.
            2) optimization of correlation / sigma parameters.
        """

        assert (y_pred.ndim == 1) and (y_pred.shape == y_true.shape),  "(N,  )"
        assert (s_pred.ndim == 2) and (s_pred.shape == s_true.shape),  "(N, K)"
        assert (rho.ndim == 1) and (rho.__len__() == s_pred.shape[1]), "(K,  )"

        _eps: float = 1e-7
        _normal = torch.distributions.Normal(loc=0., scale=1.)

        # 1. Loss for unselected-ness: - \log p(S_l = 0, where l \neq k)
        #   Shape; (N + ?,  );
        s_pred_in_probs = _normal.cdf(s_pred)
        loss_not_selected = F.binary_cross_entropy(
            s_pred_in_probs, s_true.float(), weights=None, reduction='none'
        )  # (N, K)

        # 2. Loss for selected-ness: - \log p(y, S_k = 1 | x)
        #    Since `s_true[i]` is not necessarily a one-hot vector; we need to
        #    find the {row, col} indices where it is non-zero, and use them
        #    to create duplicates of {y_pred, y_true, s_pred, rho}.
        row_idx, col_idx = s_true.gt(0).long().nonzero(as_tuple=True)
        y_pred = y_pred[row_idx]             # (N + ?,  )
        y_true = y_true[row_idx]             # (N + ?,  )
        s_pred_k = s_pred[row_idx, col_idx]  # (N + ?,  )
        rho = rho[col_idx]                   # (N + ?,  )

        # - \log p(y, S_k = 1 | x) = -\log p(y | x) - \log p(S_k = 1 | y, x)
        #   Shape: (N + ?,  ); where `?` is due to some multi-hot vectors of `s_true[i]`
        loss_selected = - torch.log(
            _normal.cdf(
                (s_pred_k + rho * (y_true - y_pred).div(_eps + sigma)) / torch.sqrt(_eps + 1 - rho ** 2)
            ) + _eps
        ) + 0.5 * (
            torch.log(2 * torch.pi * (sigma ** 2)) \
                + F.mse_loss(y_pred, y_true, reduction='none').div(_eps + sigma ** 2)
        )

        # bypass possible {NaN, -inf, +inf};
        loss_not_selected = torch.nan_to_num(
            loss_not_selected, nan=0., posinf=0., neginf=0.,
        )  # (N    , K)
        loss_not_selected = loss_not_selected.mean(dim=1, keepdim=False)  # (N,  )
        loss_selected = torch.nan_to_num(
            loss_selected, nan=0., posinf=0., neginf=0.,
        )  # (N + ?,  )

        # aggregate the two losses
        return torch.cat([loss_not_selected, loss_selected], dim=0).mean()

    @staticmethod
    def loss_for_training_selection_model(s_pred: torch.FloatTensor,
                                          s_true: torch.FloatTensor,
                                          **kwargs, ) -> torch.FloatTensor:
        """Domain selection model loss function."""
        
        assert (s_pred.shape == s_true.shape), "(N, K)"
        s_pred_in_probs = torch.distributions.Normal(loc=0., scale=1.).cdf(s_pred)
        
        return F.binary_cross_entropy(s_pred_in_probs, s_true.float())

    @staticmethod
    def loss_for_training_outcome_model(y_pred: torch.FloatTensor,
                                        y_true: torch.FloatTensor,
                                        s_pred: torch.FloatTensor,
                                        s_true: typing.Union[torch.FloatTensor, torch.LongTensor],
                                        rho: torch.FloatTensor,
                                        sigma: typing.Union[float, torch.FloatTensor],
                                        **kwargs, ) -> torch.FloatTensor:

        assert (y_pred.shape == y_true.shape) and (y_pred.ndim == 1), "y: (N,  )"
        assert (s_pred.shape == s_true.shape) and (s_pred.ndim == 2), "s; (N, K)"
        assert (rho.ndim == 1) and (len(rho) == s_pred.shape[1]), "rho; (K,  )"

        _eps: float = 1e-7
        _normal = torch.distributions.Normal(loc=0., scale=1.)

        # Since `s_true[i]` is not necessarily a one-hot vector; we need to
        # find the {row, col} indices where it is non-zero, and use them
        # to create duplicates of {y_pred, y_true, s_pred, rho}.
        row_idx, col_idx = s_true.gt(0).long().nonzero(as_tuple=True)
        y_pred = y_pred[row_idx]             # (N + ?,  )
        y_true = y_true[row_idx]             # (N + ?,  )
        s_pred_k = s_pred[row_idx, col_idx]  # (N + ?,  )
        rho = rho[col_idx]                   # (N + ?,  )

        # - log p[y, S_k = 1 | x] = - log p(y|x) - log P(S_k = 1 | y, x)
        #   Shape: (N,  )
        loss_selected = - torch.log(
            _normal.cdf(
                (s_pred_k + rho * (y_true - y_pred).div(_eps + sigma)) / torch.sqrt(_eps + 1 - rho ** 2)
            ) + _eps
        ) + 0.5 * (
            torch.log(2 * torch.pi * (sigma ** 2)) \
                + F.mse_loss(y_pred, y_true, reduction='none').div(_eps + sigma ** 2)
        )

        return loss_selected.mean()  # float tensor (singleton)

    def _l2_penalty_outcome(self):
        l2_reg = 0.
        for param in self.outcome_model.parameters():
            l2_reg += torch.pow(param, 2)
        return l2_reg


class HeckmanDGRegressionSequential(_HeckmanDGRegressionBase):
    def __init__(self,
                 input_features: int,
                 num_train_domains: int,
                 device: str = 'cpu', ) -> None:
        super(HeckmanDGRegressionSequential, self).__init__(
            input_features=input_features,
            num_train_domains=num_train_domains,
            device=device,
        )

    def compile(self,
                selection_optimizer: str = 'lbfgs',
                selection_lr: float = 0.01,
                selection_weight_decay: float = 0.,
                selection_lbfgs_max_iter: int = 5,
                outcome_optimizer: str = 'lbfgs',
                outcome_lr: float = 0.01,
                outcome_weight_decay: float = 0.,
                outcome_lbfgs_max_iter: int = 5,
                correlation_optimizer: str = 'lbfgs',
                correlation_lr: float = 0.01,
                correlation_weight_decay: float = 0.,
                correlation_lbfgs_max_iter: int = 5,
                ) -> None:
        """..."""

        self.hparams = dict(
            selection_optimizer=selection_optimizer,
            selection_lr=selection_lr,
            selection_weight_decay=selection_weight_decay,
            selection_lbfgs_max_iter=selection_lbfgs_max_iter,
            outcome_optimizer=outcome_optimizer,
            outcome_lr=outcome_lr,
            outcome_weight_decay=outcome_weight_decay,
            outcome_lbfgs_max_iter=outcome_lbfgs_max_iter,
            correlation_optimizer=correlation_optimizer,
            correlation_lr=correlation_lr,
            correlation_weight_decay=correlation_weight_decay,
            correlation_lbfgs_max_iter=correlation_lbfgs_max_iter,
        )
        
        self._init_model()
        self._init_optimizer()
        self._init_cuda()

    def _init_optimizer(self) -> None:
        """Initialize optimizers."""
        self._init_selection_optimizer()
        self._init_outcome_optimizer()
        self._init_correlation_optimizer()

    def _init_selection_optimizer(self) -> None:
        """Initialize `selection` optimizer."""
        cfg = {
            'params': self.selection_model.parameters(),
            'name': self.hparams['selection_optimizer'],
            'lr': self.hparams['selection_lr'],
            'weight_decay': self.hparams.get('selection_weight_decay', 0.),
        }
        if cfg['name'].lower() in ('lbfgs', 'l-bfgs'):
            _ = cfg.pop('weight_decay')  # no weight decay for L-BFGS
            cfg['max_iter'] = self.hparams.get('selection_lbfgs_max_iter', 5)
            cfg['history_size'] = self.hparams.get('selection_lbfgs_history_size', 100)
        self.selection_optimizer = configure_optimizer(**cfg)

    def _init_outcome_optimizer(self) -> None:
        """Initialize `outcome` optimizer."""
        cfg = {
            'params': self.outcome_model.parameters(),
            'name': self.hparams['outcome_optimizer'],
            'lr': self.hparams['outcome_lr'],
            'weight_decay': self.hparams.get('outcome_weight_decay', 0.),
        }
        if cfg['name'].lower() in ('lbfgs', 'l-bfgs'):
            _ = cfg.pop('weight_decay')  # no weight decay for L-BFGS
            cfg['max_iter'] = self.hparams.get('outcome_lbfgs_max_iter', 5)
            cfg['history_size'] = self.hparams.get('outcome_lbfgs_history_size', 100)
        self.outcome_optimizer = configure_optimizer(**cfg)

    def _init_correlation_optimizer(self) -> None:
        """Initialize `correlation` optimizer."""
        cfg = {
            'params': [self._rho, self.sigma],
            'name': self.hparams['correlation_optimizer'],
            'lr': self.hparams['correlation_lr'],
            'weight_decay': self.hparams['correlation_weight_decay'],
        }
        if cfg['name'].lower() in ('lbfgs', 'l-bfgs'):
            _ = cfg.pop('weight_decay')
            cfg['max_iter'] = self.hparams.get('correlation_lbfgs_max_iter', 5)
            cfg['history_size'] = self.hparams.get('correlation_lbfgs_history_size', 100)
        self.correlation_optimizer = configure_optimizer(**cfg)

    def fit(self,
            train_set: torch.utils.data.Dataset,
            validation_set: torch.utils.data.Dataset,
            selection_epochs: int,
            outcome_epochs: int,
            batch_size: typing.Optional[int] = None,
            description: str = 'HeckmanDG Regression (sequential)',
            **kwargs, ) -> typing.Dict[str, torch.Tensor]:
        """Fit HeckmanDG sequentially."""

        # `full batch training` if batch_size = None
        batch_size: int = len(train_set) if batch_size is None else batch_size

        # Data loader
        train_loader = DataLoader(dataset=train_set, batch_size=batch_size,
                                  shuffle=True, drop_last=True)

        validation_loader = DataLoader(validation_set, batch_size=len(validation_set),
                                       shuffle=False, drop_last=False)

        if self.num_train_domains != 2:
            raise NotImplementedError

        # buffers
        results = {
            'selection/train/loss': torch.zeros(selection_epochs, device=self.device),
            'selection/train/accuracy': torch.zeros(selection_epochs, device=self.device),
            'selection/train/f1': torch.zeros(selection_epochs, device=self.device),
            'outcome/train/loss': torch.zeros(outcome_epochs, device=self.device),
            'outcome/validation/loss': torch.zeros(outcome_epochs, device=self.device),
            'correlation/train/loss': torch.zeros(outcome_epochs, device=self.device),
            'rho/1': torch.zeros(outcome_epochs, device=self.device),
            'rho/2': torch.zeros(outcome_epochs, device=self.device),
            'sigma': torch.zeros(outcome_epochs, device=self.device),
        }

        # 1. train (selection)
        _desc: str = f":roller_coaster: {description}, step 1: selection model training... "
        for e in track(range(selection_epochs), total=selection_epochs, description=_desc):
            sel_train_loss = self.train_selection(train_loader)
            results['selection/train/loss'][e] = sel_train_loss
            sel_train_metrics = self.evaluate_selection_model(train_loader)
            results['selection/train/accuracy'][e] = sel_train_metrics['accuracy']
            results['selection/train/f1'][e] = sel_train_metrics['f1']

        # 2. train (outcome, correlation)
        _desc: str = f":star_of_david: {description}, step 2: outcome model / correlation training... "
        for e in track(range(outcome_epochs), total=outcome_epochs, description=_desc):
            out_train_loss = self.train_outcome(train_loader)       # train outcome model
            corr_train_loss = self.train_correlation(train_loader)  # train correlation / sigma
            results['outcome/train/loss'][e] = out_train_loss
            results['correlation/train/loss'][e] = corr_train_loss
            out_validation_loss = self.evaluate(validation_loader)['loss']
            results['outcome/validation/loss'][e] = out_validation_loss
            results['rho/1'][e] = torch.tanh(self._rho[0]).detach().clone()
            results['rho/2'][e] = torch.tanh(self._rho[1]).detach().clone()
            results['sigma'][e] = self._sigma.detach().clone() 

        return results

    def train_selection(self, loader: torch.utils.data.DataLoader, ) -> torch.FloatTensor:
        """..."""
        losses = torch.zeros(len(loader), device=self.device)
        for i, batch in enumerate(loader):
            losses[i] = self.train_selection_batch(batch).detach()

        return losses.mean()

    def train_selection_batch(self, batch: dict) -> torch.FloatTensor:
        """..."""
        assert self._is_valid_batch(batch, exclude=['y', ])  # check
        self.selection_model.train();                        # train mode

        # fetch data
        x = batch['x'].to(self.device)       # (N, P)
        s = batch['domain'].to(self.device)  # (N, K)

        def _selection_closure():
            self.selection_optimizer.zero_grad()
            s_pred_in_probits = self.selection_model(x)
            loss = self.loss_for_training_selection_model(
                s_pred=s_pred_in_probits, s_true=s,
            )
            loss.backward()
            return loss

        loss = self.selection_optimizer.step(_selection_closure)
        
        return loss.detach().clone()

    def train_outcome(self, loader: torch.utils.data.DataLoader,) -> torch.FloatTensor:
        """..."""
        losses = torch.zeros(len(loader), device=self.device)
        for i, batch in enumerate(loader):
            losses[i] = self.train_outcome_batch(batch)

        return losses.mean()

    def train_outcome_batch(self, batch: dict) -> torch.FloatTensor:
        """..."""
        assert self._is_valid_batch(batch)  # check
        self.selection_model.eval()
        self.outcome_model.train()

        # fetch data
        x = batch['x'].to(self.device)       # (N, P)
        y = batch['y'].to(self.device)       # (N,  )
        s = batch['domain'].to(self.device)  # (N, K)

        # these values are not optimized
        with torch.no_grad():
            # predict selection probits
            s_pred_in_probits = self.selection_model(x)
            # get correlation; self._rho is the arctanh(rho)
            rho = torch.tanh(self._rho)
            # get `sigma`
            sigma = self._sigma

        def _outcome_closure():
            # zero out gradients
            self.outcome_optimizer.zero_grad()
            # get outcome predictions
            y_pred = self.outcome_model(x).squeeze(1)
            loss = self.loss_for_training_outcome_model(
                y_pred=y_pred, y_true=y, s_pred=s_pred_in_probits, s_true=s,
                rho=rho, sigma=sigma,
            )
            if isinstance(self.outcome_optimizer, torch.optim.LBFGS):
                loss += self.hparams['outcome_weight_decay'] * self._l2_penalty_outcome()
            loss.backward()
            return loss

        loss = self.outcome_optimizer.step(_outcome_closure)
        
        return loss.detach().clone()

    def train_correlation(self, loader: torch.utils.data.DataLoader) -> torch.FloatTensor:
        """..."""
        losses = torch.zeros(len(loader), device=self.device)
        for i, batch in enumerate(loader):
            losses[i] = self.train_correlation_batch(batch).detach()

        return losses.mean()

    def train_correlation_batch(self, batch: dict) -> torch.FloatTensor:
        """..."""
        assert self._is_valid_batch(batch)
        self.selection_model.eval();
        self.outcome_model.eval();

        # fetch data
        x = batch['x'].to(self.device)       # (N, P)
        y = batch['y'].to(self.device)       # (N,  )
        s = batch['domain'].to(self.device)  # (N, K)

        with torch.no_grad():
            y_pred = self.outcome_model(x).squeeze(1)  # (N,  )
            s_pred = self.selection_model(x)           # (N, K)

        def _corr_closure():
            """
            Note that we need not do model predictions
            inside the closure function, because we are only optimizing
            for `rho` and `sigma`.
            """
            # zero-out gradients
            self.correlation_optimizer.zero_grad()
            # get correlations / sigma
            rho = torch.tanh(self._rho); assert rho.requires_grad;
            sigma = self._sigma; assert sigma.requires_grad;
            # compute loss
            loss = self.loss_for_training_outcome_model(
                y_pred=y_pred, y_true=y,
                s_pred=s_pred, s_true=s,
                rho=rho, sigma=sigma,
            )
            loss.backward()  # backward pass
            return loss

        loss = self.correlation_optimizer.step(_corr_closure)
        return loss.detach().clone()
