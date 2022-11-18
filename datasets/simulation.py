
import typing

import torch
import numpy as np


class DatasetWithSelectionBias(torch.utils.data.Dataset):
    def __init__(self) -> None:
        super(DatasetWithSelectionBias, self).__init__()

    def __getitem__(self, index: int) -> typing.Dict[str, torch.Tensor]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class BivariateExample(DatasetWithSelectionBias):
    def __init__(self,
                 input_covariance_matrix: torch.FloatTensor,
                 outcome_intercept: torch.FloatTensor,
                 outcome_coef: torch.FloatTensor,
                 num_train_domains: int = 2,
                 num_test_domains: int = 1,
                 selection_feature_index: typing.List[int] = [1, ],
                 alpha_prior_mean: typing.Union[float, typing.List[float]] = 5.0,
                 alpha_prior_sigma: typing.Union[float, typing.List[float]] = 3.0,
                 sigma_s: float = 1.0,
                 sigma_y: float = 1.0,
                 rho: float = 0.8,
                 population_size: int = 100000,
                 sampling_rate_per_domain: typing.Union[float, list] = 0.01
                 ) -> None:
        super(BivariateExample, self).__init__()

        self.input_covariance_matrix = input_covariance_matrix
        self.outcome_intercept = outcome_intercept
        self.outcome_coef = outcome_coef

        self.num_train_domains: int = num_train_domains
        self.num_test_domains: int = num_test_domains
        self.num_domains: int = self.num_train_domains + self.num_test_domains

        self.selection_feature_index: list = selection_feature_index
        
        if isinstance(alpha_prior_mean, float):
            self.alpha_prior_mean = torch.tensor([alpha_prior_mean, ])
        elif isinstance(alpha_prior_mean, list):
            if len(self.selection_feature_index) != len(alpha_prior_mean):
                raise ValueError(
                    f"Length of `alpha_prior_mean` ({len(alpha_prior_mean)}) must match that "
                    f"of `selection_feature_index` ({len(self.selection_feature_index)})."
                )
            self.alpha_prior_mean = torch.tensor(alpha_prior_mean)
        else:
            raise NotImplementedError
        
        if isinstance(alpha_prior_sigma, float):
            self.alpha_prior_sigma = torch.tensor([alpha_prior_sigma, ])
        elif isinstance(self.alpha_prior_sigma, list):
            if len(self.selection_feature_index) != len(alpha_prior_sigma):
                raise ValueError(
                    f"Length of `alpha_prior_sigma` ({len(alpha_prior_sigma)}) must match that "
                    f"of `selection_feature_index` ({len(self.selection_feature_index)})."
                )
            self.alpha_prior_sigma = torch.tensor(alpha_prior_sigma)
        else:
            raise NotImplementedError

        self.sigma_s: float = sigma_s
        self.sigma_y: float = sigma_y
        self.rho: float = rho
        self.population_size: int = population_size
        self.sampling_rate_per_domain: float = sampling_rate_per_domain

        self.num_features: int = len(self.input_covariance_matrix)
        if self.num_features != len(self.outcome_coef):
            raise ValueError(
                f"Number of input features ({self.num_features}) "
                f"must match with the size of beta coefficents ({self.outcome_coef.__len__()})."
            )
        self._input_distribution = \
            torch.distributions.MultivariateNormal(
                loc=torch.zeros(self.num_features),
                covariance_matrix=self.input_covariance_matrix
            )

        self._error_distribution = \
            torch.distributions.MultivariateNormal(
                loc=torch.zeros(2),
                covariance_matrix=self.error_covariance_matrix
            )

        # Generate true population
        self._population: dict = self._generate_population()

        # Fix selection models for train & test data
        # selection_coef: shape (num_features, num_domains)
        # selection_intercept: shape (1, num_domains)
        self.selection_coef, self.selection_intercept = self.get_selection_models(
            num_domains=self.num_domains,
            alpha_prior_mean=self.alpha_prior_mean,
            alpha_prior_sigma=self.alpha_prior_sigma,
            selection_feature_index=self.selection_feature_index
        )

        # Sample training data
        self.data = self.sample_data(
            selection_coef=self.selection_coef[:, :self.num_train_domains],
            selection_intercept=self.selection_intercept[:, :self.num_train_domains],
            mode='train',
        )

        # Sample in-distribution test data (TODO; use self.num_test_domains)
        self.test_data = self.sample_data(
            selection_coef=self.selection_coef[:, self.num_train_domains: ],
            selection_intercept=self.selection_intercept[:, self.num_train_domains: ],
            mode='test',
        )

        # Save compliment set, which is not selected by training selection models
        self.compliment_data = self.sample_compliment_data(
            selection_coef=self.selection_coef[:, :self.num_train_domains],
            selection_intercept=self.selection_intercept[:, :self.num_train_domains],
            size=self.data['x'].__len__() * 2,
        )

        # Miscellaneous statistics
        self.unique_train_counts: int = self.data['true_index'].unique().__len__()
        self.train_test_overlap: int = np.intersect1d(
            self.train_data['true_index'].cpu().numpy(),
            self.test_data['true_index'].cpu().numpy(),
        ).shape[0]

    def __getitem__(self, index: int) -> typing.Dict[str, torch.Tensor]:
        return dict(
            x=self.data['x'][index],
            y=self.data['y'][index],
            domain=self.data['domain'][index],
            true_index=self.data['true_index'][index]
        )
    
    def _generate_population(self) -> dict:
        raise NotImplementedError(
            "Subclasses inheriting this class must have this function implemented."
        )

    def get_selection_models(self,
                             num_domains: int,
                             alpha_prior_mean: torch.FloatTensor,
                             alpha_prior_sigma: torch.FloatTensor,
                             selection_feature_index: typing.List[int],
                             ) -> typing.Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Samples `num_domains` number of alpha cofficients and intercepts that match the sampling rate.
        Arguments:
            num_domains: `int`, number of selection models.
            alpha_prior_mean: `torch.FloatTensor` of shape (s, ), where
                s is the number of features associated with selection.
            alpha_prior_sigma: `torch.FloatTensor` of shape (s, ), where
                s is the number of features associated with selection.
            selection_feature_index: `list` of length s, where
                each value indicates the index of the feature associated with selection.
        Returns:
            a tuple of (`selection_coef`, `selection_intercept`), where
                `selection_coef` is a 2D float tensor of shape (P, K), and
                `selection_intercept` is a 2D float tensor of shape (1, K).
                P is the number of input features, and K is the number of domains (=`num_domains`).
        """

        # 1. Generate selection coefficients; shape: (P, K)
        selection_coef = torch.zeros(self.num_features, num_domains)
        for i, feat_idx in enumerate(selection_feature_index):
            if alpha_prior_sigma[i] != 0.:
                # generate coefficient ~ N(mean, scale)
                loc = alpha_prior_mean[i].item() 
                scale = alpha_prior_sigma[i].item()
                selection_coef[feat_idx] = \
                    torch.distributions.Normal(loc=loc, scale=scale).sample((num_domains, ), )    
            else:
                # fixed value = mean (since sigma = 0)
                selection_coef[feat_idx] = torch.full(size=(num_domains, ), fill_value=alpha_prior_mean[i])
        
        # 2. Determine selection intercepts that match the sampling rate; shape: (1, K)
        selection_intercept = self.find_selection_intercept(
            x=self._population['x'],
            selection_coef=selection_coef,
            selection_errors=self._population['errors_s'],
            desired_sample_size=int(self.population_size * self.sampling_rate_per_domain),
            steps=1000
        )

        return selection_coef, selection_intercept  # (P, K), (1, K)

    @staticmethod
    def find_selection_intercept(x: torch.FloatTensor,
                                 selection_coef: torch.FloatTensor,
                                 selection_errors: torch.FloatTensor,
                                 desired_sample_size: int,
                                 steps: int = 1000, ) -> torch.FloatTensor:
        """
        Arguments:
            x: 2D float tensor with shape (N, P).
            selection_coef: 2D float tensor with shape (P, K). 
                P is the number of input features, and K is the number of domains.
            selection_errors: 1D float tensor with shape (N,  ).
            desired_sample_size: int. 
            steps: int. Resolution of search space ranging [-100, 100].
        Returns:
            selection_intercept: 2D float tensor of shape (1, K).
        """

        num_domains: int = selection_coef.size(1)       # (P, K)
        selection_intercept = torch.zeros(num_domains)  # arbitrary buffer
        for k in range(num_domains):
            search_space = torch.linspace(-100, 100, steps=steps)
            # Shape: (steps, N) <- (steps, 1) + (steps, N, P) @ (steps, P, 1) + (steps, N)
            s_star_k = search_space.view(-1, 1) \
                + torch.bmm(
                    x.unsqueeze(0).repeat(steps, 1, 1),  # (steps, N, P)
                    selection_coef[:, k].view(-1, 1).unsqueeze(0).repeat(steps, 1, 1)  # (steps, P, 1)
                ).squeeze(2) \
                + selection_errors.unsqueeze(0).repeat(steps, 1)  # (steps, N)
            selection_counts = s_star_k.gt(0).sum(dim=1, keepdim=False)  # count number of samples g(x) > 0
            for i in range(steps):
                if selection_counts[i] >= desired_sample_size:
                    selection_intercept[k] = search_space[i]
                    break

        return selection_intercept.view(1, -1)  # (1, K)

    def sample_data(self,
                    selection_coef: torch.FloatTensor,
                    selection_intercept: torch.FloatTensor,
                    mode: str = 'train') -> typing.Dict[str, torch.Tensor]:
        """
        From the true population, sample data according to the selection model(s).
        If a sample is chosen at least once, it is included in the data returned.
        Arguments:
            selection_coef: 2D tensor of shape (P, K).
            selection_intercept: 2D tensor of shape (1, K).
            mode: str, if not `train`, domain is always a vector of zeros.
        Returns:
            dictionary with keys {x, y, domain, true_index}. Their corresponding values are tensors.
        """

        if mode == 'train':
            assert selection_coef.shape[1] == self.num_train_domains

        # A. Generate latent selection values using the true selection model(s)
        #    Intercept terms are determined to match the desired sampling rate.
        #    Shape: (N, K) <- (1, K) + (N, P) @ (P, K) + (N, 1)
        s_star = selection_intercept \
            + torch.matmul(self._population['x'], selection_coef) \
            + self._population['errors_s'].unsqueeze(1)  # broadcasted
        assert s_star.ndim == 2

        # B. Get observed selection values (0/1)
        #    Shape: (N, K)
        s_observed: torch.LongTensor = s_star.gt(0).long()
        mask: torch.BoolTensor = torch.any(s_observed.eq(1), dim=1, keepdim=False)
        
        # C. Sample data
        inputs: torch.FloatTensor = self._population['x'][mask]  # (B, P); B <= N
        targets: torch.Tensor = self._population['y'][mask]      # (B,  )
        domain: torch.LongTensor = s_observed[mask]              # (B, K)
        true_index: torch.LongTensor = mask.long().nonzero(as_tuple=True)[0]

        if mode != 'train':
            domain = torch.zeros_like(domain).long()

        return dict(
            x=inputs,
            y=targets,
            domain=domain,
            true_index=true_index,
        )

    def sample_compliment_data(self,
                               selection_coef: torch.FloatTensor,
                               selection_intercept: torch.FloatTensor,
                               size: int = None,
                               ) -> typing.Dict[str, torch.Tensor]:
        """
        Samples the compliment set of selection models, never selected.
        Arguments:
            selection_coef: 2D tensor of shape (P, K).
            selection_intercept: 2D tensor of shape (1, K).
            size: int, sample size.
        """

        assert selection_coef.shape[1] == self.num_train_domains

        s_star = selection_intercept \
            + torch.matmul(self._population['x'], selection_coef) \
            + self._population['errors_s'].unsqueeze(1)
        s_observed = s_star.gt(0).long()  # (N, K)
        
        compliment_mask = s_observed.sum(dim=1).eq(0)  # not in any of the domains
        compliment_idx = compliment_mask.nonzero(as_tuple=True)[0]
        
        if size is not None:
            compliment_idx = torch.randperm(len(compliment_idx))[:size]
        
        return dict(
            x=self._population['x'][compliment_idx],
            y=self._population['y'][compliment_idx],
            domain=torch.zeros(len(compliment_idx), self.num_train_domains).long(),
            true_index=compliment_idx,
        )

    def randomly_sample_data(self, size: int) -> typing.Dict[str, torch.Tensor]:
        """
        Randomly sample data from the true population.
        Arguments:
            size: int, sample size
        """
        
        random_idx = torch.randperm(
            self._population['x'].__len__(),
            device=self._population['x'].device)[:size]
        
        return dict(
            x=self._population['x'][random_idx],
            y=self._population['y'][random_idx],
            domain=torch.zeros(len(random_idx), self.num_train_domains, dtype=torch.long),
            true_index=random_idx,
        )

    def __len__(self):
        return len(self.data['y'])

    @property
    def error_covariance_matrix(self):
        raise NotImplementedError

    @property
    def train_data(self) -> typing.Dict[str, torch.Tensor]:
        return self.data

    def __str__(self):
        """Add function docstring."""
        _repr = f"Population size: {self.population_size:,}\n"
        _repr += f"{self.num_train_domains} training domains, {self.num_test_domains} test domain(s)\n"
        _repr += f"Unique samples in training data: {self.unique_train_counts:,}\n"  # TODO: remove
        
        _repr += f"Outcome model:\n\t{self.outcome_intercept.item():.2f} + "
        _repr += " + ".join([f"{v.item():.2f} * x{i+1}" for i, v in enumerate(self.outcome_coef)])
        if self.__class__.__name__ == 'ClassificationExample':
            _repr += " > 0"

        _repr += "\nSelection models:\n"
        for k in range(self.num_domains):
            if k < self.num_train_domains:
                size: int = self.data['domain'][:, k].eq(1.).sum().item()
            else:
                size: int = len(self.test_data['y'])  # (FIXME)
            _sel_coef = self.selection_coef[:, k]
            _sub_repr = f"\t(Domain {k}, size = {size:,}) {self.selection_intercept.squeeze()[k].item():.2f} + "
            _sub_repr += " + ".join([f"{v.item():.2f} * x{i+1}" for i, v in enumerate(_sel_coef)])
            _sub_repr += "\n"
            _repr += _sub_repr

        return _repr


class RegressionExample(BivariateExample):
    def __init__(self,
                 input_covariance_matrix: torch.FloatTensor,
                 outcome_intercept: torch.FloatTensor = torch.FloatTensor([1.0, ]),
                 outcome_coef: torch.FloatTensor = torch.FloatTensor([1.5, 3.0]),
                 num_train_domains: int = 2,
                 num_test_domains: int = 1,
                 selection_feature_index: typing.List[int] = [1, ],
                 alpha_prior_mean: typing.Union[float, typing.List[float]] = 5.0,
                 alpha_prior_sigma: typing.Union[float, typing.List[float]] = 3.0,
                 sigma_s: float = 1.0,
                 sigma_y: float = 1.0,
                 rho: float = 0.8,
                 population_size: int = 100000,
                 sampling_rate_per_domain: float = 0.01):
        super(RegressionExample, self).__init__(
            input_covariance_matrix=input_covariance_matrix,
            outcome_intercept=outcome_intercept,
            outcome_coef=outcome_coef,
            num_train_domains=num_train_domains,
            num_test_domains=num_test_domains,
            selection_feature_index=selection_feature_index,
            alpha_prior_mean=alpha_prior_mean,
            alpha_prior_sigma=alpha_prior_sigma,
            sigma_s=sigma_s,
            sigma_y=sigma_y,
            rho=rho,
            population_size=population_size,
            sampling_rate_per_domain=sampling_rate_per_domain
        )
            
    def _generate_population(self) -> typing.Dict[str, torch.FloatTensor]:
        """Generate true population (regression)."""

        # 1. Generate input covariates; (N, P)
        x = self._input_distribution.sample((self.population_size, ))
        assert x.ndim == 2

        # 2. Generate errors following a bivariate normal distribution; (N, 2)
        errors = self._error_distribution.sample((self.population_size, ))
        errors_s, errors_y = errors[:, 0], errors[:, 1]
        assert (errors_s.ndim == 1) and (errors_y.ndim == 1)

        # 3. Generate target outcomes using the true outcome model
        #    Shape: (N, ) <- (1, ) + (N, P) @ (P, ) + (N, )
        y = self.outcome_intercept \
            + torch.matmul(x, self.outcome_coef) \
            + errors_y
        assert y.ndim == 1

        return dict(
            x=x,
            y=y,
            errors_s=errors_s,
            errors_y=errors_y,
        )
            
    @property
    def error_covariance_matrix(self) -> torch.FloatTensor:
        return torch.tensor([[self.sigma_s ** 2, self.rho * self.sigma_s * self.sigma_y],
                             [self.rho * self.sigma_s * self.sigma_y, self.sigma_y ** 2]])
