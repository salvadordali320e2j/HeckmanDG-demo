
import os
import typing
import yaml
import argparse
import datetime
import itertools

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rich.console import Console
from torch.utils.data import DataLoader
from torchmetrics.functional import mean_squared_error

from models.baselines.linear_models import LinearRegression
from models.heckman.linear_models import HeckmanDGRegressionSequential

from datasets.simulation import RegressionExample
from datasets.utils import DictionaryDataset
from utils.misc import fix_random_seed


def parse_arguments(description: str = 'Regression example with 2 covariates.'):
    """Parse arguments."""
    parser = argparse.ArgumentParser(description=description)
    
    parser.add_argument('--num_train_domains', type=int,
                        default=2,
                        help='Number of training domains (default: 2)')
    parser.add_argument('--num_test_domains', type=int,
                        default=1,
                        help='Number of testing domains (default: 1)')
    parser.add_argument('--input_features', type=int,
                        default=2,
                        help='Number of input covariates (default: 2)')
    parser.add_argument('--selection_feature_index', nargs='+', type=int,
                        default=[1, ],
                        help='Index of input feature (starting from 0) that participates in the sample selection equations (default: 1)')
    parser.add_argument('--alpha_prior_mean', type=float,
                        default=5.0,
                        help='Mean of alpha coefficients for selection models (default: 5.0)')
    parser.add_argument('--alpha_prior_sigma', type=float,
                        default=3.0,
                        help='Standard deviation of alpha coefficients for selection models (default: 3.0)')
    parser.add_argument('--sigma_s', type=float,
                        default=1.0,
                        help='Standard deviation of selection error terms (default: 1.0)')
    parser.add_argument('--sigma_y', type=float,
                        default=1.0,
                        help='Standard deviation of outcome error term (default: 1.0)')
    parser.add_argument('--rho', type=float,
                        default=0.8,
                        help='Correlation between error terms (default: 0.8)')
    parser.add_argument('--population_size', type=int,
                        default=100000,
                        help='True population size (default: 100k)')
    parser.add_argument('--sampling_rate_per_domain', type=float,
                        default=0.01,
                        help='Sampling rate for each domain (default: 0.01)')

    parser.add_argument('--epochs', type=int,
                        default=200,
                        help='Number of training epochs (default: 200)')
    parser.add_argument('--batch_size', type=int,
                        default=None,
                        help='Training batch size (default: None)')
    parser.add_argument('--device', type=str,
                        default='cpu',
                        help='Device configuration (default: cpu)')

    parser.add_argument('--num_experiments', type=int,
                        default=30,
                        help='Number of repeated trials (default: 30)')
    parser.add_argument('--write_dir', type=str,
                        default='./checkpoints/simulations/',
                        help='Base directory to write result files.')

    # unique hash used to create a checkpoint directory.
    args = parser.parse_args()
    setattr(args, 'hash', datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S"))

    return args


def write_dict_to_yaml(d: dict, yaml_file: str):
    if not yaml_file.endswith('.yaml'):
        yaml_file += '.yaml'
    with open(yaml_file, 'w') as outfile:
        yaml.dump(d, outfile, default_flow_style=False)


def create_data(args: argparse.Namespace) -> typing.Tuple[RegressionExample, typing.Dict[str, DictionaryDataset]]:
    
    # 0. create example
    example = RegressionExample(
        input_covariance_matrix=torch.eye(args.input_features),
        outcome_intercept=torch.tensor([1.0, ]),
        outcome_coef=torch.tensor([1.5, 3.0, ]),
        num_train_domains=args.num_train_domains,
        num_test_domains=args.num_test_domains,
        alpha_prior_mean=args.alpha_prior_mean,
        alpha_prior_sigma=args.alpha_prior_sigma,
        selection_feature_index=args.selection_feature_index,
        sigma_s=args.sigma_s,
        sigma_y=args.sigma_y,
        rho=args.rho,
        population_size=args.population_size,
        sampling_rate_per_domain=args.sampling_rate_per_domain,
    )

    # 1. train set
    train_set = DictionaryDataset(example.train_data)

    # 2. in-distribution test set is created upon dataset initialization by default
    id_test_set = DictionaryDataset(example.test_data)

    # 3-1. random train set
    random_train_size = int(args.population_size * args.sampling_rate_per_domain)
    random_train_set = DictionaryDataset(example.randomly_sample_data(size=random_train_size))

    # 3-1. random validation set
    random_validation_size = random_train_size
    random_validation_set = DictionaryDataset(example.randomly_sample_data(size=random_validation_size))

    # 3-3. random test set
    random_test_size = random_train_size
    random_test_set = DictionaryDataset(example.randomly_sample_data(size=random_test_size))

    # 4. out-of-distribution test set
    #   use a different prior for alpha coefficients:
    #      if `alpha_prior_mean` is zero, a mean shift is occured (+5 by default).
    #      if `alpha_prior_mean` is non-zero, the sign is reversed (e.g., 5 -> -5, vice versa).
    #   currently only supports cases where only one covariate is associated with sample selection.
    
    if len(example.selection_feature_index) > 1:
        raise NotImplementedError
    
    ood_alpha_prior_mean = torch.zeros_like(example.alpha_prior_mean)
    
    for i, _a_mean in enumerate(example.alpha_prior_mean):
        if _a_mean == 0.:
            ood_alpha_prior_mean[i] = 5.0  # mean shift: +5
        else:
            ood_alpha_prior_mean[i] = torch.neg(_a_mean)  # flip signs
    
    ood_selection_coef, ood_selection_intercept = example.get_selection_models(
        num_domains=1,
        alpha_prior_mean=ood_alpha_prior_mean,
        alpha_prior_sigma=example.alpha_prior_sigma,
        selection_feature_index=example.selection_feature_index,
    )
    ood_test_set = DictionaryDataset(
        example.sample_data(
            selection_coef=ood_selection_coef,
            selection_intercept=ood_selection_intercept,
            mode='ood'  # values other than `train` will do
        )
    )

    # 5. compliment data with respect to the training domains
    compliment_set = DictionaryDataset(example.compliment_data)

    data = dict(
        train=train_set,
        id_test=id_test_set,
        random_train=random_train_set,
        random_validation=random_validation_set,
        random_test=random_test_set,
        ood_test=ood_test_set,
        compliment=compliment_set,
    )

    return example, data


def hyperparameter_search(trainer: object,
                          hparam_grid: typing.Dict[str, list],
                          fit_params: dict,
                          metric: str = 'validation/loss',
                          lower_is_better: bool = True,
                          **kwargs, ) -> None:

    # create a grid of hyperparameters
    hparam_names = [k for k, _ in hparam_grid.items()]   # e.g., ['optimizer', 'lr']
    hparam_values = [v for _, v in hparam_grid.items()]  # e.g., [['lbfgs', 'adam'], [1e-2, 1e-1]]
    hparam_sweep = [_cartes for _cartes in itertools.product(*hparam_values)]
    print(f"Total number of hyperparameter sweeps: {len(hparam_sweep):,}")

    trainers = list()
    search_results = list()
    for i, sweep in enumerate(hparam_sweep):
        
        # print (for sanity check)
        sweep_str: str = " | ".join(
            [f"{name} = {val}" for name, val in zip(hparam_names, sweep)]
        )
        print(f"Sweep {i}, {sweep_str} ")
        
        # make new hyperparameter setting
        _hparams = {k: v for k, v in zip(hparam_names, sweep)}

        # compile model with hyperparameters
        trainer.compile(**_hparams)

        # fit model
        fit_results = trainer.fit(**fit_params)
        
        # save results
        search_results.append(fit_results)
        trainers.append(trainer)
    
    # re-format results
    metric_values = torch.zeros(len(hparam_sweep), device=trainer.device)
    for i, result in enumerate(search_results):
        # get last value of metric
        metric_values[i] = result[metric][-1]
    
    # find the best index
    if lower_is_better:
        best_index: int = torch.argmin(metric_values)
    else:
        best_index: int = torch.argmax(metric_values)
    
    # return best trainer object & its fit history
    return trainers[best_index], search_results[best_index]


def test(trainer: object,
         data: typing.Dict[str, torch.utils.data.Dataset], ) -> typing.Dict[str, float]:

    # 0. buffer
    test_results = dict()

    # 1-2. Train
    train_loader = DataLoader(dataset=data['train'], batch_size=len(data['train']))
    for i, batch in enumerate(train_loader):
        y_pred = trainer.predict(batch['x'].to(trainer.device))
        y_true = batch['y'].to(trainer.device)
        test_results['train/mse']: float = mean_squared_error(y_pred, y_true).item()

    # 1-2. ID-test
    id_test_loader = DataLoader(dataset=data['id_test'], batch_size=len(data['id_test']))
    for i, batch in enumerate(id_test_loader):
        y_pred = trainer.predict(batch['x'].to(trainer.device))
        y_true = batch['y'].to(trainer.device)
        test_results['id_test/mse']: float = mean_squared_error(y_pred, y_true).item()

    # 2. OOD-test
    ood_test_loader = DataLoader(dataset=data['ood_test'], batch_size=len(data['ood_test']))
    for i, batch in enumerate(ood_test_loader):
        y_pred = trainer.predict(batch['x'].to(trainer.device))
        y_true = batch['y'].to(trainer.device)
        test_results['ood_test/mse']: float = mean_squared_error(y_pred, y_true).item()

    # 3. Random-test
    random_test_loader = DataLoader(dataset=data['random_test'], batch_size=len(data['random_test']))
    for i, batch in enumerate(random_test_loader):
        y_pred = trainer.predict(batch['x'].to(trainer.device))
        y_true = batch['y'].to(trainer.device)
        test_results['random_test/mse']: float = mean_squared_error(y_pred, y_true).item()

    return test_results


def main():

    #######################################################################

    # parse arguments
    args = parse_arguments()

    # rich console for printing
    console = Console()
    console.print(vars(args))

    # create output directory and file
    write_dir = os.path.join(args.write_dir, str(args.hash))
    os.makedirs(write_dir, exist_ok=False)

    # save configurations in a *.yaml file
    with open(os.path.join(write_dir, 'configuration.yaml'), 'w') as outfile:
        yaml.dump(vars(args), outfile, default_flow_style=False)

    #######################################################################

    console.print(f"Starting {args.num_experiments:,} experiments...")

    for t in range(args.num_experiments):

        experiment_results = list()

        # set random seed
        fix_random_seed(t)
        console.print(f"Experiment: {t}")

        # create data
        example, data = create_data(args)
        console.print(example)  # prints `__repr__`

        # save data statistics
        exp_dir = os.path.join(write_dir, f'exp_{t}')
        os.makedirs(exp_dir, exist_ok=False)
        with open(os.path.join(exp_dir, 'data.txt'), 'w') as file:
            print(example, file=file)

        #######################################################################

        # 1. Linear regression (Oracle)

        name: str = "ERM(Oracle)"
        console.print(f"1. {name}, trial={t}")
        trainer = LinearRegression(
            input_features=args.input_features, device=args.device,
        )
        
        # these values are passed to `trainer.compile(...)`
        hparam_grid = dict(
            optimizer=['lbfgs', ],
            lr=[1e-2, 1e-1, ],
            lbfgs_max_iter=[1, ]
        )

        # these values are passed to `trainer.fit(...)`
        fit_params = dict(
            train_set=data['random_train'],
            validation_set=data['random_train'],
            epochs=args.epochs,
            batch_size=args.batch_size,
        )

        # fit
        trainer, fit_results = hyperparameter_search(
            trainer=trainer, hparam_grid=hparam_grid, fit_params=fit_params,
            metric='validation/loss', lower_is_better=True,
        )

        # plot loss trajectory
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(fit_results['train/loss'].cpu().numpy(), label='Train')
        ax.plot(fit_results['validation/loss'].cpu().numpy(), label='Validation')
        ax.set_title(f'Model = {name}', fontweight='bold', fontsize=20)
        ax.set_xlabel('Iteration', fontsize=18)
        ax.set_ylabel('Loss', fontsize=18)
        ax.legend(loc='upper right', fontsize=15)
        ax.grid(True, alpha=0.2)
        fig.savefig(os.path.join(exp_dir, f"{name.lower()}.pdf"), bbox_inches='tight')

        # test
        write_dict_to_yaml(trainer.hparams, yaml_file=os.path.join(exp_dir, f"hparams_{name.lower()}.yaml"))
        console.print(f"Testing with hyperparameters: ", trainer.hparams)
        test_results = test(trainer=trainer, data=data)
        test_results['name'] = name
        test_results['beta_0'] = trainer.intercept_.item()
        test_results['beta_1'] = trainer.coef_[0].item()
        test_results['beta_2'] = trainer.coef_[1].item()
        experiment_results.append(test_results)
        console.print('Done!\n')

        #######################################################################

        # 2. Linear regression (ERM)
        name: str = "ERM"
        console.print(f"2. {name}, trial={t}")
        trainer = LinearRegression(
            input_features=args.input_features, device=args.device
        )

        # these values are passed to `trainer.compile(...)`
        hparam_grid = dict(
            optimizer=['lbfgs', 'adam', ],
            lr=[1e-2, 1e-1, ],
            lbfgs_max_iter=[1, ],
        )

        # these values are passed to `trainer.fit(...)`
        fit_params = dict(
            train_set=data['train'],
            validation_set=data['train'],
            epochs=args.epochs,
            batch_size=args.batch_size,
        )

        # fit
        trainer, fit_results = hyperparameter_search(
            trainer=trainer, hparam_grid=hparam_grid, fit_params=fit_params,
            metric='validation/loss', lower_is_better=True,
        )

        # plot loss trajectory
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(fit_results['train/loss'].cpu().numpy(), label='Train')
        ax.plot(fit_results['validation/loss'].cpu().numpy(), label='Validation')
        ax.set_title(f'Model = {name}', fontweight='bold', fontsize=20)
        ax.set_xlabel('Iteration', fontsize=18)
        ax.set_ylabel('Loss', fontsize=18)
        ax.legend(loc='upper right', fontsize=15)
        ax.grid(True, alpha=0.2)
        fig.savefig(os.path.join(exp_dir, f"{name.lower()}.pdf"), bbox_inches='tight')

        # test
        write_dict_to_yaml(trainer.hparams, yaml_file=os.path.join(exp_dir, f"hparams_{name.lower()}.yaml"))
        console.print(f"Testing with hyperparameters: ", trainer.hparams)
        test_results = test(trainer=trainer, data=data)
        test_results['name'] = name
        test_results['beta_0'] = trainer.intercept_.item()
        test_results['beta_1'] = trainer.coef_[0].item()
        test_results['beta_2'] = trainer.coef_[1].item()
        experiment_results.append(test_results)
        console.print('Done!\n')

        #######################################################################

        # 3. HeckmanDG (sequential, a.k.a two-step optimization)
        name: str = "HeckmanDG-Sequential"
        console.print(f"3. {name}, trial={t}")
        trainer = HeckmanDGRegressionSequential(input_features=args.input_features,
                                                num_train_domains=args.num_train_domains,
                                                device=args.device
        )

        # these values are passed to `trainer.compile(...)`
        hparam_grid = dict(
            selection_optimizer=['adam', ],
            selection_lr=[1e-2, ],
            selection_lbfgs_max_iter=[5, ],
            outcome_optimizer=['adam', ],
            outcome_lr=[1e-2, 1e-1],
            outcome_weight_decay=[1e-5, ],
            outcome_lbfgs_max_iter=[5, ],
            correlation_optimizer=['adam', ],
            correlation_lr=[1e-2, 1e-1, ],
            correlation_lbfgs_max_iter=[5, ],
        )

        # these values are passed to `trainer.fit(...)`
        fit_params = dict(
            train_set=data['train'],
            validation_set=data['train'],
            selection_epochs=int(args.epochs * 1.5),
            outcome_epochs=int(args.epochs * 1.5),
            batch_size=args.batch_size,
        )

        # fit
        trainer, fit_results = hyperparameter_search(
            trainer=trainer, hparam_grid=hparam_grid, fit_params=fit_params,
            metric='outcome/validation/loss', lower_is_better=True,
        )

        # plot loss trajectory
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes = axes.flatten()
        
        axes[0].plot(fit_results['outcome/train/loss'].cpu().numpy(), label='Train')
        axes[0].plot(fit_results['outcome/validation/loss'].cpu().numpy(), label='Validation')
        axes[0].set_xlabel('Iteration', fontsize=18)
        axes[0].set_ylabel('Loss', fontsize=18)
        axes[0].legend(loc='best', fontsize=15)
        axes[0].grid(True, alpha=.2);
        
        axes[1].plot(fit_results['rho/1'].cpu().numpy(), label=r"$\hat{\rho_1}$")
        axes[1].plot(fit_results['rho/2'].cpu().numpy(), label=r"$\hat{\rho_2}$")
        axes[1].set_xlabel('Iteration', fontsize=18)
        axes[1].set_ylabel(r"$\rho$", fontsize=18)
        axes[1].legend(loc='best', fontsize=15)
        axes[1].grid(True, alpha=.2);
        
        axes[2].plot(fit_results['sigma'].cpu().numpy(), label=r"$\hat{\sigma}$")
        axes[2].set_xlabel('Iteration', fontsize=18)
        axes[2].set_ylabel(r"$\sigma$", fontsize=18)
        axes[2].legend(loc='best', fontsize=15)
        axes[2].grid(True, alpha=.2);
        
        axes[3].plot(fit_results['selection/train/accuracy'].cpu().numpy(), label='accuracy')
        axes[3].plot(fit_results['selection/train/f1'].cpu().numpy(), label='f1')
        axes[3].set_xlabel('Iteration', fontsize=18)
        axes[3].set_ylabel('Selection model performance', fontsize=18)
        axes[3].legend(loc='best', fontsize=15)
        axes[3].grid(True, alpha=.2);
        
        fig.suptitle(f'Model = {name}', fontweight='bold', fontsize=20)
        fig.savefig(os.path.join(exp_dir, f"{name.lower()}.pdf"), bbox_inches='tight')

        # test
        write_dict_to_yaml(trainer.hparams, yaml_file=os.path.join(exp_dir, f"hparams_{name.lower}.yaml"))
        console.print(f"Testing with hyperparameters: ", trainer.hparams)
        test_results = test(trainer=trainer, data=data)
        test_results['name'] = name
        test_results['beta_0'] = trainer.intercept_.item()
        test_results['beta_1'] = trainer.coef_[0].item()
        test_results['beta_2'] = trainer.coef_[1].item()
        experiment_results.append(test_results)
        console.print('Done!\n')

        #######################################################################

        # Write results to pandas dataframe and csv file.
        result_file: str = os.path.join(exp_dir, 'results.csv')
        console.print(f"Saving results to {result_file}")
        df = pd.DataFrame.from_dict(experiment_results, orient='columns')
        df.to_csv(result_file, index=False, header=True)


if __name__ == '__main__':
    try:
        with torch.autograd.set_detect_anomaly(False):
            _ = main()
    except KeyboardInterrupt:
        pass
