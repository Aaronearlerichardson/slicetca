from slicetca.run.decompose import decompose

import torch.multiprocessing as mp
from functools import partial
from concurrent.futures import ProcessPoolExecutor as Pool
from tqdm import tqdm
import torch
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

from typing import Sequence, Union


# To be fixed: high memory usage when using GPU.

def grid_search(data: Union[torch.Tensor], # Only works with torch.Tensor atm
                max_ranks: Sequence[int],
                mask_train: torch.Tensor = None,
                mask_test: torch.Tensor = None,
                min_ranks: Sequence[int] = None,
                sample_size: int = 1,
                processes_sample: int = 1,
                processes_grid: int = 1,
                seed: int = 7,
                verbose: int = 0,
                checkpoint: str = None,
                **kwargs):
    """
    Performs a gridsearch over different number of components (ranks) to see which has the lowest cross-validated loss.

    :param data: Data tensor to decompose.
    :param max_ranks: Maximum number of components of each type.
    :param mask_train: Mask representing over which entries to compute the backpropagated loss. None is full tensor.
    :param mask_test: Mask representing over which entries to compute the loss for validation. None is full tensor.
    :param min_ranks: Minimum number of components of each type.
    :param sample_size: Number of seeds to use for a given number of components.
    :param processes_sample: Number of processes (threads) to use for a given number of components across seeds.
    :param processes_grid: Number of processes (threads) to use over different number of components.
    :param seed: Numpy seed.
    :param kwargs: Same kwargs as decompose.
    :return: A (max_rank_1-min_rank_1, max_rank_2-min_rank_2, ..., sample_size) ndarray of losses masked entries.
    """

    np.random.seed(seed)

    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    if min_ranks is None: min_ranks = [0 for i in max_ranks]
    max_ranks = [i+1 for i in max_ranks]
    rank_span = [max_ranks[i]-min_ranks[i] for i in range(len(max_ranks))]

    grid = get_grid_sample(min_ranks, max_ranks)
    grid = np.concatenate([grid, np.random.randint(10**2,10**6, grid.shape[0])[:,np.newaxis]], axis=-1)

    print('Grid shape:', str(rank_span),
          '- Samples:', sample_size,
          '- Grid entries:', torch.tensor(grid).size()[0],
          '- Number of models to fit:', torch.tensor(grid).size()[0]*sample_size)

    dec = partial(decompose_mp_sample, data=data, mask_train=mask_train, mask_test=mask_test, sample_size=sample_size,
                  processes_sample=processes_sample, verbose=verbose, checkpoint=checkpoint, **kwargs)
    out_grid = []
    if processes_grid == 1:
        for i in tqdm(range(torch.tensor(grid).size()[0]), desc='Number of components (completed): - '):
            out_grid.append(dec(grid[i]))
    else:
        with Pool(max_workers=processes_grid) as pool:
            iterator = tqdm(pool.map(dec, grid), total=torch.tensor(grid).size()[0])
            iterator.set_description('Number of components (completed): - ', refresh=True)
            for i, p in enumerate(iterator):
                out_grid.append(p)
                iterator.set_description('Number of components (completed): '+str(np.unravel_index(i, tuple(max_ranks))) + ' ', refresh=True)

    out_grid = np.array(out_grid, dtype=np.float64)

    loss_grid = out_grid[:,0]
    seed_grid = out_grid[:,1].astype(int)

    loss_grid = loss_grid.reshape(rank_span+[sample_size])
    seed_grid = seed_grid.reshape(rank_span+[sample_size])

    return loss_grid, seed_grid


def decompose_mp_sample(number_components_seed, data, mask_train, mask_test,
                        sample_size, processes_sample, verbose, checkpoint, **kwargs):

    number_components = number_components_seed[:-1]
    seed = number_components_seed[-1]

    np.random.seed(seed)

    # Checkpoint logic
    checkpoint_data = {}
    if checkpoint is not None and isinstance(checkpoint, str):
        if os.path.exists(checkpoint):
            with open(checkpoint, 'rb') as f:
                try:
                    checkpoint_data = pickle.load(f)
                except Exception:
                    checkpoint_data = {}

    dec = partial(decompose_mp,
                  data=data.clone(),
                  mask_train=(mask_train.clone() if mask_train is not None else None),
                  mask_test=(mask_test.clone() if mask_test is not None else None),
                  verbose=verbose,
                  **kwargs)

    sample = number_components[np.newaxis].repeat(sample_size, 0)
    seeds = np.random.randint(10**2,10**6, sample_size)

    sample = np.concatenate([sample, seeds[:,np.newaxis]], axis=-1)
    if processes_sample == 1:
        out = []
        for s, seed_val in zip(sample, seeds):
            key = tuple(s)
            if checkpoint is not None and isinstance(checkpoint, str) and key in checkpoint_data:
                loss = checkpoint_data[key]
            else:
                loss = dec(s)
                if checkpoint is not None and isinstance(checkpoint, str):
                    checkpoint_data[key] = loss
                    with open(checkpoint, 'wb') as f:
                        pickle.dump(checkpoint_data, f)
                    # Plotting after checkpoint update
                    try:
                        # Prepare data for plotting
                        keys = np.array(list(checkpoint_data.keys()))
                        losses = np.array(list(checkpoint_data.values()))
                        if keys.ndim == 2 and keys.shape[1] > 0:
                            # Only plot the first rank dimension for x-axis if possible
                            x = [k[0] for k in keys]
                            plt.figure()
                            plt.scatter(x, losses, c='k')
                            plt.xlabel('First rank dimension')
                            plt.ylabel('Loss')
                            plt.title('Checkpointed Losses')
                            plot_path = checkpoint + '.png'
                            plt.savefig(plot_path)
                            plt.close()
                    except Exception as e:
                        print(e)
            out.append(loss)
        loss = np.array(out)
    else:
        # For multiprocessing, checkpointing is not thread-safe; skip checkpointing in this mode
        loss = np.array(list(map(dec, sample)))

    return loss, seeds


def decompose_mp(number_components_seed, data, mask_train, mask_test, verbose,
                 *args, **kwargs):

    number_components, seed = number_components_seed[:-1], number_components_seed[-1]
    loss_function = kwargs.pop('loss_function',
                               torch.nn.MSELoss(reduction='sum'))

    if (number_components == np.zeros_like(number_components)).all():
        data_hat = 0
    else:
        progress_bar = False if verbose == 0 else True
        _, model = decompose(data, number_components, mask=mask_train,
                             verbose=verbose, progress_bar=progress_bar, *args,
                             seed=seed,loss_function=loss_function,
                             testing=True, **kwargs)
        data_hat = model.construct()
        if data_hat.device != data.device:
            data_hat = data_hat.to(data.device)

    if mask_test is not None:
        loss = model._loss_calc(data, data_hat, mask_test)
    elif 'batch_dim' in kwargs and kwargs['batch_dim'] is not None:
        loss = model.trainer.test(datamodule=model.trainer.datamodule,
                                  ckpt_path='last')[0]['test_loss']
    else:
        loss = model.losses[-1]

    if torch.is_tensor(loss): loss = loss.item()

    return loss


def get_grid_sample(min_dims, max_dims):

    grid = np.meshgrid(*[np.array([i for i in range(min_dims[j],max_dims[j])]) for j in range(len(max_dims))],
                       indexing='ij')

    grid = np.stack(grid)

    return grid.reshape(grid.shape[0], -1).T
