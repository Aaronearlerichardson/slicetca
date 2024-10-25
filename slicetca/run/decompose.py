from slicetca.core import SliceTCA, TCA
from slicetca.core.helper_functions import poisson_log_likelihood
from slicetca.run.data import Data
from slicetca.invariance import invariance

import torch
from typing import Union, Sequence
import numpy as np
import scipy
from functools import partial
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor


def decompose(data: Union[torch.Tensor, np.array],
              number_components: Union[Sequence[int], int],
              positive: bool = False,
              initialization: str = 'uniform',
              learning_rate: float = 5 * 10 ** -3,
              batch_dim: int = None,
              max_iter: int = 10000,
              min_std: float = None,
              iter_std: int = 100,
              mask: torch.Tensor = None,
              progress_bar: bool = True,
              seed: int = 7,
              weight_decay: float = None,
              batch_prop_decay: int = 1,
              batch_prop: float = 0.2,
              init_bias: float = 0.,
              loss_function: callable = None,
              verbose: int = 0,
              compile: bool = False) -> (list, Union[SliceTCA, TCA]):
    """
    High-level function to decompose a data tensor into a SliceTCA or TCA decomposition.

    :param data: Torch tensor.
    :param number_components: If list or tuple number of sliceTCA components, else number of TCA components.
    :param positive: Whether to use a positive decomposition. Defaults the initialization to 'uniform-positive'.
    :param initialization: Components initialization 'uniform'~U(-1,1), 'uniform-positive'~U(0,1), 'normal'~N(0,1).
    :param learning_rate: Learning rate of the optimizer.
    :param batch_prop: Proportion of entries used to compute the gradient at every training iteration.
    :param max_iter: Maximum training iterations.
    :param min_std: Minimum std of the loss under which to return.
    :param iter_std: Number of iterations over which this std is computed.
    :param mask: Entries which are not used to compute the gradient at any training iteration.
    :param verbose: Whether to print the loss at every step.
    :param progress_bar: Whether to have a tqdm progress bar.
    :param seed: Torch seed.
    :param weight_decay: Decay of the parameters. If None defaults to Adam, else AdamW.
    :param batch_prop_decay: Exponential decay steps of the proportion of entries not used to compute the gradient.
    :return: components: A list (over component types) of lists (over factors) of rank x component_shape tensors.
    :return: model: A SliceTCA or TCA model. It can be used to access the losses over training and much more.
    """

    if seed is not None:
        pl.seed_everything(seed, workers=True)

    if isinstance(data, np.ndarray): data = torch.tensor(data)
    elif not isinstance(data, torch.Tensor):
        raise ValueError("data must be a torch.Tensor or a numpy.ndarray")

    if loss_function is None:
        if data.dtype != torch.long:
            loss_function = torch.nn.MSELoss(reduction='sum')
        else:
            spikes_factorial = torch.tensor(scipy.special.factorial(
                data.numpy(force=True)), device=data.device)
            loss_function = partial(poisson_log_likelihood,
                                    spikes_factorial=spikes_factorial)

    dimensions = list(data.shape)

    if batch_dim is not None:
        dimensions.pop(batch_dim)

    if isinstance(number_components, int):
        decomposition = TCA
    elif len(number_components) == 1:
        decomposition = TCA
    else:
        decomposition = SliceTCA

    if min_std is not None:
        min_std *= 2
        iter_std //= 2
    model = decomposition(dimensions, number_components, positive,
                          initialization, dtype=data.dtype, lr=learning_rate,
                          weight_decay=weight_decay, loss=loss_function,
                          init_bias=init_bias, threshold=min_std,
                          patience=iter_std)
    if compile:
        model = torch.compile(model)
    if verbose == 0:
        profiler = None
        detect_anomaly = False
    elif verbose == 1:
        profiler = "simple"
        detect_anomaly = False
    elif verbose == 2:
        profiler = "advanced"
        detect_anomaly = False
    elif verbose == 3:
        profiler = None
        detect_anomaly = True
    else:
        raise ValueError("verbose must be 0, 1, 2, or 3")

    if min_std is not None:
        early_stop_callback = EarlyStopping(monitor="val_loss", verbose=False)
        learning_rate_monitor = LearningRateMonitor(logging_interval='epoch', )
        cb = [early_stop_callback, learning_rate_monitor]
    else:
        early_stop_callback = EarlyStopping(monitor="val_loss", verbose=False, patience=iter_std)
        learning_rate_monitor = LearningRateMonitor(logging_interval='epoch', )
        cb = [early_stop_callback, learning_rate_monitor]

    batch_num = data.shape[batch_dim] if batch_dim is not None else 1
    inputs = Data(data, mask, n_folds=5, prop=batch_prop, test=False)
    inputs.setup()


    for i in range(batch_prop_decay):
        # model.to('cuda')
        # invariance(model, L2='orthogonality', L3=None, max_iter=1000, iter_std=10)
        trainer = pl.Trainer(max_epochs=max_iter, min_epochs=100,
                             accelerator='cuda' if torch.cuda.is_available() else 'cpu',
                             # strategy='ddp' if torch.cuda.is_available() else None,
                             limit_train_batches=batch_num,
                             limit_val_batches=batch_num,
                             enable_progress_bar=progress_bar,
                             enable_model_summary=detect_anomaly,
                             enable_checkpointing=True,
                             callbacks=cb, profiler=profiler,
                             detect_anomaly=detect_anomaly,
                             # precision=64 if data.dtype == torch.float64 else 32,
                             deterministic=True if seed is not None else False)
        inputs.prop = batch_prop ** i
        model.to('cuda')
        # model.training = True
        # trainer.training = True
        # trainer.validating = True
        trainer.fit(model, datamodule=inputs)

    model.to('cpu')

    # trainer.fit(model, _feed(data, mask, batch_dim, batch_prop))

    return model.get_components(numpy=True), model



