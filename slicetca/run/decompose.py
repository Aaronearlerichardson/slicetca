from slicetca.core import SliceTCA, TCA
from slicetca.core.helper_functions import poisson_log_likelihood

import torch
from typing import Union, Sequence
import numpy as np
import scipy
from functools import partial
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from ieeg.calc.fast import mixup


def decompose(data: Union[torch.Tensor, np.array],
              number_components: Union[Sequence[int], int],
              positive: bool = False,
              initialization: str = 'uniform',
              learning_rate: float = 5*10**-3,
              batch_dim: int = None,
              max_iter: int = 10000,
              min_std: float = 10**-5,
              iter_std: int = 100,
              mask: torch.Tensor = None,
              progress_bar: bool = True,
              seed: int = 7,
              weight_decay: float = None,
              batch_prop_decay: int = 1,
              batch_prop = 0.2,
              init_bias: float = 0.,
              loss_function: callable = None,
              verbose: int = 0) -> (list, Union[SliceTCA, TCA]):
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

    torch.manual_seed(seed)

    if isinstance(data, np.ndarray): data = torch.tensor(
        data, device='cuda' if torch.cuda.is_available() else 'cpu')

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

    if isinstance(number_components, int): decomposition = TCA
    elif len(number_components) == 1: decomposition = TCA
    else: decomposition = SliceTCA

    model = decomposition(dimensions, number_components, positive,
                          initialization, dtype=data.dtype, lr=learning_rate,
                          weight_decay=weight_decay, loss=loss_function,
                          init_bias=init_bias)
    # model = torch.compile(model)

    if verbose==0:
        profiler = None
        detect_anomaly = False
    elif verbose==1:
        profiler = "simple"
        detect_anomaly = False
    elif verbose==2:
        profiler = "advanced"
        detect_anomaly = False
    elif verbose==3:
        profiler = None
        detect_anomaly = True
    else:
        raise ValueError("verbose must be 0, 1, 2, or 3")

    early_stop_callback = EarlyStopping(monitor="train_loss",
                                        min_delta=min_std,
                                        patience=iter_std,
                                        verbose=False,
                                        mode="min",
                                        check_on_train_epoch_end=True)
    cb = [early_stop_callback]

    batch_num = data.shape[batch_dim] if batch_dim is not None else 1
    trainer = pl.Trainer(max_epochs=max_iter, min_epochs=0,
                         limit_train_batches=batch_num,
                         enable_progress_bar=progress_bar,
                         enable_model_summary=detect_anomaly,
                         callbacks=cb, profiler=profiler,
                         detect_anomaly=detect_anomaly)

    if mask is None:
        mask = torch.ones_like(data, dtype=torch.bool)
    data[~mask] = 0
    for i in range(batch_prop_decay):
        trainer.fit(model, _feed(data, mask, batch_dim, batch_prop**i))

    return model.get_components(numpy=True), model


def _feed(data, mask, batch_dim=None, batch_prop = 1.0):
    assert 0 < batch_prop <= 1.0, "batch_prop must be in (0, 1]"
    assert data.shape == mask.shape, f"Data and mask must have the same shape, got {data.shape} and {mask.shape}"
    rand_gen = torch.distributions.uniform.Uniform(0, 1)
    while True:
        if batch_prop < 1.0:
            batch = rand_gen.sample(mask.shape) < batch_prop
            mask_out = mask & batch
        else:
            mask_out = mask

        if batch_dim is None:
            yield data, mask_out
        else:
            for i in range(data.shape[batch_dim]):
                idx = [slice(None) if j != batch_dim else i
                       for j in range(data.ndim)]
                if mask_out[idx].any():
                    yield data[idx], mask_out[idx]
