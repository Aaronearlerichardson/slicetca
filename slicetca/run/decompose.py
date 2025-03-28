from lightning.pytorch.callbacks.progress.tqdm_progress import Tqdm

from slicetca.core import SliceTCA, TCA
from slicetca.core.helper_functions import poisson_log_likelihood
from slicetca.run.data import BatchedData, MaskedData
import slicetca.run.utils
from typing import Any

import torch
from typing import Union, Sequence
import numpy as np
import scipy
from functools import partial
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, TQDMProgressBar


def decompose(data: Union[torch.Tensor, np.array],
              number_components: Union[Sequence[int], int],
              positive: bool = False,
              initialization: str = 'uniform',
              learning_rate: float = 5 * 10 ** -3,
              batch_dim: int = None,
              shuffle_dim: int | tuple[int] = 0,
              max_iter: int = 10000,
              min_iter: int = 10,
              min_std: float = None,
              iter_std: int = 100,
              mask: torch.Tensor = None,
              progress_bar: bool = True,
              seed: int = 7,
              weight_decay: float = None,
              batch_prop_decay: int = 1,
              batch_prop: float = 1.,
              init_bias: float = 0.,
              loss_function: callable = None,
              device: str = None,
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
        loss_function = default_loss(data, device)

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

    device = handle_device(device, data, mask, model, compile)
    if batch_dim is None:
        batch_num = 1
        inputs = MaskedData(data, mask, 5, batch_prop, shuffle_dim, device, False)
    else:
        batch_num = data.shape[batch_dim] if batch_dim is not None else 1
        inputs = BatchedData(data, batch_dim, shuffle_dim, mask, 5, batch_prop, device, False)

    profiler, detect_anomaly = handle_verbosity(verbose)

    for i in range(1, batch_prop_decay + 1):

        if min_std is not None:
            early_stop_callback = EarlyStopping(monitor="val_loss", verbose=False)
            learning_rate_monitor = LearningRateMonitor(logging_interval='epoch', )
            cb = [early_stop_callback, learning_rate_monitor]
        else:
            early_stop_callback = EarlyStopping(monitor="val_loss", verbose=False, patience=min_iter)
            cb = [early_stop_callback]

        if progress_bar:
            cb.append(LitProgressBar(leave=True))

        # invariance(model, L2='orthogonality', L3=None, max_iter=1000, iter_std=10)
        trainer = pl.Trainer(max_epochs=max_iter, min_epochs=min_iter,
                             accelerator='auto' if device == 'xpu' else device,
                             # strategy='ddp' if torch.cuda.is_available() else None,
                             strategy=slicetca.run.utils.SingleXPUStrategy() if device == 'xpu' else 'auto',
                             limit_train_batches=batch_num,
                             limit_val_batches=batch_num,
                             enable_progress_bar=progress_bar,
                             enable_model_summary=detect_anomaly,
                             enable_checkpointing=False,
                             callbacks=cb, profiler=profiler,
                             detect_anomaly=detect_anomaly,
                             # accumulate_grad_batches=30,
                             # precision='auto',
                             deterministic=True if seed is not None else False,
                             # reload_dataloaders_every_n_epochs=max_iter,
                             )
        true_prop = 1 - (1 - batch_prop) ** i
        inputs.prop = 1. if true_prop > .9 or i == batch_prop_decay else true_prop
        model.to(device)
        model.training = True
        model.trainer = trainer
        trainer.fit(model, datamodule=inputs)

    model.to('cpu')

    # trainer.fit(model, _feed(data, mask, batch_dim, batch_prop))

    return model.get_components(numpy=True), model


class LitProgressBar(TQDMProgressBar):

    def __init__(self, *args, **kwargs):
        super(LitProgressBar, self).__init__(*args, **kwargs)
        self.val_progress_bar = self.init_validation_tqdm()

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        n = trainer.global_step
        if self._should_update(n, self.val_progress_bar.total):
            _update_n(self.val_progress_bar, n)
        metrics = self.get_metrics(trainer, pl_module)
        self.val_progress_bar.set_postfix(metrics)

    def init_validation_tqdm(self) -> Tqdm:
        return Tqdm(disable=False, leave=True)

    def init_train_tqdm(self) -> Tqdm:
        return Tqdm(disable=True)

    def init_predict_tqdm(self) -> Tqdm:
        return Tqdm(disable=True)

    def init_test_tqdm(self) -> Tqdm:
        return Tqdm(disable=True)

    def on_validation_batch_start(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule",
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> None:
        pass

    def on_validation_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass

    def on_sanity_check_start(self, *_: Any) -> None:
        pass

    def on_sanity_check_end(self, *_: Any) -> None:
        pass

def _update_n(bar, value: int) -> None:
    if not bar.disable:
        bar.n = value
        bar.refresh()

def handle_device(dev, data, mask, model, compile):
    if dev is not None:
        device = dev
    elif torch.cuda.is_available():
        device = 'cuda'
    elif torch.xpu.is_available():
        device = 'xpu'
    else:
        device = 'cpu'

    data.to(device)
    if mask is not None:
        mask.to(device)

    model.set_loss(mask)

    if compile:
        model.to(device)
        model.compile(mode='reduce-overhead', fullgraph=True)

    return device

def default_loss(data, device):
    if data.dtype != torch.long:
        return torch.nn.MSELoss(reduction='sum')
    else:
        spikes_factorial = torch.tensor(scipy.special.factorial(
            data.numpy(force=True)), device=device)
        return partial(poisson_log_likelihood,
                                spikes_factorial=spikes_factorial)

def handle_verbosity(verbose):
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
    return profiler, detect_anomaly