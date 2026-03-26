from functools import partial
from typing import Sequence, Union

import lightning.pytorch as pl
import numpy as np
import scipy
import torch
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor

import slicetca.run.utils
from slicetca.core import SliceTCA, TCA
from slicetca.core.helper_functions import poisson_log_likelihood
from slicetca.invariance.invariance import invariance
from slicetca.run.data import BatchedData, MaskedData


def decompose(
    data: Union[torch.Tensor, np.array],
    number_components: Union[Sequence[int], int],
    positive: bool = False,
    initialization: str = "uniform",
    learning_rate: float = 5 * 10**-3,
    batch_dim: int = None,
    avg_batches: int = 1,
    shuffle_dim: int | tuple[int] = (),
    max_iter: int = 10000,
    min_iter: int = 10,
    min_std: float = None,
    iter_std: int = 100,
    mask: torch.Tensor = None,
    progress_bar: bool = True,
    seed: int = 7,
    weight_decay: float = None,
    batch_prop_decay: int = 1,
    batch_prop: float = 1.0,
    init_bias: float = 0.0,
    loss_function: callable = None,
    device: str = None,
    verbose: int = 0,
    compile: bool = False,
    regularization: str = None,
    dtype: torch.dtype = None,
    testing: bool = False,
    blocklength: int = 10,
    **kwargs,
) -> (list, Union[SliceTCA, TCA]):
    """
    High-level function to decompose a data tensor into a SliceTCA or TCA decomposition.
    All extra kwargs are passed directly to `lightning.pytorch.Trainer`.
    """

    if seed is not None:
        pl.seed_everything(seed, workers=True)

    data = _ensure_tensor(data)
    if loss_function is None:
        loss_function = default_loss(data, device)

    model_cls = _select_decomposition(number_components)
    min_std, iter_std = _resolve_stopping(min_std, iter_std)
    dtype = data.dtype if dtype is None else dtype
    dimensions = list(data.shape)
    if batch_dim is not None:
        dimensions.pop(batch_dim)

    model = model_cls(
        dimensions,
        number_components,
        positive,
        initialization,
        dtype=dtype,
        lr=learning_rate,
        weight_decay=weight_decay,
        loss=loss_function,
        init_bias=init_bias,
        threshold=min_std,
        patience=iter_std,
    )

    profiler, detect_anomaly = handle_verbosity(verbose)
    device = handle_device(device, model, mask, compile)
    batch_num, inputs = _build_inputs(
        data=data,
        mask=mask,
        batch_dim=batch_dim,
        batch_prop=batch_prop,
        shuffle_dim=shuffle_dim,
        testing=testing,
        blocklength=blocklength,
        avg_batches=avg_batches,
    )

    for i in range(1, batch_prop_decay + 1):
        callbacks = _build_callbacks(min_std=min_std, min_iter=min_iter)

        if regularization is not None:
            model.to(device)
            invariance(model, L2=regularization, L3=None, max_iter=1000, iter_std=10)
            model.to("cpu")

        trainer = _build_trainer(
            device=device,
            max_iter=max_iter,
            min_iter=min_iter,
            batch_num=batch_num,
            progress_bar=progress_bar,
            detect_anomaly=detect_anomaly,
            profiler=profiler,
            callbacks=callbacks,
            deterministic=(seed is not None),
            trainer_kwargs=kwargs,
        )

        true_prop = 1 - (1 - batch_prop) ** i
        inputs.prop = 1.0 if true_prop > 0.9 or i == batch_prop_decay else true_prop

        model.to(device)
        model.training = True
        model.trainer = trainer
        if device == "cuda":
            torch.cuda.empty_cache()
        trainer.fit(model, datamodule=inputs)

    model.to("cpu")
    return model.get_components(numpy=True), model


def _ensure_tensor(data: Union[torch.Tensor, np.array]) -> torch.Tensor:
    if isinstance(data, np.ndarray):
        return torch.tensor(data)
    if isinstance(data, torch.Tensor):
        return data
    raise ValueError("data must be a torch.Tensor or a numpy.ndarray")


def _select_decomposition(number_components: Union[Sequence[int], int]):
    if isinstance(number_components, int):
        return TCA
    return TCA if len(number_components) == 1 else SliceTCA


def _resolve_stopping(min_std: float, iter_std: int) -> tuple[float, int]:
    if min_std is None:
        return min_std, iter_std
    return min_std * 2, iter_std // 2


def _build_inputs(
    data: torch.Tensor,
    mask: torch.Tensor,
    batch_dim: int,
    batch_prop: float,
    shuffle_dim: int | tuple[int],
    testing: bool,
    blocklength: int,
    avg_batches: int,
):
    if batch_dim is None:
        inputs = MaskedData(data, mask, 5, batch_prop, shuffle_dim, testing, blocklength)
        return 1, inputs
    inputs = BatchedData(data, batch_dim, shuffle_dim, mask, 5, batch_prop, testing, avg_batches=avg_batches)
    return 1.0, inputs


def _build_callbacks(min_std: float, min_iter: int):
    if min_std is not None:
        return [
            EarlyStopping(monitor="val_loss", verbose=False),
            LearningRateMonitor(logging_interval="epoch"),
        ]
    return [EarlyStopping(monitor="val_loss", verbose=False, patience=min_iter)]


def _build_trainer(
    device: str,
    max_iter: int,
    min_iter: int,
    batch_num: float,
    progress_bar: bool,
    detect_anomaly: bool,
    profiler,
    callbacks,
    deterministic: bool,
    trainer_kwargs: dict,
) -> pl.Trainer:
    trainer_kwargs = dict(trainer_kwargs)
    trainer_kwargs.setdefault("num_sanity_val_steps", 0)
    return pl.Trainer(
        max_epochs=max_iter,
        min_epochs=min_iter,
        accelerator="auto" if device == "xpu" else device,
        strategy=slicetca.run.utils.SingleXPUStrategy() if device == "xpu" else "auto",
        limit_train_batches=batch_num,
        limit_val_batches=batch_num,
        enable_progress_bar=progress_bar,
        enable_model_summary=detect_anomaly,
        enable_checkpointing=False,
        callbacks=callbacks,
        profiler=profiler,
        detect_anomaly=detect_anomaly,
        deterministic=deterministic,
        **trainer_kwargs,
    )


def handle_device(dev, model, mask, compile):
    if dev is not None:
        device = dev
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.xpu.is_available():
        device = "xpu"
    else:
        device = "cpu"

    model.set_loss(mask)

    if compile:
        model.to(device)
        model.compile(mode="reduce-overhead", fullgraph=True)

    return device


def default_loss(data, device):
    if data.dtype != torch.long:
        return torch.nn.MSELoss(reduction="sum")
    spikes_factorial = torch.tensor(scipy.special.factorial(data.numpy(force=True)), device=device)
    return partial(poisson_log_likelihood, spikes_factorial=spikes_factorial)


def handle_verbosity(verbose):
    if verbose == 0:
        return None, False
    if verbose == 1:
        return "simple", False
    if verbose == 2:
        return "advanced", False
    if verbose == 3:
        return None, True
    raise ValueError("verbose must be 0, 1, 2, or 3")
