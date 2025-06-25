import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn, optim, jit
import numpy as np
from collections.abc import Iterable

from typing import Any, Sequence, Union, Callable
from slicetca.core.helper_functions import generate_orthogonal_tensor
import lightning.pytorch as pl
import functools


def trial_average(X, mask=None, axis=None):
    if mask is None:
        mask = torch.ones_like(X, dtype=torch.bool)
    # Apply the mask to the matrix X
    masked_X = X * mask

    # Calculate the sum of the masked elements for each column
    sum_masked_X = masked_X.sum(dim=axis)

    # Calculate the count of the non-masked elements for each column
    count_non_masked = mask.sum(dim=axis)

    # Compute the mean by dividing the sum by the count for each column
    return sum_masked_X / count_non_masked


def explained_variance(X, X_hat, mask=None, axis=None):
    if mask is None:
        mask = torch.ones_like(X, dtype=torch.bool, device=X.device)
    masked_X = X * mask
    masked_X_hat = X_hat * mask
    count_non_masked = mask.sum(dim=axis)
    X_centered = X - masked_X.sum(dim=axis) / count_non_masked
    X_hat_centered = X_hat - masked_X_hat.sum(dim=axis) / count_non_masked
    masked_X_centered = X_centered * mask
    masked_X_hat_centered = X_hat_centered * mask
    diff = masked_X_centered - masked_X_hat_centered
    out = diff.pow(2).sum(dim=axis) / masked_X_centered.pow(2).sum(dim=axis)
    out[out > 1] = 1 / out[out > 1]
    return out


def error(X, X_hat, mask=None, axis=None):
    if mask is None:
        mask = torch.ones_like(X, dtype=torch.bool)
    diff = X - X_hat
    diff_masked = diff * mask
    count_non_masked = mask.sum(dim=axis)
    return diff_masked.abs().sum(axis) / count_non_masked


def set_loss(loss_fn, has_mask):
    if loss_fn.reduction == 'none':
        if has_mask:
            loss_calc = functools.partial(loss_fn_with_mask, loss_fn=loss_fn)
        else:
            loss_calc = functools.partial(loss_fn_no_mask, loss_fn=loss_fn)
    elif loss_fn.reduction == 'sum':
        if has_mask:
            loss_calc = functools.partial(loss_fn_sum_with_mask, loss_fn=loss_fn)
        else:
            loss_fn.reduction = 'mean'
            loss_calc = functools.partial(loss_fn_no_mask, loss_fn=loss_fn)
    elif loss_fn.reduction == 'mean':
        if has_mask:
            loss_calc = functools.partial(loss_fn_mean_with_mask, loss_fn=loss_fn)
        else:
            loss_calc = functools.partial(loss_fn_no_mask, loss_fn=loss_fn)
    else:
        raise ValueError('Invalid reduction method for loss function.')

    return loss_calc

def loss_fn_with_mask(X, X_hat, mask, loss_fn):
    X_mask = torch.where(mask, X, 0)
    X_hat_mask = torch.where(mask, X_hat, 0)
    return loss_fn(X_mask, X_hat_mask).sum(dtype=torch.float32) / mask.sum(dtype=torch.float32)

def loss_fn_no_mask(X, X_hat, mask, loss_fn):
    return loss_fn(X, X_hat).mean()

def loss_fn_sum_with_mask(X, X_hat, mask, loss_fn):
    X_mask = torch.where(mask, X, 0)
    X_hat_mask = torch.where(mask, X_hat, 0)
    return loss_fn(X_mask, X_hat_mask) / mask.sum(dtype=torch.float32)

def loss_fn_mean_with_mask(X, X_hat, mask, loss_fn):
    return loss_fn(X[mask], X_hat[mask])

class PartitionTCA(pl.LightningModule):

    def __init__(self,
                 dimensions: Sequence[int],
                 partitions: Sequence[Sequence[Sequence[int]]],
                 ranks: Sequence[int],
                 positive: Union[bool, Sequence[Sequence[Callable]]] = False,
                 initialization: str = 'uniform',
                 lr: float = 5 * 10 ** -3,
                 weight_decay: float = None,
                 init_weight: float = None,
                 init_bias: float = 0.0,
                 dtype: torch.dtype = torch.float64,
                 loss: callable = nn.MSELoss(reduction='none'),
                 threshold: float = None,
                 patience: int = 10):
        """
        Parent class for the sliceTCA and TCA decompositions.

        :param dimensions: Dimensions of the data to decompose.
        :param partitions: List of partitions of the legs of the tensor.
                        [[[0],[1]]] would be a matrix rank decomposition.
        :param ranks: Number of components of each partition.
        :param positive: If False does nothing.
                         If True constrains all components to be positive.
                         If list of list, the list of functions to apply to a given partition and component.
        :param initialization: Components initialization 'uniform'~U(-1,1), 'uniform-positive'~U(0,1), 'normal'~N(0,1).
        :param init_weight: Coefficient to multiply the initial component by.
        :param init_bias: Coefficient to add to the initial component.
        :param device: Torch device.
        """

        super(PartitionTCA, self).__init__()

        components = [[[dimensions[k] for k in j] for j in i] for i in partitions]

        if init_weight is None:
            if initialization in ['normal', 'uniform'] : init_weight = 1/np.sqrt(sum(ranks))
            elif initialization in ['uniform-positive', 'orthogonal'] : init_weight = ((0.5 / sum(ranks)) ** (1 / max([len(p) for p in partitions])))*2
            else: raise Exception('Undefined initialization, select one of : normal, uniform, uniform-positive')

        if isinstance(positive, bool):
            if positive: positive_function = torch.abs
            else: positive_function = self.identity
        elif isinstance(positive, tuple) or isinstance(positive, list): positive_function = positive
        else: raise Exception('Positive must be a boolean or a list of functions')

        vectors = nn.ModuleList([])
        init_params = dict(device=self.device, dtype=dtype)
        self.to(**init_params)
        for i in range(len(ranks)):
            r = ranks[i]
            dim = components[i]

            # k-tensors of the outer product
            if initialization == 'normal':
                v = [nn.Parameter(positive_function(torch.randn([r]+d, **init_params)*init_weight + init_bias)) for j, d in enumerate(dim)]
            elif initialization == 'uniform':
                v = [nn.Parameter(positive_function(2*(torch.rand([r] + d, **init_params)-0.5)*init_weight + init_bias)) for j, d in enumerate(dim)]
            elif initialization == 'uniform-positive':
                v = [nn.Parameter(positive_function(torch.rand([r] + d, **init_params)*init_weight + init_bias)) for j, d in enumerate(dim)]
            elif initialization == 'orthogonal':
                v = [nn.Parameter(positive_function(generate_orthogonal_tensor(*([r] + d), positive=True, **init_params)*init_weight + init_bias)) for j, d in enumerate(dim)]
            else:
                raise Exception('Undefined initialization, select one of : normal, uniform, uniform-positive')

            vectors.append(nn.ParameterList(v))

        self.vectors = vectors

        self.dimensions = dimensions
        self.partitions = partitions
        self.ranks = torch.as_tensor(ranks, device=self.device, dtype=torch.int32)
        self.positive = positive
        self.initialization = initialization
        self.init_weight = init_weight
        self.init_bias = init_bias

        self.components = components
        self.positive_function = positive_function
        self.valence = len(dimensions)

        self.loss = loss
        # # check for 'dtype' and 'device' attributes and set them if not present
        # if not hasattr(self, 'dtype'):
        #     self.dtype = dtype
        # if not hasattr(self, 'device'):
        #     self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.losses = []
        self._lr = lr
        self._weight_decay = weight_decay
        self._threshold = threshold
        self._patience = patience
        self._cache = {}

        self.inverse_permutations = []
        self.flattened_permutations = []
        for i in self.partitions:
            temp = []
            for j in i:
                for k in j:
                    temp.append(k)
            self.flattened_permutations.append(temp)
            self.inverse_permutations.append(torch.argsort(torch.tensor(temp)).tolist())

        self.set_einsums()

    def identity(self, x):
        return x

    def explained_variance(self, X, mask=None, axis=None):
        x_hat = self.construct()
        if X.device != x_hat.device:
            x_hat = x_hat.to(X.device)
        return explained_variance(X, x_hat, mask, axis)

    def error(self, X, mask=None, axis=None):
        return error(X, self.construct(), mask, axis)

    def set_einsums(self):

        self.einsums = []
        for i in self.partitions:
            lhs = ''
            rhs = ''
            for j in range(len(i)):
                for k in i[j]:
                    lhs += chr(105 + k)
                    rhs += chr(105 + k)
                if j != len(i) - 1:
                    lhs += ','
            self.einsums.append(lhs + '->' + rhs)

    def set_loss(self, mask):
        self._loss_calc = set_loss(self.loss, mask is not None)

    def construct_single_component(self, partition: int, k: int):
        """
        Constructs the kth term of the given partition.

        :param partition: Type of the partition
        :param k: Number of the component
        :return: Tensor of shape self.dimensions
        """
        temp = []
        for q in range(len(self.components[partition])):
            temp.append(self.positive_function(self.vectors[partition][q][k]))
        # temp = [self.positive_function[partition][q](self.vectors[partition][q][k]) for q in range(len(self.components[partition]))]
        outer = torch.einsum(self.einsums[partition], temp)
        outer = outer.permute(self.inverse_permutations[partition])

        return outer

    def construct_single_partition(self, partition: int):
        """
        Constructs the sum of the terms of a given type of partition.

        :param partition: Type of the partition
        :return: Tensor of shape self.dimensions
        """

        temp = torch.zeros(self.dimensions, dtype=self.vectors[0][0].dtype, device=self.vectors[0][0].device)
        for j in range(self.ranks[partition].item()):
            temp += self.construct_single_component(partition, j)

        return temp

    def construct(self):
        """
        Constructs the full tensor.
        :return: Tensor of shape self.dimensions
        """

        temp = torch.zeros(self.dimensions, dtype=self.vectors[0][0].dtype, device=self.vectors[0][0].device)

        for i in range(len(self.partitions)):
            for j in range(self.ranks[i]):
                temp += self.construct_single_component(i, j)

        return temp

    def forward(self):
        return self.construct()

    def get_components(self, detach=False, numpy=False):
        """
        Returns the components of the model.
        
        :param detach: Whether to detach the gradient.
        :param numpy: Whether to cast them to numpy arrays.
        :return: list of list of tensors.
        """

        temp = [[] for _ in range(len(self.vectors))]

        for i in range(len(self.vectors)):
            for j in range(len(self.vectors[i])):
                if numpy:
                    temp[i].append( self.positive_function(self.vectors[i][j]).data.detach().cpu().numpy())
                else:
                    if not detach: temp[i].append(self.positive_function(self.vectors[i][j]).data.detach())
                    else: temp[i].append(self.positive_function(self.vectors[i][j]).data)

        return temp

    def set_components(self, components: Sequence[Sequence[torch.Tensor]]):  # bug if positive_function != abs
        """
        Set the model's components. 
        If the positive functions are abs or the identity model.set_components(model.get_components) 
        has no effect besides resetting the gradient.
        
        :param components: list of list tensors.
        """

        for i in range(len(self.vectors)):
            for j in range(len(self.vectors[i])):
                with torch.no_grad():
                    if isinstance(components[i][j], torch.Tensor):
                        self.vectors[i][j].copy_(components[i][j].to(self.device))
                    else:
                        self.vectors[i][j].copy_(torch.tensor(components[i][j], device=self.device))
        self.zero_grad()

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        X, mask = batch
        loss = self._loss_calc(X.squeeze(), self.construct(), mask.squeeze())
        self.log_dict({"train_loss": loss}, prog_bar=True, logger=True,
                      add_dataloader_idx=False, sync_dist=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        X, mask = batch
        loss = self._loss_calc(X.squeeze(), self.construct(), mask.squeeze())
        to_log = {"val_loss": loss}
        self.log_dict(to_log, prog_bar=True, logger=True,
                      add_dataloader_idx=False, sync_dist=True)
        self.losses.append(loss.item())
        return loss

    def configure_optimizers(self):
        eps = 1e-9 if self.dtype != torch.float16 else 1e-7
        if self._weight_decay is None:
            optimizer = torch.optim.AdamW(self.parameters(), self._lr, eps=eps)
        elif callable(self._weight_decay):
            optimizer = self._weight_decay(self.parameters(), self._lr)
        elif issubclass(type(self._weight_decay), torch.optim.Optimizer):
            optimizer = type(self._weight_decay)(self.parameters(), self._lr)
        else:
            optimizer = torch.optim.AdamW(self.parameters(), self._lr, eps=eps,
                                                # momentum=0.9, centered=True,
                                          weight_decay=self._weight_decay)
        if self._threshold is None:
            return optimizer
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5,
            patience=self._patience, threshold=self._threshold)
        # lr_scheduler = CosineWarmupScheduler(optimizer, self._patience,
        #                                      self.trainer.max_epochs)
        # lr_scheduler = torch.optim.swa_utils.SWALR(
        #     optimizer, anneal_strategy = "cos",
        #     anneal_epochs = self._patience, swa_lr = self._threshold)
        lr_scheduler_config = {
            # REQUIRED: The scheduler instance
            "scheduler": lr_scheduler,
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": 'epoch',
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 1,
            # Metric to to monitor for schedulers like `ReduceLROnPlateau`
            "monitor": "val_loss",
            # If set to `True`, will enforce that the value specified 'monitor'
            # is available when the scheduler is updated, thus stopping
            # training if not found. If set to `False`, it will only produce a warning
            "strict": True,
            # If using the `LearningRateMonitor` callback to monitor the
            # learning rate progress, this keyword can be used to specify
            # a custom logged name
            "name": "learning_rate",
        }
        super().configure_optimizers()
        return {
                "optimizer": optimizer,
                "lr_scheduler": lr_scheduler_config,
                }

    def copy(self):
        """
        Returns a copy of the model.
        """
        out = self.__class__(dimensions=self.dimensions,
                              ranks=self.ranks,
                              positive=self.positive,
                              initialization=self.initialization,
                              lr=self._lr,
                              weight_decay=self._weight_decay,
                              init_weight=self.init_weight,
                              init_bias=self.init_bias,
                              dtype=self.dtype,
                              loss=self.loss)
        out.set_components(self.get_components())
        return out


class SliceTCA(PartitionTCA):
    def __init__(self, 
                 dimensions: Sequence[int], 
                 ranks: Sequence[int], 
                 positive: bool = False, 
                 initialization: str = 'uniform',
                 lr: float = 5 * 10 ** -3,
                 weight_decay: float = None,
                 init_weight: float = None,
                 init_bias: float = 0.0,
                 dtype: torch.dtype = torch.float64,
                 loss: callable = nn.MSELoss(),
                 threshold: float = None,
                 patience: int = 10):
        """
        Main sliceTCA decomposition class.

        :param dimensions: Dimensions of the data to decompose.
        :param ranks: Number of components of each slice type.
        :param positive: If False does nothing.
                         If True constrains all components to be positive.
                         If list of list, the list of functions to apply to a given partition and component.
        :param initialization: Components initialization 'uniform'~U(-1,1), 'uniform-positive'~U(0,1), 'normal'~N(0,1).
        :param init_weight: Coefficient to multiply the initial component by.
        :param init_bias: Coefficient to add to the initial component.
        :param device: Torch device.
        """

        valence = len(dimensions)
        partitions = [[[i], [j for j in range(valence) if j != i]] for i in range(valence)]

        super().__init__(dimensions=dimensions,
                         ranks=ranks,
                         partitions=partitions,
                         positive=positive,
                         initialization=initialization,
                         lr=lr,
                         weight_decay=weight_decay,
                         init_weight=init_weight,
                         init_bias=init_bias,
                         dtype=dtype,
                         loss=loss,
                         threshold=threshold,
                         patience=patience)


class TCA(PartitionTCA):
    def __init__(self,
                 dimensions: Sequence[int],
                 ranks: int,
                 positive: bool = False,
                 initialization: str = 'uniform',
                 lr: float = 5 * 10 ** -3,
                 weight_decay: float = None,
                 init_weight: float = None,
                 init_bias: float = 0.0,
                 dtype: torch.dtype = torch.float64,
                 loss: callable = nn.MSELoss(),
                 threshold: float = None,
                 patience: int = 10):
        """
        Main TCA decomposition class.

        :param dimensions: Dimensions of the data to decompose.
        :param ranks: Number of components.
        :param positive: If False does nothing.
                         If True constrains all components to be positive.
                         If list of list, the list of functions to apply to a given partition and component.
        :param initialization: Components initialization 'uniform'~U(-1,1), 'uniform-positive'~U(0,1), 'normal'~N(0,1).
        :param init_weight: Coefficient to multiply the initial component by.
        :param init_bias: Coefficient to add to the initial component.
        :param device: Torch device.
        """

        if not isinstance(ranks, Iterable):
            ranks = (ranks,)

        valence = len(dimensions)
        partitions = [[[j] for j in range(valence)]]

        super().__init__(dimensions=dimensions,
                         ranks=ranks,
                         partitions=partitions,
                         positive=positive,
                         initialization=initialization,
                         lr=lr,
                         weight_decay=weight_decay,
                         init_weight=init_weight,
                         init_bias=init_bias,
                         dtype=dtype,
                         loss=loss,
                         threshold=threshold,
                         patience=patience)


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor
