import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
import numpy as np
from collections.abc import Iterable

from typing import Any, Sequence, Union, Callable
from slicetca.core.helper_functions import generate_square_wave_tensor
import pytorch_lightning as pl


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
                 loss: callable = nn.MSELoss(reduction='none')):
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
            if initialization in ['normal', 'uniform', 'steps'] : init_weight = 1/np.sqrt(sum(ranks))
            elif initialization == 'uniform-positive': init_weight = ((0.5 / sum(ranks)) ** (1 / max([len(p) for p in partitions])))*2
            else: raise Exception('Undefined initialization, select one of : normal, uniform, uniform-positive')

        if isinstance(positive, bool):
            if positive: positive_function = [[torch.abs for j in i] for i in partitions]
            else: positive_function = [[self.identity for j in i] for i in partitions]
        elif isinstance(positive, tuple) or isinstance(positive, list): positive_function = positive

        vectors = nn.ModuleList([])

        init_params = dict(device=self.device, dtype=dtype)
        for i in range(len(ranks)):
            r = ranks[i]
            dim = components[i]

            # k-tensors of the outer product
            if initialization == 'normal':
                v = [nn.Parameter(positive_function[i][j](torch.randn([r]+d, **init_params)*init_weight + init_bias)) for j, d in enumerate(dim)]
            elif initialization == 'uniform':
                v = [nn.Parameter(positive_function[i][j](2*(torch.rand([r] + d, **init_params)-0.5)*init_weight + init_bias)) for j, d in enumerate(dim)]
            elif initialization == 'uniform-positive':
                v = [nn.Parameter(positive_function[i][j](torch.rand([r] + d, **init_params)*init_weight + init_bias)) for j, d in enumerate(dim)]
            elif initialization == 'steps':
                v = [nn.Parameter(positive_function[i][j](generate_square_wave_tensor(*([r] + d))*init_weight + init_bias)) for j, d in enumerate(dim)]
            else:
                raise Exception('Undefined initialization, select one of : normal, uniform, uniform-positive')

            vectors.append(nn.ParameterList(v))

        self.vectors = vectors

        self.dimensions = dimensions
        self.partitions = partitions
        self.ranks = ranks
        self.positive = positive
        self.initialization = initialization
        self.init_weight = init_weight
        self.init_bias = init_bias
        self.to(dtype)

        self.components = components
        self.positive_function = positive_function
        self.valence = len(dimensions)

        self.loss = loss
        self.losses = []
        self._lr = lr
        self._weight_decay = weight_decay
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

    def construct_single_component(self, partition: int, k: int):
        """
        Constructs the kth term of the given partition.

        :param partition: Type of the partition
        :param k: Number of the component
        :return: Tensor of shape self.dimensions
        """
        temp = []
        for q in range(len(self.components[partition])):
            func = self.positive_function[partition][q]
            temp.append(func(self.vectors[partition][q][k]))
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

        temp = torch.zeros(self.dimensions, dtype=self.dtype, device=self.device)
        for j in range(self.ranks[partition]):
            temp += self.construct_single_component(partition, j)

        return temp

    def construct(self):
        """
        Constructs the full tensor.
        :return: Tensor of shape self.dimensions
        """

        temp = torch.zeros(self.dimensions, dtype=self.dtype, device=self.device)

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
                    temp[i].append( self.positive_function[i][j](self.vectors[i][j]).data.detach().cpu().numpy())
                else:
                    if not detach: temp[i].append(self.positive_function[i][j](self.vectors[i][j]).data.detach())
                    else: temp[i].append(self.positive_function[i][j](self.vectors[i][j]).data)

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
        X *= mask
        X_hat = self.construct()
        X_hat *= mask
        loss = self.loss(X, X_hat).sum(dtype=torch.float64) / mask.sum(dtype=torch.int64)
        self.losses.append(loss.item())
        self.log("train_loss", loss, on_step=True,
                 on_epoch=True, prog_bar=True, logger=True)
        return loss

    # def on_train_epoch_end(self):
    #     # do something with all training_step outputs, for example:
    #     start_idx = self.trainer.num_training_batches * self.current_epoch
    #     epoch_mean = torch.as_tensor(self.losses[start_idx:]).mean()
    #     # pl_module.log("training_mean", epoch_mean.item(), prog_bar=True)
    #     super().on_train_epoch_end()

    def configure_optimizers(self):
        if self._weight_decay is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=self._lr)
        else:
            optimizer = torch.optim.AdamW(self.parameters(), lr=self._lr,
                                          weight_decay=self._weight_decay)
        super().configure_optimizers()
        return optimizer


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
                 loss: callable = nn.MSELoss(reduction='none')):
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
                         loss=loss)


class TCA(PartitionTCA):
    def __init__(self,
                 dimensions: Sequence[int],
                 rank: int,
                 positive: bool = False,
                 initialization: str = 'uniform',
                 lr: float = 5 * 10 ** -3,
                 weight_decay: float = None,
                 init_weight: float = None,
                 init_bias: float = 0.0,
                 dtype: torch.dtype = torch.float64,
                 loss: callable = nn.MSELoss(reduction='none')):
        """
        Main TCA decomposition class.

        :param dimensions: Dimensions of the data to decompose.
        :param rank: Number of components.
        :param positive: If False does nothing.
                         If True constrains all components to be positive.
                         If list of list, the list of functions to apply to a given partition and component.
        :param initialization: Components initialization 'uniform'~U(-1,1), 'uniform-positive'~U(0,1), 'normal'~N(0,1).
        :param init_weight: Coefficient to multiply the initial component by.
        :param init_bias: Coefficient to add to the initial component.
        :param device: Torch device.
        """

        if not isinstance(rank, Iterable):
            rank = (rank,)

        valence = len(dimensions)
        partitions = [[[j] for j in range(valence)]]

        super().__init__(dimensions=dimensions,
                         ranks=rank,
                         partitions=partitions,
                         positive=positive,
                         initialization=initialization,
                         lr=lr,
                         weight_decay=weight_decay,
                         init_weight=init_weight,
                         init_bias=init_bias,
                         dtype=dtype,
                         loss=loss)
