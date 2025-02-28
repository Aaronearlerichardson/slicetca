from __future__ import annotations
import numpy as np
import warnings
from datetime import timedelta
from logging import getLogger
from typing import Iterable, Union, List, Dict, Any

from lightning.pytorch.accelerators import Accelerator, AcceleratorRegistry
from lightning.pytorch.strategies import SingleDeviceStrategy, StrategyRegistry
import torch
from torch import distributed as dist

default_pg_timeout = timedelta(seconds=1800)

logger = getLogger(__file__)

__all__ = ["block_mask"]
# IPEX is not absolutely required for XPU usage for torch>=2.5.0
if torch.xpu.is_available():
    __all__ += ["XPUAccelerator", "SingleXPUStrategy"]

    class XPUAccelerator(Accelerator):
        """
        Implements a Lightning Accelerator class for Intel GPU usage. Depends
        on Intel Extension for PyTorch to be installed.
        """

        @staticmethod
        def parse_devices(devices: Union[int, List[int]]) -> List[int]:
            """
            Parse the `trainer` input for devices and homogenize them.
            Parameters
            ----------
            devices : Union[int, List[int]]
                Single or list of device numbers to use
            Returns
            -------
            List[int]
                List of device numbers to use
            """
            if isinstance(devices, int):
                # assume that this is the number of devices to use
                devices = list(range(devices))
            return devices

        def setup_device(self, device: torch.device) -> None:
            """
            Configure the current process to use a specified device.
            Perhaps unreliably and misguiding, the IPEX implementation of this method
            tries to mirror the CUDA version but `ipex.xpu.set_device` actually refuses
            to accept anything other than an index. I've tried to work around this
            by grabbing the index from the device if possible, and just setting
            it to the first device if not using a distributed/multitile setup.
            """
            # first try and see if we can grab the index from the device
            index = getattr(device, "index", None)
            if index is None and not dist.is_initialized():
                index = 0
            torch.xpu.set_device(index)

        def teardown(self) -> None:
            # as it suggests, this is run on cleanup
            torch.xpu.empty_cache()

        def get_device_stats(self, device) -> Dict[str, Any]:
            return torch.xpu.memory_stats(device)

        @staticmethod
        def get_parallel_devices(devices: List[int]) -> List[torch.device]:
            """
            Return a list of torch devices corresponding to what is available.
            Essentially maps indices to `torch.device` objects.
            Parameters
            ----------
            devices : List[int]
                List of integers corresponding to device numbers
            Returns
            -------
            List[torch.device]
                List of `torch.device` objects for each device
            """
            return [torch.device("xpu", i) for i in devices]

        @staticmethod
        def auto_device_count() -> int:
            # by default, PVC has two tiles per GPU
            return torch.xpu.device_count()

        @staticmethod
        def is_available() -> bool:
            """
            Determines if an XPU is actually available.
            Returns
            -------
            bool
                True if devices are detected, otherwise False
            """
            try:
                return torch.xpu.device_count() != 0
            except (AttributeError, NameError):
                return False

        @classmethod
        def register_accelerators(cls, accelerator_registry) -> None:
            accelerator_registry.register(
                "xpu",
                cls,
                description="Intel Data Center GPU Max - codename Ponte Vecchio",
            )

    # add PVC to the registry
    AcceleratorRegistry.register("xpu", XPUAccelerator)

    class SingleXPUStrategy(SingleDeviceStrategy):
        """
        This class implements the strategy for using a single PVC tile.
        """

        strategy_name = "pvc_single"

        def __init__(
            self,
            device: str | None = "xpu",
            checkpoint_io=None,
            precision_plugin=None,
        ):
            super().__init__(
                device=device,
                accelerator=XPUAccelerator(),
                checkpoint_io=checkpoint_io,
                precision_plugin=precision_plugin,
            )

        @property
        def is_distributed(self) -> bool:
            return False

        def setup(self, trainer) -> None:
            self.model_to_device()
            super().setup(trainer)

        def setup_optimizers(self, trainer) -> None:
            super().setup_optimizers(trainer)

        def model_to_device(self) -> None:
            self.model.to(self.root_device)

        @classmethod
        def register_strategies(cls, strategy_registry) -> None:
            strategy_registry.register(
                cls.strategy_name,
                cls,
                description=f"{cls.__class__.__name__} - uses a single XPU tile for compute.",
            )

    StrategyRegistry.register(
        "single_xpu",
        SingleXPUStrategy,
        description="Strategy utilizing a single Intel GPU device or tile.",
    )
else:
    logger.warning(
        "IPEX was not installed or XPU is not available. `slicetca.run.utils.xpu` will be empty."
    )

def block_mask(dimensions: Iterable[int],
               train_blocks_dimensions: Iterable[int],
               test_blocks_dimensions: Iterable[int],
               number_blocks: int = None,
               fraction_test: float = 0.1,
               exact:bool = True,
               device:str = 'cpu'):
    """
    Builds train and test masks.
    The train mask has block of entries masked.
    The test mask has the opposite entries masked, plus the boundaries of the blocks.

    :param dimensions: Dimensions of the mask.
    :param train_blocks_dimensions: Dimensions of the blocks discarded for training will be 2*train_blocks_dimensions+1
    :param test_blocks_dimensions: Dimensions of the blocks retained for testing will be 2*test_blocks_dimensions+1
    :param number_blocks: The number of blocks. Deprecated, use fraction_test.
    :param fraction_test: The maximum fraction of entries in the test_mask
    :param exact:   If exact then the number of blocks will be number_blocks (slower).
                    If not exact, the number of blocks will be on average number_blocks (faster).
    :param device: torch device (e.g. 'cuda' or 'cpu').
    :return: train_mask, test_mask
    """

    valence = len(dimensions)

    flattened_max_dim = np.prod(dimensions)

    if not np.prod((np.array(train_blocks_dimensions)-np.array(test_blocks_dimensions))>=0):
        raise Exception('For all i it should be that train_blocks_dimensions[i]>=test_blocks_dimensions[i].')

    if number_blocks is None:
        number_blocks = int(fraction_test * flattened_max_dim / np.prod(2*np.array(test_blocks_dimensions)+1))
    else:
        warnings.warn('The parameter number_blocks is deprecated, use fraction_test instead.', DeprecationWarning)

    if exact:
        start = torch.zeros(flattened_max_dim, device=device)
        start[:number_blocks] = 1
        start = start[torch.randperm(flattened_max_dim, device=device)]
        start = start.reshape(dimensions)
    else:
        start = (torch.rand(tuple(dimensions), device=device) < fraction_test).long()

    start_index = start.nonzero()
    number_blocks = len(start_index)

    # Build outer-blocks mask
    a = [[slice(torch.clip(start_index[j][i]-train_blocks_dimensions[i],min=0, max=dimensions[i]),
                torch.clip(start_index[j][i]+train_blocks_dimensions[i]+1,min=0, max=dimensions[i]))
          for i in range(valence)] for j in range(number_blocks)]

    train_mask = torch.full(dimensions, True, device=device)

    for j in a: train_mask[j] = 0

    # Build inner-blocks tensor
    a = [[slice(torch.clip(start_index[j][i]-test_blocks_dimensions[i],min=0, max=dimensions[i]),
                torch.clip(start_index[j][i]+test_blocks_dimensions[i]+1,min=0, max=dimensions[i]))
                for i in range(valence)] for j in range(number_blocks)]

    test_mask = torch.full(dimensions, False, device=device)

    for j in a: test_mask[j] = 1

    return train_mask, test_mask
