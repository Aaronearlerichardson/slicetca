import lightning as L
import torch
from slicetca.run.utils import block_mask
from torch.utils.data import DataLoader, IterableDataset


class BatchedData(L.LightningDataModule):
    def __init__(self, data: torch.Tensor, dim: int,
                 shuffle_dim = (0,), mask: torch.Tensor = None,
                 n_folds: int = 5, prop: float = 1.0,  device: str = None, test: bool = False):
        super().__init__()

        self.batch_dim = dim
        self.batch_size = 1 if dim is None else data.shape[dim]
        self.shuffle_dims = shuffle_dim if isinstance(shuffle_dim, tuple) else (shuffle_dim,)
        self.n_folds = n_folds
        self.prop = prop
        self.test = test
        self.device = self.set_device(device)
        data, mask = handle_data(data, mask, device)
        self.dims = data.shape
        # self._setup(data, mask, n_folds, dim)
        perm = torch.randperm(data.shape[dim], device=self.device)
        idx = tuple(slice(None) if i != dim else perm for i in range(data.ndim))
        self.data = data[idx]
        self.mask = mask[idx]
        fold_size = self.data.shape[dim] // n_folds
        data_slice = [slice(None)] * self.data.ndim
        data_slice[dim] = slice(0, fold_size)
        self.val_data = self.data[tuple(data_slice)]
        self.val_mask = self.mask[tuple(data_slice)]
        if self.test:
            data_slice[dim] = slice(fold_size, 2 * fold_size)
            self.test_data = self.data[tuple(data_slice)]
            self.test_mask = self.mask[tuple(data_slice)]
            data_slice[dim] = slice(2 * fold_size, None)
            self.train_data = self.data[tuple(data_slice)]
            self.train_mask = self.mask[tuple(data_slice)]
        else:
            self.test_data = None
            self.test_mask = None
            data_slice[dim] = slice(fold_size, None)
            self.train_data = self.data[tuple(data_slice)]
            self.train_mask = self.mask[tuple(data_slice)]

    def set_device(self, device):
        if device is not None:
            return device
        elif torch.cuda.is_available():
            return 'cuda'
        elif torch.xpu.is_available():
            return 'xpu'
        else:
            return 'cpu'

    def _setup(self, data, mask, n_folds, dim):
        data = torch.as_tensor(data)
        mask = torch.as_tensor(mask)
        perm = torch.randperm(data.shape[dim], device=self.device)
        self.data = torch.take_along_dim(data, perm, dim)
        self.mask = torch.take_along_dim(mask, perm, dim)
        if self.data.device.type != self.device:
            self.data.pin_memory(self.device)
            self.mask.pin_memory(self.device)
        fold_size = self.data.shape[dim] // n_folds
        data_slice = [slice(None)] * self.data.ndim
        data_slice[dim] = slice(0, fold_size)
        self.val_data = self.data[tuple(data_slice)]
        self.val_mask = self.mask[tuple(data_slice)]
        if self.test:
            data_slice[dim] = slice(fold_size, 2 * fold_size)
            self.test_data = self.data[tuple(data_slice)]
            self.test_mask = self.mask[tuple(data_slice)]
            data_slice[dim] = slice(2 * fold_size, None)
            self.train_data = self.data[tuple(data_slice)]
            self.train_mask = self.mask[tuple(data_slice)]
        else:
            self.test_data = None
            self.test_mask = None
            data_slice[dim] = slice(fold_size, None)
            self.train_data = self.data[tuple(data_slice)]
            self.train_mask = self.mask[tuple(data_slice)]


    def setup(self, stage: str) -> None:
        pass

    def train_dataloader(self):
        return DataLoader(CustomIterableDataset(self.train_data, self.train_mask, self.prop, self.batch_dim, self.shuffle_dims),
                          batch_size=1, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(CustomIterableDataset(self.val_data, self.val_mask, 1., self.batch_dim, self.shuffle_dims),
                          batch_size=1, pin_memory=True)

    def test_dataloader(self):
        if not self.test:
            raise ValueError("No test data")
        return DataLoader(CustomIterableDataset(self.test_data, self.test_mask, 1., self.batch_dim, self.shuffle_dims),
                          batch_size=1, pin_memory=True)

class MaskedData(L.LightningDataModule):
    def __init__(self, data: torch.Tensor, mask: torch.Tensor = None,
                 n_folds: int = 5, prop: float = 1.0, shuffle_dims = (0, 1),
                 device: str = None, test: bool = False):
        super().__init__()
        if mask is None:
            self.mask = torch.ones_like(data, dtype=torch.bool)
        else:
            self.mask = mask

        self.data = data
        self.n_folds = n_folds
        self.prop = prop
        self.test = test
        self.device = self.set_device(device)
        self.val_mask = torch.empty_like(self.mask, dtype=torch.bool)
        self.train_mask = torch.empty_like(self.mask, dtype=torch.bool)
        self.test_mask = torch.empty_like(self.mask, dtype=torch.bool)
        self.dims = self.data.shape
        self.shuffle_dims = shuffle_dims if isinstance(shuffle_dims, tuple) else (shuffle_dims,)
        self._setup()

    def set_device(self, device):
        if device is not None:
            return device
        elif torch.cuda.is_available():
            return 'cuda'
        elif torch.xpu.is_available():
            return 'xpu'
        else:
            return 'cpu'

    def _setup(self):
        self.data = torch.as_tensor(self.data)
        self.mask = torch.as_tensor(self.mask)
        if self.data.device.type != self.device:
            self.data.pin_memory(self.device)
            self.mask.pin_memory(self.device)
        n_folds = self.n_folds
        train_dim = tuple(1 if i in self.shuffle_dims else 10 for i in range(self.data.ndim))
        test_dim = tuple(1 if i in self.shuffle_dims else 5 for i in range(self.data.ndim))
        if self.test:
            train_mask1, test_mask = block_mask(dimensions=self.mask.shape,
                                                train_blocks_dimensions=train_dim,
                                                test_blocks_dimensions=test_dim,
                                                fraction_test=1/n_folds,
                                                device=self.data.device.type)
            self.test_mask = test_mask & self.mask
            n_folds -= 1
        else:
            train_mask1 = torch.ones_like(self.mask, dtype=torch.bool)
            test_mask = torch.zeros_like(self.mask, dtype=torch.bool)

        train_mask2, val_mask = block_mask(dimensions=self.mask.shape,
                                           train_blocks_dimensions=train_dim,
                                           test_blocks_dimensions=test_dim,
                                           fraction_test=1/n_folds,
                                           device=self.data.device.type)
        self.train_mask = (train_mask1 & train_mask2) & self.mask
        self.val_mask = (val_mask & ~test_mask) & self.mask

    def setup(self, stage: str) -> None:
        pass

    def train_dataloader(self):
        return DataLoader(CustomIterableDataset(self.data, self.train_mask, self.prop, None, self.shuffle_dims),
                          batch_size=None, num_workers=0, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(CustomIterableDataset(self.data, self.val_mask, 1., None, self.shuffle_dims),
                          batch_size=None, num_workers=0, pin_memory=True)

    def test_dataloader(self):
        if not self.test:
            raise ValueError("No test data")
        return DataLoader(CustomIterableDataset(self.data, self.test_mask, 1., None, self.shuffle_dims),
                          batch_size=None, num_workers=0, pin_memory=True)

class CustomIterableDataset(IterableDataset):
    def __init__(self, data, mask, batch_prop=1.0, batch_dim=None, shuffle_dims=(0,)):
        assert data.shape == mask.shape, f"Data and mask must have the same shape, got {data.shape} and {mask.shape}"
        assert 0 < batch_prop <= 1.0, "batch_prop must be in (0, 1]"
        self.data = data
        self.mask = mask
        self.batch_prop = batch_prop
        self.batch_dim = batch_dim
        self.shuffle_dims = shuffle_dims
        self.gen = torch.Generator(device=self.data.device.type)

    def __iter__(self):
        if self.batch_prop < 1.0 and self.batch_dim is None:
            yield from self.__iter1()
        elif self.batch_prop < 1.0 and self.batch_dim is not None:
            yield from self.__iter2()
        elif self.batch_prop == 1.0 and self.batch_dim is None:
            yield from self.__iter3()
        elif self.batch_prop == 1.0 and self.batch_dim is not None:
            yield from self.__iter4()
        else:
            raise ValueError("Invalid combination of batch_prop and batch_dim")

    def __iter1(self):
        """Batch_prop < 1.0, batch_dim is None"""

        dist = torch.empty([self.data.shape[i] if i in self.shuffle_dims else 1 for i in range(self.data.ndim)],
                           dtype=torch.float16, device=self.data.device)
        while True:
            torch.nn.init.uniform_(dist, 0, 1)
            batch = dist < self.batch_prop
            mask_out = self.mask & batch

            if mask_out.any():
                yield self.data, mask_out

    def __iter2(self):
        """Batch_prop < 1.0, batch_dim is not None"""
        dist = torch.empty([self.data.shape[i] if i in self.shuffle_dims + (self.batch_dim,) else 1 for i in range(self.data.ndim)],
                           dtype=torch.float16, device=self.data.device)
        while True:
            for dim in self.shuffle_dims:
                self.shuffle(dim)
            torch.nn.init.uniform_(dist, 0, 1)
            batch = dist < self.batch_prop
            mask_out = self.mask & batch

            for i in range(self.data.shape[self.batch_dim]):
                idx = [slice(None) if j != self.batch_dim else i
                       for j in range(self.data.ndim)]
                mask_out_2 = mask_out[idx]
                if mask_out_2.any():
                    yield self.data[idx], mask_out_2

    def __iter3(self):
        """Batch_prop == 1.0, batch_dim is None"""
        while True:
            yield self.data, self.mask

    def __iter4(self):
        """Batch_prop == 1.0, batch_dim is not None"""

        while True:
            for dim in self.shuffle_dims:
                self.shuffle(dim)

            for i in range(self.data.shape[self.batch_dim]):
                idx = [slice(None) if j != self.batch_dim else i
                       for j in range(self.data.ndim)]
                mask_out = self.mask[idx]
                if mask_out.any():
                    yield self.data[idx], mask_out

    # def shuffle(self, dim: int):
    #     """Shuffle the data and mask along the specified dimensions."""
    #     perms = mult(self.data.shape[self.batch_dim],
    #                  (self.data.shape[dim], self.data.shape[self.batch_dim]), self.gen)
    #
    #     # Create an index tensor for advanced indexing
    #     idx = torch.arange(self.data.shape[dim], device=self.data.device).unsqueeze(-1).expand(-1, self.data.shape[
    #         self.batch_dim])
    #
    #     # Use advanced indexing to shuffle the data and mask
    #     self.data = self.data.index_select(dim, idx).gather(self.batch_dim, perms)
    #     self.mask = self.mask.index_select(dim, idx).gather(self.batch_dim, perms)
    def shuffle(self, dim: int):
        """Shuffle the data and mask along the specified dimensions."""
        shuffle_slice = [slice(None)] * self.data.ndim
        perms = mult(self.data.shape[self.batch_dim],
                     (self.data.shape[dim], self.data.shape[self.batch_dim]), self.gen)
        for i in range(self.data.shape[dim]):
            shuffle_slice[dim] = i
            perm_slice = shuffle_slice.copy()
            perm_slice[self.batch_dim] = perms[i]
            self.data[tuple(shuffle_slice)] = self.data[tuple(perm_slice)]
            self.mask[tuple(shuffle_slice)] = self.mask[tuple(perm_slice)]

def handle_data(data, mask=None, device=None):
    if mask is None:
        mask = torch.ones_like(data, dtype=torch.bool)
    if not torch.is_tensor(data):
        data = torch.as_tensor(data)
        mask = torch.as_tensor(mask)
    if data.device.type != device:
        data = data.to(device)
        mask = mask.to(device)
    return data, mask

def mult(pop_size, shape, generator, replacement=True):
    """Use torch.Tensor.multinomial to generate indices on a GPU tensor."""
    num_samples = 1
    for s in shape:
        num_samples *= s
    p = torch.ones(pop_size, device=generator.device) / pop_size
    out = p.multinomial(num_samples=num_samples, replacement=replacement, generator=generator)
    return out.reshape(shape)