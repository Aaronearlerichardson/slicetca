import lightning as L
import torch
from slicetca.run.utils import block_mask
from torch.utils.data import DataLoader, IterableDataset
import numpy as np


class BatchedData(L.LightningDataModule):
    def __init__(self, data: torch.Tensor, dim: int,
                 shuffle_dim = (), mask: torch.Tensor = None,
                 n_folds: int = 5, prop: float = 1.0, test: bool = False,
                 avg_batches: int = 1):
        super().__init__()

        self.batch_dim = dim
        self.shuffle_dims = shuffle_dim if isinstance(shuffle_dim, tuple) else (shuffle_dim,)
        self.n_folds = n_folds
        self.prop = prop
        self.test = test
        if not isinstance(avg_batches, int) or avg_batches < 1:
            raise ValueError("avg_batches must be an integer >= 1")
        self.avg_batches = avg_batches
        data, mask = handle_data(data, mask)
        self.dims = data.shape
        self.data = data
        if mask is None:
            mask = torch.ones_like(data, dtype=torch.bool)
        self.mask = mask

    def prepare_data(self) -> None:
        perm = torch.randperm(self.data.shape[self.batch_dim])
        idx = tuple(slice(None) if i != self.batch_dim else perm for i in range(self.data.ndim))
        self.data = self.data[idx]
        self.mask = self.mask[idx]

    def setup(self, stage: str) -> None:
        if stage == "fit":
            fold_size = self.data.shape[self.batch_dim] // self.n_folds
            data_slice = [slice(None)] * self.data.ndim
            data_slice[self.batch_dim] = slice(0, fold_size)
            self.val_data = self.data[tuple(data_slice)]
            self.val_mask = self.mask[tuple(data_slice)]
            if self.test:
                data_slice[self.batch_dim] = slice(fold_size, 2 * fold_size)
                self.test_data = self.data[tuple(data_slice)]
                self.test_mask = self.mask[tuple(data_slice)]
                data_slice[self.batch_dim] = slice(2 * fold_size, None)
                self.train_data = self.data[tuple(data_slice)]
                self.train_mask = self.mask[tuple(data_slice)]
            else:
                self.test_data = None
                self.test_mask = None
                data_slice[self.batch_dim] = slice(fold_size, None)
                self.train_data = self.data[tuple(data_slice)]
                self.train_mask = self.mask[tuple(data_slice)]

        if stage == "test":
            assert getattr(self, "test_data", None) is not None,\
                "Test data and mask must be set up before testing."

        if stage == "validate":
            assert hasattr(self, "val_data")

        if stage == "train":
            assert hasattr(self, "train_data")

    def train_dataloader(self):
        return DataLoader(CustomIterableDataset(self.train_data, self.train_mask, self.prop, self.batch_dim,
                                                self.shuffle_dims, self.avg_batches),)
                          # batch_size = self.train_data.shape[self.batch_dim])

    def val_dataloader(self):
        return DataLoader(CustomIterableDataset(self.val_data, self.val_mask, 1., self.batch_dim,
                                                self.shuffle_dims, self.avg_batches),)
                          # batch_size = self.val_data.shape[self.batch_dim])

    def test_dataloader(self):
        if not self.test:
            raise ValueError("No test data")
        return DataLoader(CustomIterableDataset(self.test_data, self.test_mask, 1., self.batch_dim,
                                                self.shuffle_dims, self.avg_batches),)
                          # batch_size = self.test_data.shape[self.batch_dim])

class MaskedData(L.LightningDataModule):
    def __init__(self, data: torch.Tensor, mask: torch.Tensor = None,
                 n_folds: int = 5, prop: float = 1.0, shuffle_dims = (0,),
                 test: bool = False, blocklength: int = 10):
        super().__init__()

        self.shuffle_dims = shuffle_dims if isinstance(shuffle_dims, tuple) else (shuffle_dims,)
        self.n_folds = n_folds
        self.prop = prop
        self.test = test
        data, mask = handle_data(data, mask)
        self.dims = data.shape
        self.data = data
        if mask is None:
            mask = torch.ones_like(data, dtype=torch.bool)
        self.mask = mask
        self.blocklength = blocklength

    def prepare_data(self) -> None:
        self.val_mask = torch.empty_like(self.mask, dtype=torch.bool)
        self.train_mask = torch.empty_like(self.mask, dtype=torch.bool)
        self.test_mask = torch.empty_like(self.mask, dtype=torch.bool)

    def setup(self, stage: str) -> None:
        if stage == "fit":
            n_folds = self.n_folds
            train_dim = tuple(1 if i in self.shuffle_dims else self.blocklength for i in range(self.data.ndim))
            test_dim = tuple(1 if i in self.shuffle_dims else self.blocklength // 2 for i in range(self.data.ndim))
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

        if stage == "test":
            assert getattr(self, "test_mask", None) is not None,\
                "Test mask must be set up before testing."

    def train_dataloader(self):
        return DataLoader(CustomIterableDataset(self.data, self.train_mask, self.prop, None, self.shuffle_dims),
                          batch_size=None)

    def val_dataloader(self):
        return DataLoader(CustomIterableDataset(self.data, self.val_mask, 1., None, self.shuffle_dims),
                          batch_size=None)

    def test_dataloader(self):
        if not self.test:
            raise ValueError("No test data")
        return DataLoader(CustomIterableDataset(self.data, self.test_mask, 1., None, self.shuffle_dims),
                          batch_size=None)

class CustomIterableDataset(IterableDataset):
    def __init__(self, data, mask, batch_prop=1.0, batch_dim=None, shuffle_dims=(), avg_batches: int = 1):
        assert data.shape == mask.shape, f"Data and mask must have the same shape, got {data.shape} and {mask.shape}"
        assert 0 < batch_prop <= 1.0, "batch_prop must be in (0, 1]"
        if not isinstance(avg_batches, int) or avg_batches < 1:
            raise ValueError("avg_batches must be an integer >= 1")
        self.data = data
        self.mask = mask
        self.batch_prop = batch_prop
        self.batch_dim = batch_dim
        self.shuffle_dims = shuffle_dims
        self.avg_batches = avg_batches

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

    def __len__(self):
        if self.batch_dim is None:
            return 1
        else:
            n = self.data.shape[self.batch_dim]
            return (n + self.avg_batches - 1) // self.avg_batches

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

            if self.avg_batches == 1:
                for i in range(self.data.shape[self.batch_dim]):
                    idx = [slice(None) if j != self.batch_dim else i
                           for j in range(self.data.ndim)]
                    mask_out_2 = mask_out[idx]
                    if mask_out_2.any():
                        yield self.data[idx], mask_out_2
            else:
                num_batches = self.data.shape[self.batch_dim]
                step = self.avg_batches
                for start in range(0, num_batches, step):
                    idx = [slice(None)] * self.data.ndim
                    idx[self.batch_dim] = slice(start, min(start + step, num_batches))
                    data_slice = self.data[tuple(idx)]
                    mask_slice = mask_out[tuple(idx)]
                    data_avg, mask_avg = self._masked_average(data_slice, mask_slice)
                    if mask_avg.any():
                        yield data_avg, mask_avg

    def __iter3(self):
        """Batch_prop == 1.0, batch_dim is None"""
        while True:
            yield self.data, self.mask

    def __iter4(self):
        """Batch_prop == 1.0, batch_dim is not None"""

        while True:
            for dim in self.shuffle_dims:
                self.shuffle(dim)

            if self.avg_batches == 1:
                for i in range(self.data.shape[self.batch_dim]):
                    idx = [slice(None) if j != self.batch_dim else i
                           for j in range(self.data.ndim)]
                    mask_out = self.mask[idx]
                    if mask_out.any():
                        yield self.data[idx], mask_out
            else:
                num_batches = self.data.shape[self.batch_dim]
                step = self.avg_batches
                for start in range(0, num_batches, step):
                    idx = [slice(None)] * self.data.ndim
                    idx[self.batch_dim] = slice(start, min(start + step, num_batches))
                    data_slice = self.data[tuple(idx)]
                    mask_slice = self.mask[tuple(idx)]
                    data_avg, mask_avg = self._masked_average(data_slice, mask_slice)
                    if mask_avg.any():
                        yield data_avg, mask_avg

    def _masked_average(self, X: torch.Tensor, mask: torch.Tensor):
        # Avoid NaNs from masked-out positions (NaN * 0 = NaN).
        X_mask = torch.where(mask, X, 0)
        X_sum = X_mask.sum(dim=self.batch_dim, keepdim=True)
        counts = mask.sum(dim=self.batch_dim, keepdim=True)
        return X_sum / counts, counts > 0


    def shuffle(self, dim: int):
        """Shuffle the data and mask along the specified dimensions."""
        shuffle_slice = [slice(None)] * self.data.ndim
        perms = mult(self.data.shape[self.batch_dim],
                     (self.data.shape[dim], self.data.shape[self.batch_dim]))
        for i in range(self.data.shape[dim]):
            shuffle_slice[dim] = i
            perm_slice = shuffle_slice.copy()
            perm_slice[self.batch_dim] = perms[i]
            self.data[tuple(shuffle_slice)] = self.data[tuple(perm_slice)]
            self.mask[tuple(shuffle_slice)] = self.mask[tuple(perm_slice)]

def handle_data(data, mask=None):
    if mask is None:
        mask = torch.ones_like(data, dtype=torch.bool)
    if not torch.is_tensor(data):
        data = torch.as_tensor(data)
        mask = torch.as_tensor(mask)
    return data, mask

def mult(pop_size, shape, replacement=True):
    """Use torch.Tensor.multinomial to generate indices on a GPU tensor."""
    num_samples = 1
    for s in shape:
        num_samples *= s
    p = torch.ones(pop_size) / pop_size
    out = p.multinomial(num_samples=num_samples, replacement=replacement)
    return out.reshape(shape)
