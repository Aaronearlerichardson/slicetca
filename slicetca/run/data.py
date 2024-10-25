import lightning as L
import torch
from slicetca.run.utils import block_mask
from torch.utils.data import DataLoader, IterableDataset


class Data(L.LightningDataModule):
    def __init__(self, data: torch.Tensor, mask: torch.Tensor = None,
                 n_folds: int = 5, prop: float = 1.0, test: bool = False):
        super().__init__()
        if mask is None:
            self.mask = torch.ones_like(data, dtype=torch.bool)
        else:
            self.mask = mask

        self.data = data
        self.batch_size = 1
        self.n_folds = n_folds
        self.prop = prop
        self.test = test

    def setup(self, stage: str = None):
        self.data = torch.as_tensor(self.data)
        self.mask = torch.as_tensor(self.mask)
        n_folds = self.n_folds
        if self.test:
            train_mask1, test_mask = block_mask(dimensions=self.mask.shape,
                                                train_blocks_dimensions=(1, 1, 10),
                                                test_blocks_dimensions=(1, 1, 5),
                                                fraction_test=1/n_folds,
                                                device=self.data.device.type)
            self.test_mask = test_mask & self.mask
            n_folds -= 1
        else:
            train_mask1 = torch.ones_like(self.mask, dtype=torch.bool)
            test_mask = torch.zeros_like(self.mask, dtype=torch.bool)

        train_mask2, val_mask = block_mask(dimensions=self.mask.shape,
                                           train_blocks_dimensions=(1, 1, 10),
                                           test_blocks_dimensions=(1, 1, 5),
                                           fraction_test=1/n_folds,
                                           device=self.data.device.type)
        self.train_mask = (train_mask1 & train_mask2) & self.mask
        self.val_mask = (val_mask & ~test_mask) & self.mask
        self.dims = self.data.shape

    def train_dataloader(self):
        return DataLoader(CustomIterableDataset(self.data, self.train_mask, batch_prop=self.prop),
                          batch_size=None, num_workers=0, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(CustomIterableDataset(self.data, self.val_mask, batch_prop=self.prop),
                          batch_size=None, num_workers=0, pin_memory=True)

    def test_dataloader(self):
        if not self.test:
            raise ValueError("No test data")
        return DataLoader(CustomIterableDataset(self.data, self.test_mask, batch_prop=self.prop),
                          batch_size=None, num_workers=0, pin_memory=True)

class CustomIterableDataset(IterableDataset):
    def __init__(self, data, mask, batch_prop=1.0, batch_dim=None):
        assert data.shape == mask.shape, f"Data and mask must have the same shape, got {data.shape} and {mask.shape}"
        assert 0 < batch_prop <= 1.0, "batch_prop must be in (0, 1]"
        self.data = data
        self.mask = mask
        self.batch_prop = batch_prop
        self.batch_dim = batch_dim

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
        dist = torch.empty_like(self.data, dtype=torch.float16)
        while True:
            torch.nn.init.uniform_(dist, 0, 1)
            batch = dist < self.batch_prop
            mask_out = self.mask & batch

            if mask_out.any():
                yield self.data, mask_out

    def __iter2(self):
        """Batch_prop < 1.0, batch_dim is not None"""
        dist = torch.empty_like(self.data, dtype=torch.float16)
        while True:
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

            for i in range(self.data.shape[self.batch_dim]):
                idx = [slice(None) if j != self.batch_dim else i
                       for j in range(self.data.ndim)]
                mask_out = self.mask[idx]
                if mask_out.any():
                    yield self.data[idx], mask_out
