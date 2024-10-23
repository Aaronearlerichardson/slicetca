import lightning as L
import torch
from slicetca.run.utils import block_mask
from torch.utils.data import DataLoader, IterableDataset


def _feed(data, mask, batch_dim=None, batch_prop=1.0):
    assert 0 < batch_prop <= 1.0, "batch_prop must be in (0, 1]"
    assert data.shape == mask.shape, f"Data and mask must have the same shape, got {data.shape} and {mask.shape}"
    dist = torch.empty_like(data)
    while True:
        if batch_prop < 1.0:
            torch.nn.init.uniform_(dist, 0, 1)
            batch = dist < batch_prop
            mask_out = mask & batch
        else:
            mask_out = mask

        if batch_dim is None:
            if mask_out.any():
                yield data, mask_out
        else:
            for i in range(data.shape[batch_dim]):
                idx = [slice(None) if j != batch_dim else i
                       for j in range(data.ndim)]
                if mask_out[idx].any():
                    yield data[idx], mask_out[idx]


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
                                              # Note that the blocks will be of size 2*train_blocks_dimensions + 1
                                              test_blocks_dimensions=(1, 1, 5),
                                              # Same, 2*test_blocks_dimensions + 1
                                              fraction_test=1/n_folds,
                                              device=self.data.device.type)
            self.test_mask = test_mask & self.mask
            n_folds -= 1
        else:
            train_mask1 = torch.ones_like(self.mask, dtype=torch.bool)
            test_mask = torch.zeros_like(self.mask, dtype=torch.bool)

        train_mask2, val_mask = block_mask(dimensions=self.mask.shape,
                                          train_blocks_dimensions=(1, 1, 10),
                                          # Note that the blocks will be of size 2*train_blocks_dimensions + 1
                                          test_blocks_dimensions=(1, 1, 5),
                                          # Same, 2*test_blocks_dimensions + 1
                                          fraction_test=1/n_folds,
                                          device=self.data.device.type)
        self.train_mask = (train_mask1 & train_mask2) & self.mask
        self.val_mask = (val_mask & ~test_mask) & self.mask
        self.dims = self.data.shape

    def train_dataloader(self):
        return DataLoader(CustomIterableDataset(self.data, self.train_mask, batch_prop=self.prop),
                          verbose=False)

    def val_dataloader(self):
        return DataLoader(CustomIterableDataset(self.data, self.val_mask, batch_prop=self.prop),
                          verbose=False)

    def test_dataloader(self):
        if not self.test:
            raise ValueError("No test data")
        return DataLoader(CustomIterableDataset(self.data, self.test_mask, batch_prop=self.prop),
                          verbose=False)


class CustomIterableDataset(IterableDataset):
    def __init__(self, data, mask, batch_dim=None, batch_prop=1.0):
        self.data = data
        self.mask = mask
        self.batch_dim = batch_dim
        self.batch_prop = batch_prop

    def __iter__(self):
        return _feed(self.data, self.mask, self.batch_dim, self.batch_prop)


