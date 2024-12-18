import os
import warnings

import h5py
import numpy as np
import torch


# Adapted from:
# https://github.com/NVlabs/edm2/blob/4bf8162f601bcc09472ce8a32dd0cbe8889dc8fc/torch_utils/misc.py#L122
class InfiniteSampler(torch.utils.data.Sampler):
    def __init__(
        self, dataset, rank=0, num_replicas=1, shuffle=True, seed=0, start_idx=0
    ):
        assert len(dataset) > 0
        assert num_replicas > 0
        assert 0 <= rank < num_replicas
        warnings.filterwarnings(
            "ignore", "`data_source` argument is not used and will be removed"
        )
        super().__init__(dataset)
        self.dataset_size = len(dataset)
        self.start_idx = start_idx + rank
        self.stride = num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        idx = self.start_idx
        epoch = None
        while True:
            if epoch != idx // self.dataset_size:
                epoch = idx // self.dataset_size
                order = np.arange(self.dataset_size)
                if self.shuffle:
                    np.random.RandomState(hash((self.seed, epoch)) % (1 << 31)).shuffle(
                        order
                    )
            yield int(order[idx % self.dataset_size])
            idx += self.stride


class AbstractSDADataset(torch.utils.data.Dataset):
    def __init__(self, window, flatten):
        self._window = window
        self._flatten = flatten

    @property
    def window(self):
        return self._window

    @property
    def flatten(self):
        return self._flatten

    def load_window(self, i: int):
        raise NotImplementedError


class COSMODataset(AbstractSDADataset):
    def __init__(
        self,
        data_path,
        num_features,
        spatial_res,
        cached=False,
        **super_kwargs,
    ):
        super().__init__(**super_kwargs)

        # SET UP DATASET
        self._data_path = os.path.abspath(data_path)
        self._h5_data_var = "x"
        assert os.path.exists(self._data_path)
        assert os.path.isfile(self._data_path)
        assert os.path.splitext(self._data_path)[-1] == ".h5"

        self._cached = cached
        if self._cached:
            with h5py.File(self._data_path, mode="r") as f:
                self.dataset = f[self._h5_data_var][:]  # [[N], L, C, H, W]
                self._h5_ds_shape = self.dataset.shape
        else:
            self.dataset = None
            with h5py.File(self._data_path, mode="r") as f:
                self._h5_ds_shape = f[self._h5_data_var].shape

        assert self._h5_ds_shape[-1] == self._h5_ds_shape[-2] == spatial_res
        self.spatial_res = spatial_res

        assert (
            num_features == self.num_features
        ), f"The number of specified features ({num_features}) does not match the number of features in the data ({self.num_features})."

    def __len__(self) -> int:
        return self._h5_ds_shape[0] - self.window + 1

    @property
    def raw_data_shape(self):
        return self._h5_ds_shape

    @property
    def raw_spatial_res(self):
        return self.spatial_res

    @property
    def num_features(self):
        return self._h5_ds_shape[-3]

    @property
    def data_path(self):
        return self._data_path

    def load_window(self, i: int):  # -> [L, C, H, W]
        if (not self._cached) and (self.dataset is None):
            self.dataset = h5py.File(self._data_path, "r")[self._h5_data_var]

        traj = torch.from_numpy(self.dataset[i : i + self.window, ...])
        return traj

    def __getitem__(self, i):
        x = self.load_window(i)  # [L, C, H, W]
        if self.flatten:
            return x.flatten(0, 1)  # [L * C, H, W]
        else:
            return x
