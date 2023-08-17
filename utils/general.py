import time
import math
import torch
import numpy as np
from typing import Union
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


def try_gpu(index_cuda: int = None) -> torch.device:
    """Return available device, gpu or cpu.

    Args:
        index_cuda: index of gpu

    Returns:
        Return available device, gpu or cpu.
    """

    "Return a available gpu or cpu."
    devices = [torch.device('cpu')]
    if torch.cuda.is_available():
        # Return the gpu of the specified index first
        if index_cuda is not None and torch.cuda.device_count() > index_cuda:
            return torch.device(f'cuda:{index_cuda}')
        for i in range(torch.cuda.device_count()):
            devices.append(torch.device(f'cuda:{i}'))
    return devices[-1]


def create_data_iterator(X: Union[torch.Tensor, np.ndarray, list],
                         y: Union[torch.Tensor, np.ndarray, list],
                         sample_weight: dict = None,
                         batch_size: int = 1,
                         shuffle: bool = False,
                         data_type: torch.dtype=torch.float32,
                         ) -> DataLoader:
    """Create iterator from gave feature set and label set.

    Considering balance of samples, if gave weight of class, only 90% samples will be
    sampled, otherwise 100%.

    Args:
        X: feature matrix whose shape should is n*m represent n samples and
          every sample has m features.
        y: label vector whose shape should is m*1.
        sample_weight: each class's weight, e.g. {0: 0.45, 1: 0.55},
          sample more important class first.
        batch_size: samples' number that train in network each time.
        shuffle: shuffle the order of sample training.
        data_type: X and y default data_access type.

    Returns:
        a DataLoader instance contain features and labels tuple
    """

    if sample_weight is not None and not isinstance(sample_weight, dict):
        raise TypeError(f'expect dict, but get {type(sample_weight)}')
    if type(X) == list or type(X) == np.ndarray:
        X = torch.tensor(X, dtype=data_type)
    elif not isinstance(X, torch.Tensor):
        raise TypeError(f'expect tensor, numpy or list, but get {type(X)}')
    if type(y) == list or type(y) == np.ndarray:
        y = torch.tensor(y, dtype=data_type)
    elif not isinstance(y, torch.Tensor):
        raise TypeError(f'expect tensor, numpy or list, but get {type(y)}')
    assert (isinstance(X, torch.Tensor) and isinstance(y, torch.Tensor))

    if X.dtype is not data_type:
        if data_type is torch.float32:
            X = X.float()
        elif data_type is torch.float64:
            X = X.double()
        else:
            raise TypeError(f'data_access type of X should set {data_type}')
    if y.dtype is not data_type:
        if data_type is torch.float32:
            y = y.float()
        elif data_type is torch.float64:
            y = y.double()
        else:
            raise TypeError(f'data_access type of y should set {data_type}')

    y.reshape((X.shape[0], 1))
    data_set = TensorDataset(X, y)
    sampler = None
    if sample_weight is not None:
        weights = [sample_weight[y.item()] for x, y in data_set]
        num_samples = math.floor(X.shape[0] * 0.9)
        sampler = WeightedRandomSampler(weights, num_samples,
                                        replacement=False)
    return DataLoader(data_set, batch_size, shuffle, sampler)


class Timer:
    """Record multiple running times."""
    def __init__(self):
        """Defined in :numref:`subsec_linear_model`"""
        self.times = []
        self.start()

    def start(self):
        """Start the timer."""
        self.tik = time.time()

    def stop(self):
        """Stop the timer and record the time in a list."""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """Return the average time."""
        return sum(self.times) / len(self.times)

    def sum(self):
        """Return the sum of time."""
        return sum(self.times)

    def cumsum(self):
        """Return the accumulated time."""
        return np.array(self.times).cumsum().tolist()

    def sum_human_read(self):
        """Return total time in human-readable form"""
        sum_time = self.sum()
        sec = sum_time % 60
        hour = sum_time // 3600
        min = (sum_time - (sec - hour * 3600)) / 60
        time_string = f'{hour} hour' if hour > 0 else '' + \
                      f'{min} min' if min > 0 else '' + \
                      f'{sec:.2f} sec' if sec > 0 else ''
        return time_string


class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        """Defined in :numref:`sec_softmax_scratch`"""
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]