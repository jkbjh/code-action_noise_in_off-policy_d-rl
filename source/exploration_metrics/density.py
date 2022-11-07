import numpy as np
from .scipy_knn import ScipyKNN

try:
    import knncuda
    import ctypes
    import warnings

    class CudaKNN(object):
        def __init__(self, data, max_allocation=1024 ** 3 * 6):
            data = np.asarray(data)
            self.data = data.astype(np.float32)
            self.max_allocation = max_allocation
            self.data_size = data.size * np.dtype(np.float32).itemsize
            self.batch_size = 2 ** int(
                np.log(self.max_allocation / self.data_size) / np.log(2)
            )
            assert self.batch_size > 16

        def query(self, query_data, k, p=2):
            query_data = query_data.astype(np.float32)
            query_data = np.asarray(query_data)
            assert p == 2
            idcs = []
            dists = []
            if len(query_data) // self.batch_size > 1:
                splits = np.array_split(query_data, len(query_data) // self.batch_size)
            else:
                splits = [query_data]
            for batch in splits:
                idx, dist = knncuda.cuda_global(self.data, batch, k)
                idcs.append(idx)
                dists.append(dist)
            return np.vstack(dists), np.vstack(idcs)

    CUDART_DLL = ctypes.cdll.LoadLibrary("libcudart.so")
    number_of_devices = ctypes.c_int()
    CUDART_DLL.cudaGetDeviceCount(ctypes.byref(number_of_devices))
    if number_of_devices.value > 0:
        KNN = CudaKNN
    else:
        warnings.warn("Trying to use CudaKNN but no CUDA DEVICES AVAILABLE.")
        raise ImportError(
            "Tried to use CudaKNN but there are no CUDA DEVICES AVAILABLE."
        )
except ImportError:
    KNN = ScipyKNN


def suggest_knn_order(n, d):
    return int(round(2 * n ** (1.0 / d)))
