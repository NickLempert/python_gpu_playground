import numpy as np
from numba import cuda

from utils import gpu_program


@cuda.jit
def test_race(array: np.ndarray):
    pos = cuda.grid(1)
    for _ in range(1000-pos):
        pass
    if pos < array.size-1:
        array[pos] = array[pos+1]


if __name__ == '__main__':
    arr = np.array([i for i in range(1000)], dtype=float)
    gpu_program(dimensions=1, auto_compile=False)(test_race)(arr, shape=[arr.size])
    print(arr)
    s_dif = 0
    for i in range(1, len(arr)):
        s_dif += abs(arr[i]-arr[i-1])
    print(s_dif/999)

