import time

from numba import cuda
import numpy as np

from utils import gpu_program


@gpu_program(['void(int64[:])'], dimensions=1)
def get_primes(output: np.array):
    pos = cuda.grid(1)
    if pos < output.size:
        output[pos] = 1
        a = pos//2
        while a > 1:
            if pos % a == 0:
                output[pos] = 0
                a = 0
            a -= 1


@gpu_program(['void(int64, int64[:])'], dimensions=1)
def is_prime(inp: int, output: np.array):
    pos = cuda.grid(1)
    if 1 < pos < inp and inp % pos == 0:
        output[0] = 0


if __name__ == '__main__':
    # arr = np.array([i+1 for i in range(10000000)], dtype=int)
    # get_primes(arr, shape=[arr.size])
    # for i in range(len(arr)):
    #     if arr[i]:
    #         print(i)
    start_at = 100000000000000000000
    search_count = 10000
    for i in range(start_at, start_at+search_count):
        arr = np.array([1])
        is_prime(i, arr, shape=[max(1000, int(i ** 0.5))])
        if arr[0]:
            print(i)

