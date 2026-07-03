import math
import random
import time

import numpy as np
from PIL import Image

from numba import jit, vectorize, float64, cuda

from utils import gpu_program


@vectorize([float64(float64, float64, float64)])
def disk(x, y, r):
    return (x**2 + y**2) - r**2


@vectorize([float64(float64, float64, float64, float64)])
def circle(x, y, r, w):
    return max(disk(x, y, r), -disk(x, y, r-w))


@jit("f8[:,:](f8[:,:],f8,f8,f8,f8)")
def draw_circle(np_image: np.ndarray, off_x, off_y, r, w):
    for x in range(np_image.shape[0]):
        for y in range(np_image.shape[1]):
            np_image[x, y] = circle(x-off_x, y-off_y, r, w)
    return np_image


@jit("f8[:,:](f8[:,:],f8,f8,f8)")
def draw_disk(np_image: np.ndarray, off_x, off_y, r):
    return draw_circle(np_image, off_x, off_y, r, r)


@cuda.jit
def gpu_draw_circle(np_image: np.ndarray, off_x, off_y, r, w):
    x, y = cuda.grid(2)
    if np_image.shape[0] > x and np_image.shape[1] > y:
        outer = ((x-off_x)**2 + (y-off_y)**2) - r**2
        inner = ((x-off_x)**2 + (y-off_y)**2) - (r-w)**2
        if outer > -inner:
            np_image[x, y] = outer
        else:
            np_image[x, y] = -inner


@gpu_program(["f8[:,:],f8[:,:]"])
def gpu_draw_circles(np_image: np.ndarray, circles: np.ndarray):
    x, y = cuda.grid(2)
    for off_x, off_y, r, w in circles:
        if np_image.shape[0] > x and np_image.shape[1] > y:
            outer = ((x-off_x)**2 + (y-off_y)**2) - r**2
            inner = ((x-off_x)**2 + (y-off_y)**2) - (r-w)**2
            if outer > -inner:
                out = outer
            else:
                out = -inner
            if np_image[x, y] > out:
                np_image[x, y] = out


@jit("f8[:,:](f8[:,:],f8)")
def realize(np_image: np.ndarray, threshold):
    for x in range(np_image.shape[0]):
        for y in range(np_image.shape[1]):
            np_image[x, y] = np_image[x, y] < threshold
    return np_image


@jit("f8[:,:],f8[:,:],i8")
def blur_np_image(np_image: np.ndarray, out_image: np.ndarray, distance):
    for x in range(np_image.shape[0]):
        for y in range(np_image.shape[1]):
            s = 0
            c = 0
            for d_x in range(-distance, distance + 1):
                for d_y in range(-distance, distance + 1):
                    if 0 <= x + d_x < np_image.shape[0] and 0 <= y + d_y < np_image.shape[1]:
                        s += np_image[x + d_x, y + d_y]
                        c += 1
            out_image[x, y] = s / c


@gpu_program()
def gpu_blur_np_image(np_image: np.ndarray, out_image: np.ndarray, distance):
    x, y = cuda.grid(2)
    s = 0
    c = 0
    for d_x in range(-distance, distance + 1):
        for d_y in range(-distance, distance + 1):
            if 0 <= x + d_x < np_image.shape[0] and 0 <= y + d_y < np_image.shape[1]:
                s += np_image[x + d_x, y + d_y]
                c += 1
    out_image[x, y] = s / c


if __name__ == '__main__':
    print(cuda.detect())
    print('start')
    t = time.perf_counter()
    arr = np.ones((5000, 5000))
    # arr = draw_circle(arr, 2500, 2500, 2000, 100)
    test_circles = [[random.randrange(arr.shape[0]+1), random.randrange(arr.shape[1]+1),
                     random.randrange(50, arr.shape[0]//3+1),
                     random.randrange(1, 50)] for _ in range(100)]
    c_arr = np.array(test_circles, dtype=float)
    # threadsperblock = (16, 16)
    # blockspergrid_x = math.ceil(arr.shape[0] / threadsperblock[0])
    # blockspergrid_y = math.ceil(arr.shape[1] / threadsperblock[1])
    # blockspergrid = (blockspergrid_x, blockspergrid_y)
    # gpu_draw_circle[blockspergrid, threadsperblock](arr, 2500, 2500, 2000, 100)
    gpu_draw_circles(arr, c_arr, shape=arr.shape)
    arr = realize(arr, 1)*255
    arr = 255-arr

    out_arr = np.ndarray(arr.shape)
    gpu_blur_np_image(arr, out_arr, 5, shape=arr.shape)
    arr = out_arr

    t = time.perf_counter()-t
    print('end', t)
    image = Image.fromarray(arr)
    image.show()

