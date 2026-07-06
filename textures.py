import numpy as np
from PIL import Image
from numba import cuda
from numba.cuda.random import xoroshiro128p_uniform_float32

from utils import gpu_program


@gpu_program()
def voronoi(image: np.ndarray, grid_size: int, randomness: int):
    x, y = cuda.grid(2)

    if x < image.shape[0] and y < image.shape[1]:
        p_x = x // grid_size * grid_size
        p_y = y // grid_size * grid_size

        min_dist = -1
        min_dist_id = -1

        expand_search = randomness//grid_size

        for d_x in range(-expand_search, 2 + expand_search):
            d_x = p_x + grid_size*d_x
            for d_y in range(-expand_search, 2 + expand_search):
                d_y = p_y + grid_size*d_y

                seed = (d_x * 3266489917) + (d_y**2 * 3266489917)
                seed ^= seed >> 15
                seed *= 2246822519
                seed ^= seed >> 13

                r_x = d_x + seed % randomness - randomness//2
                r_y = d_y + (seed + r_x**2) % randomness - randomness//2

                c_dist = (x - r_x) ** 2 + (y - r_y) ** 2
                # c_dist = d_x + d_y
                if min_dist == -1 or c_dist < min_dist:
                    min_dist = c_dist
                    min_dist_id = seed % 256
        image[x, y] = min_dist_id


if __name__ == '__main__':
    print('pre_start')
    arr = cuda.to_device(np.zeros((20000, 20000), dtype=float))
    print('start')
    # voronoi(arr, 20, 200)
    voronoi(arr, 20, 20)
    cuda.synchronize()
    print('end')
    arr_back = arr.copy_to_host()
    print('post_end')
    img = Image.fromarray(arr_back)
    print('image_ready')
    img.show()


