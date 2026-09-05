import math

import numpy as np
from PIL import Image
from numba import cuda

import utils
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

                seed = utils.combine_seeds((d_x, d_y))
                seed = utils.random(seed)

                r_x = d_x + seed % randomness - randomness//2
                r_y = d_y + (seed + r_x**2) % randomness - randomness//2

                c_dist = (x - r_x) ** 2 + (y - r_y) ** 2
                # c_dist = d_x + d_y
                if min_dist == -1 or c_dist < min_dist:
                    min_dist = c_dist
                    min_dist_id = seed % 256
        image[x, y] = min_dist_id


@gpu_program()
def perlin_gradient_noise(image: np.ndarray, vector_grid_resolution: float):
    x, y = cuda.grid(2)
    if x >= image.shape[0] or y >= image.shape[1]:
        return
    vector_cell_size = image.shape[0]/vector_grid_resolution, image.shape[1]/vector_grid_resolution
    closest_vector_grid_cell = round(x/vector_cell_size[0]), round(y/vector_cell_size[1])
    local_seed = utils.combine_seeds(closest_vector_grid_cell)
    sign_x = 1 if utils.random(local_seed * 997 + 126) % 2 else -1
    sign_y = 1 if utils.random(local_seed * 557 + 128) % 2 else -1
    v_x = 1 * (utils.random(local_seed * 41 + 12) % 999 + 1) * sign_x
    v_y = 1 * (utils.random(local_seed * 971 + 124) % 999 + 1) * sign_y
    d = (v_x**2 + v_y**2)**0.5
    v_x /= d
    v_y /= d
    dot_product = v_x*(x - closest_vector_grid_cell[0]*vector_cell_size[0])\
                  + v_y*(y - closest_vector_grid_cell[1]*vector_cell_size[1])
    image[x, y] += dot_product*5


@gpu_program()
def direct_perlin_noise(image: np.ndarray,
                        vector_grid_resolution: float,
                        seed: int,
                        strength: float,
                        bias: float,
                        position_x: float,
                        position_y: float):
    x, y = cuda.grid(2)
    if x >= image.shape[0] or y >= image.shape[1]:
        return
    vector_cell_size = image.shape[0]/vector_grid_resolution, image.shape[1]/vector_grid_resolution
    sum_x = 0.0

    image_x = x
    image_y = y

    x -= position_x
    y -= position_y
    for off_x in range(2):
        sum_y = 0.0
        vector_cell_x = int(x // vector_cell_size[0] + off_x)
        d_x = x/vector_cell_size[0] - vector_cell_x
        for off_y in range(2):
            vector_cell_y = int(y//vector_cell_size[1] + off_y)
            d_y = y/vector_cell_size[1] - vector_cell_y

            local_seed = utils.combine_seeds((seed, vector_cell_x, vector_cell_y))
            sign_x = 1 if utils.random(local_seed * 997 + 126) % 2 else -1
            sign_y = 1 if utils.random(local_seed * 557 + 128) % 2 else -1
            v_x = 1 * (utils.random(local_seed * 41 + 12) % 999 + 1) * sign_x
            v_y = 1 * (utils.random(local_seed * 971 + 124) % 999 + 1) * sign_y
            d = (v_x**2 + v_y**2)**0.5
            if d != 0:
                v_x /= d
                v_y /= d
            dot_product = v_x*d_x + v_y*d_y
            sum_y += dot_product*utils.smoothstep(1-abs(d_y))
        sum_x += sum_y*utils.smoothstep(1-abs(d_x))
    image[image_x, image_y] += sum_x*strength + bias


def perlin_noise(image: np.ndarray,
                 vector_grid_resolution: float,
                 strength: float = 1.0,
                 bias: float = 0.0,
                 position: tuple[float, float] = (0, 0),
                 seed=0):
    direct_perlin_noise(image, vector_grid_resolution, seed, strength, bias, position[0], position[1])


@gpu_program()
def direct_height_rings(image: np.ndarray, ring_distance: float, ring_width: float):
    x, y = cuda.grid(2)
    if x >= image.shape[0] or y >= image.shape[1]:
        return

    s = 0
    c = 0
    for off_x in range(-2, 3):
        for off_y in range(-2, 3):
            if x+off_x >= image.shape[0] or y+off_y >= image.shape[1]:
                continue
            c += 1
            s += image[x+off_x, y+off_y]
    avg = s/c
    ring = image[x, y] % ring_distance
    if ring < ring_width:
        ring -= 0.5*ring_width
        ring /= ring_width/2
        image[x, y] = avg - (avg - (255-avg))*utils.smoothstep(1-abs(ring))

    # image[x, y] = 255-avg if image[x, y] % 10 < 2 else image[x, y]


def height_rings(image: np.ndarray, ring_distance=10.0, ring_width=2.0):
    direct_height_rings(image, ring_distance, ring_width)


def generate_height_map(image: np.ndarray,
                        base_height=1.0,
                        seed: int = 0,
                        iterations=10,
                        contrast=0.0,
                        zoom=1.0,
                        position=(0.0, 0.0)):
    position = position[0]*zoom+image.shape[0]/2, position[1]*zoom+image.shape[1]/2,
    for i in range(iterations):
        layer = i+1
        bias = 0
        if layer == 1:
            bias = base_height/2
        perlin_noise(image,
                     2 ** layer / zoom,
                     base_height / (2**(layer**(1/(contrast+1)))),
                     bias,
                     position,
                     seed=seed)


if __name__ == '__main__':
    print('pre_start')
    arr = cuda.to_device(np.zeros((1000, 1000), dtype=float))
    print('start')
    # voronoi(arr, 20, 200)
    # voronoi(arr, 20, 20, limit_threads=256)
    # perlin_noise(arr, 20, 255*4)
    # generate_height_map(arr, 255*1.5, contrast=0.5)
    generate_height_map(arr, 255*1.5, contrast=0.0, iterations=6, zoom=1, position=(0, 0), seed=0)
    height_rings(arr)
    # perlin_gradient_noise(arr, 20)
    cuda.synchronize()
    print('end')
    arr_back = arr.copy_to_host()
    arr_back = np.abs(arr_back)
    print('post_end')
    img = Image.fromarray(arr_back)
    print('image_ready')
    img.rotate(90).show()


