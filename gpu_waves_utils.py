import math
import random
import time

import numpy as np
from PIL import Image
from numba import cuda

from utils import gpu_program


@gpu_program(["f8[:,:],f8[:,:],f8"])
def gpu_waves_step_heights(heights: np.ndarray, velocities: np.ndarray, dt: float):
    x, y = cuda.grid(2)
    if x < heights.shape[0] and y < heights.shape[1]:
        heights[x, y] += velocities[x, y]*dt


@gpu_program(["f8[:,:],f8[:,:],f8"])
def gpu_waves_step_velocities(heights: np.ndarray, velocities: np.ndarray, dt: float):
    x, y = cuda.grid(2)
    if x < heights.shape[0] and y < heights.shape[1]:
        sum_heights = 0
        count_heights = 0
        for d_x in range(-1, 2):
            for d_y in range(-1, 2):
                if d_x != 0 or d_y != 0:
                    n_x, n_y = x+d_x, y+d_y
                    if 0 <= n_x < heights.shape[0] and 0 <= n_y < heights.shape[1]:
                        sum_heights += heights[n_x, n_y]/(d_x**2+d_y**2)
                        count_heights += 1/(d_x**2+d_y**2)
        target_height = sum_heights/count_heights
        velocities[x, y] += (target_height-heights[x, y])*dt


@gpu_program(["f8[:,:],f8[:,:],f8"])
def full_gpu_step(heights: np.ndarray, velocities: np.ndarray, dt: float):
    x, y = cuda.grid(2)
    if x < heights.shape[0] and y < heights.shape[1]:
        sum_heights = 0
        count_heights = 0
        for d_x in range(-1, 2):
            for d_y in range(-1, 2):
                if d_x != 0 or d_y != 0:
                    n_x, n_y = x+d_x, y+d_y
                    if 0 <= n_x < heights.shape[0] and 0 <= n_y < heights.shape[1]:
                        sum_heights += heights[n_x, n_y]/(d_x**2+d_y**2)
                        count_heights += 1/(d_x**2+d_y**2)
        target_height = sum_heights/count_heights
        velocities[x, y] += (target_height-heights[x, y])*dt

        heights[x, y] += velocities[x, y]*dt


@gpu_program(["f8[:,:],f8[:,:],f8,f8,f8"], default_shape_source=0)
def gpu_waves_step_emitters(heights: np.ndarray,
                            emitters: np.ndarray,
                            simulation_time: float,
                            pixels_to_centimeter: float,
                            speed: float):
    x, y = cuda.grid(2)
    for e_x, e_y, wavelength, amplitude, offset in emitters:
        if e_x == x and e_y == y:
            heights[x, y] = np.cos(offset + simulation_time/wavelength*pixels_to_centimeter*speed)*amplitude/wavelength


if __name__ == '__main__':
    h = np.zeros((500, 500))
    v = np.zeros(h.shape)
    e = np.array([[250, 250, 5, 100, 0]], dtype=float)
    # e = np.array([[random.randint(0, h.shape[0]),
    #                random.randint(0, h.shape[1]),
    #                random.uniform(1, 10),
    #                random.uniform(1, 100),
    #                random.uniform(0, 100)] for _ in range(5)], dtype=float)

    d_h = cuda.to_device(h)
    d_v = cuda.to_device(v)
    d_e = cuda.to_device(e)

    box_size_centimeters = 200

    length_seconds = 20-1
    quality = 10
    additional_speed = 34300//quality//box_size_centimeters
    print(additional_speed)
    print(min(h.shape)/box_size_centimeters)

    t = time.perf_counter()
    for progress in range(length_seconds*quality):
        gpu_waves_step_emitters(d_h, d_e, progress/quality, min(h.shape)/box_size_centimeters, 34300)
        for _ in range(additional_speed):
            gpu_waves_step_heights(d_h, d_v, 1/quality)
            gpu_waves_step_velocities(d_h, d_v, 1/quality)
        # full_gpu_step(d_h, d_v, 1/quality, shape=h.shape)

    t2 = time.perf_counter()
    print(t2-t)

    h = d_h.copy_to_host()

    image = Image.fromarray(h*255)
    image.show()
    image = Image.fromarray(-h*255)
    image.show()

