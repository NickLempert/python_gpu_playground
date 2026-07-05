import math
import random
import time

import numpy as np
from PIL import Image
from numba import cuda

from gpu_waves_utils import gpu_waves_step_heights, gpu_waves_step_velocities, gpu_waves_step_emitters


class Emitter:
    def __init__(self, x, y, wavelength=5.0, amplitude=10.0, offset=0.0):
        self.x = x
        self.y = y
        self.wavelength = wavelength
        self.amplitude = amplitude
        self.offset = offset

    def to_list(self):
        return [self.x, self.y, self.wavelength, self.amplitude, self.offset]


class WaveSimulation:
    def __init__(self,
                 size=(500, 500),
                 centimeters_count=200,
                 emitters: list[Emitter] = None,
                 speed=34300,
                 border_fade=0):
        self.speed = speed
        self.centimeters_count = centimeters_count
        self.border_fade = border_fade

        self.heights = cuda.to_device(np.zeros(size))
        self.velocities = cuda.to_device(np.zeros(size))
        self.emitters = emitters or []
        self.steps = 0
        self.simulation_time = 0

        self.baked_emitters = None
        self.bake_emitters()

        self.uncomputed_time = 0

    def bake_emitters(self):
        self.baked_emitters = None
        if self.emitters:
            self.baked_emitters = cuda.to_device(np.array(list(map(Emitter.to_list, self.emitters)), dtype=float))

    def add_emitter(self, emitter: Emitter):
        self.emitters.append(emitter)
        self.bake_emitters()

    def add_emitters(self, *emitters: Emitter):
        self.emitters.extend(emitters)
        self.bake_emitters()

    def remove_emitter(self, emitter: Emitter):
        self.emitters.remove(emitter)
        self.bake_emitters()

    def get_pixels_per_centimeter(self):
        return min(self.heights.shape)/self.centimeters_count

    def step(self, dt: float):
        dt /= 1000
        self.steps += 1
        iterations = self.speed * self.get_pixels_per_centimeter() * dt + self.uncomputed_time
        self.uncomputed_time = iterations - int(iterations)
        for _ in range(int(iterations)):
            self.simulation_time += 1 / self.speed / self.get_pixels_per_centimeter()
            if self.emitters:
                gpu_waves_step_emitters(self.heights,
                                        self.baked_emitters,
                                        self.simulation_time,
                                        self.get_pixels_per_centimeter(),
                                        self.speed)
            gpu_waves_step_heights(self.heights, self.velocities, 1, self.border_fade)
            gpu_waves_step_velocities(self.heights, self.velocities, 1)

    def get_image(self, black_and_white=False):
        heights = self.heights.copy_to_host()
        if black_and_white:
            return Image.fromarray((heights*2000))
        velocities = abs(self.velocities.copy_to_host())

        # image = Image.fromarray(h * 255)
        # image.show()
        # image = Image.fromarray(-h * 255)
        # image.show()
        negative = heights * 100 * -1
        positive = heights * 100
        velocities = velocities * 255
        image = Image.fromarray(np.dstack((negative, positive, velocities)).astype(np.uint8), 'RGB')
        return image


if __name__ == "__main__":

    sim = WaveSimulation(
        size=(1000, 1000),
        centimeters_count=200,
        speed=34300
        # speed=int(2.998e+10)
    )
    # sim.add_emitter(Emitter(250, 250, 5, 100, 0))
    sim.add_emitters(*[Emitter(random.randint(0, sim.heights.shape[0]),
                               random.randint(0, sim.heights.shape[1]),
                               random.uniform(0.5, 100),
                               random.uniform(1, 100),
                               0) for _ in range(10)])

    length_milli_seconds = 2.5

    t = time.perf_counter()
    for progress in range(int(length_milli_seconds)):
        sim.step(1)
    sim.step(length_milli_seconds-int(length_milli_seconds))
    t2 = time.perf_counter()
    print(t2-t)

    sim.get_image().show()
