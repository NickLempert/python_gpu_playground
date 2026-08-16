import math
import random
import time

import numba
import numpy as np
import pygame.display
from numba import cuda

from NeuralNetwork import erase
from utils import polar_to_cartesian, gpu_program


def create_particles(number: int, mass: float = 1.0, max_distance: float = 500, rotation_speed=1) -> np.ndarray:
    particles = []
    for _ in range(number):
        angle, distance = random.uniform(0, math.pi*2), random.uniform(0, max_distance)
        pos = polar_to_cartesian(angle, distance)
        velocity = polar_to_cartesian(angle+math.pi/2, rotation_speed/distance**2)
        particles.append(np.array([*pos, mass, *velocity], dtype=np.float32))
    return np.array(particles)


def calculate_step_helper(func: callable):
    return lambda particles, particles_out, g, dt: func(particles, particles_out, g, dt, shape=[len(particles)]*2)


@calculate_step_helper
@gpu_program(['(float32[:,:],float32[:,:],float32,float32)'], dimensions=2)
def calculate_step(particles: np.ndarray, particles_out: np.ndarray, dt: float, g: float):
    current, other = cuda.grid(2)
    if current == other:
        if abs(particles[current, 3]) > 1000:
            cuda.atomic.add(particles_out, (current, 3), 1000)
        else:
            cuda.atomic.add(particles_out, (current, 3), 1000 if particles[current, 3] > 0 else -1000)
        if abs(particles[current, 4]) > 1000:
            cuda.atomic.add(particles_out, (current, 4), 1000 if particles[current, 4] > 0 else -1000)
        else:
            cuda.atomic.add(particles_out, (current, 4), particles[current, 4])
        cuda.atomic.add(particles_out, (current, 2), particles[current, 2])

        particles_out[current, 0] = particles[current, 0] + particles[current, 3]*dt/10
        particles_out[current, 1] = particles[current, 1] + particles[current, 4]*dt/10
        return
    if current < particles.size and other < particles.size:
        distance = ((particles[current, 0] - particles[other, 0])**2 +
                    (particles[current, 1] - particles[other, 1])**2)**0.5
        if distance <= (particles[current, 2]/math.pi)**0.5 + (particles[other, 2]/math.pi)**0.5:
            if particles[current, 2] < particles[other, 2] or \
                    (particles[current, 2] == particles[other, 2]
                     and particles[current, 0] < particles[other, 0]):
                cuda.atomic.add(particles_out, (other, 2), particles[current, 2])
                cuda.atomic.sub(particles_out, (current, 2), 100000000)
            return
        direction = np.atan2(particles[current, 1]-particles[other, 1], particles[current, 0]-particles[other, 0])
        force = g*particles[other, 2]/(distance**2)*dt
        cuda.atomic.add(particles_out, (current, 3), math.cos(direction)*force)
        cuda.atomic.add(particles_out, (current, 4), math.sin(direction)*force)


def visual_main(screen=None):
    screen = screen or pygame.display.set_mode((500, 500), pygame.RESIZABLE)
    particles = cuda.to_device(create_particles(1000, 1, 500, 0))
    clock = pygame.time.Clock()
    last_update = time.time()
    while True:
        dt = time.time()-last_update
        last_update = time.time()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return

        host_particles = particles.copy_to_host()
        print(len(host_particles))
        particles_to_remove = []
        # print(sorted(host_particles, key=lambda v: -v[2])[:10])
        ind = -1
        for x, y, m, vx, vy in host_particles[:]:
            ind += 1
            if m <= 0.0001:
                particles_to_remove.append(ind)
                continue
            if -100000 < x < 100000 and -100000 < y < 100000:
                pygame.draw.circle(screen,
                                   (255, 0, 0),
                                   ((int(x)+screen.get_width()//2), (int(y)+screen.get_height())//2),
                                   math.ceil((m/math.pi)**0.5))
        if particles_to_remove:
            new_particles = list(host_particles)
            for offset, ind in enumerate(particles_to_remove):
                del new_particles[ind-offset]
            particles = cuda.to_device(np.array(new_particles))

        particles_out = cuda.device_array_like(particles)
        particles_out[:, :] = 0

        calculate_step(particles, particles_out, dt, 5.0)
        cuda.synchronize()
        particles = particles_out

        pygame.display.update()
        clock.tick(60)
        screen.fill(0)


if __name__ == '__main__':
    visual_main()

