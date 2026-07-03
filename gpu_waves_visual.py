import random
import time

import pygame
import numpy as np

from gpu_wave_simulation import WaveSimulation, Emitter

if __name__ == "__main__":

    sim = WaveSimulation(
        size=(700, 700),
        centimeters_count=200,
        speed=34300
        # speed=int(2.998e+10)
    )
    # sim.add_emitter(Emitter(250, 250, 5, 100, 0))

    sim.add_emitters(*[Emitter(random.randint(0, sim.heights.shape[0]),
                               random.randint(0, sim.heights.shape[1]),
                               random.uniform(0.1, 10),
                               random.uniform(1, 10),
                               0) for _ in range(5)])

    speed_up = 0.1

    screen = pygame.display.set_mode((700, 700), pygame.RESIZABLE)

    clock = pygame.time.Clock()
    clock.tick(60)

    t = time.time()
    while True:
        clock.tick(60)
        dt = time.time()-t
        t = time.time()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()
        sim.step(dt*speed_up)
        img = pygame.image.frombytes(sim.get_image(True).convert('RGB').tobytes(), sim.heights.shape, 'RGB')
        img = pygame.transform.scale(img, screen.get_size())
        screen.blit(img, (0, 0))

        # if random.random() < 0.01 and sim.emitters:
        #     to_delete = random.choice(sim.emitters)
        #     if -0.01 < np.sin(to_delete.offset + sim.simulation_time / to_delete.wavelength
        #                     * sim.centimeters_count * sim.speed) < 0.01:
        #         sim.remove_emitter(to_delete)
        #     else:
        #         to_delete.amplitude /= 10
        #         sim.bake_emitters()

        pygame.display.update()
        screen.fill(0)


