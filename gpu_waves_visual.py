import random
import time

import pygame
import numpy as np

from gpu_wave_simulation import WaveSimulation, Emitter

if __name__ == "__main__":

    sim = WaveSimulation(
        size=(900, 900),
        centimeters_count=450,
        speed=34300,
        # speed=int(2.998e+10),
        border_fade=150
    )
    # sim.update_passability(1)
    # sim.update_passability(0.1, 400, 450, 600, 600)

    # sim.add_emitter(Emitter(250+sim.border_fade, 250+sim.border_fade, 5, 10, 0))

    # sim.add_emitters(Emitter(250+sim.border_fade, 250+sim.border_fade, 5, 10, 0),
    #                  Emitter(251+sim.border_fade, 250+sim.border_fade, 5, 10, 1))

    sim.add_emitters(*[Emitter(random.randint(sim.border_fade, sim.heights.shape[0]-sim.border_fade),
                               random.randint(sim.border_fade, sim.heights.shape[1]-sim.border_fade),
                               random.uniform(0.1, 20),
                               random.uniform(1, 10),
                               0) for _ in range(5)])

    # count = 500
    # distance = 300
    # sim.add_emitters(*[Emitter(100 + sim.border_fade,
    #                            250 + distance/count * y + sim.border_fade,
    #                            2,
    #                            0.1,
    #                            0) for y in range(count)])

    speed_up = 1

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
        img = pygame.transform.scale(img, (screen.get_width()+sim.border_fade*2, screen.get_height()+sim.border_fade*2))
        screen.blit(img, (-sim.border_fade, -sim.border_fade))

        # for emitter in sim.emitters:
        #     emitter.amplitude /= 1 + dt*speed_up
        # sim.bake_emitters()

        # if random.random() < 0.01 and sim.emitters:
        #     to_delete = random.choice(sim.emitters)
        #     if -0.01 < np.sin(to_delete.offset + sim.simulation_time / to_delete.wavelength
        #                     * sim.centimeters_count * sim.speed) < 0.01:
        #         sim.remove_emitter(to_delete)
        #     else:
        #         to_delete.amplitude /= 2
        #         sim.bake_emitters()

        pygame.display.update()
        screen.fill(0)


