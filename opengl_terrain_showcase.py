import threading
import time

import numpy as np
import pygame.display
pygame.init()
from PIL import Image
from numba import cuda

import TerrainShader


def main(screen=None):
    screen = screen or pygame.display.set_mode((700, 700), pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE)
    ctx = TerrainShader.init()
    clock = pygame.time.Clock()
    terrain_shader = TerrainShader.TerrainShader()
    position = [0, 0]
    zoom = 1.0
    t = time.time()
    previous_mouse_pos = [0, 0]
    displacement = [0, 0]
    resolution = screen.get_size()
    while True:
        dt = time.time()-t
        t = time.time()
        if dt != 0:
            print(1/dt)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()
            if event.type == pygame.WINDOWRESIZED:
                print(event)
                resolution = event.x, event.y
            if event.type == pygame.MOUSEWHEEL:
                zoom *= 1 + event.y/10

        if pygame.mouse.get_pressed()[1]:
            displacement[0] = -(previous_mouse_pos[0]-pygame.mouse.get_pos()[0])/zoom
            displacement[1] = (previous_mouse_pos[1]-pygame.mouse.get_pos()[1])/zoom
        else:
            position[0] += displacement[0]
            position[1] += displacement[1]
            displacement = [0, 0]
            previous_mouse_pos = pygame.mouse.get_pos()

        if pygame.key.get_pressed()[pygame.K_w]:
            position[1] += dt*100
        if pygame.key.get_pressed()[pygame.K_s]:
            position[1] -= dt*100
        if pygame.key.get_pressed()[pygame.K_d]:
            position[0] += dt*100
        if pygame.key.get_pressed()[pygame.K_a]:
            position[0] -= dt*100
        if pygame.key.get_pressed()[pygame.K_EQUALS]:
            zoom *= (1+dt)
        if pygame.key.get_pressed()[pygame.K_MINUS]:
            zoom /= (1+dt)
        print(resolution)
        terrain_shader.put_variables(zoom=zoom,
                                     position=(position[0]+displacement[0],  position[1]+displacement[1]),
                                     resolution=resolution)

        terrain_shader.render()

        pygame.display.flip()
        # screen.fill(0)
        # ctx.clear(0, 0, 0, 1)
        clock.tick(60)


if __name__ == '__main__':
    main()
