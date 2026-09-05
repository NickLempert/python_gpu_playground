import time

import numpy as np
import pygame.display
from PIL import Image
from numba import cuda

import textures
import utils


def main(screen=None):
    screen = screen or pygame.display.set_mode((500, 500), pygame.RESIZABLE)
    img_arr = cuda.to_device(np.zeros([max(screen.get_size())]*2))
    # img_arr = cuda.to_device(np.zeros(list(reversed(screen.get_size()))))
    position = [0, 0]
    zoom = 1.0
    t = time.time()
    while True:
        dt = time.time()-t
        t = time.time()
        print(1/dt if dt else 0)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()
            if event.type == pygame.WINDOWRESIZED:
                img_arr = cuda.to_device(np.zeros([min(1000, max(screen.get_size()))]*2))
                # img_arr = cuda.to_device(np.zeros(list(reversed(screen.get_size()))))

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

        utils.clear_gray_image(img_arr)
        textures.generate_height_map(img_arr,
                                     base_height=255*1.5,
                                     iterations=5,
                                     position=position,
                                     zoom=zoom,
                                     contrast=0.0)
        # textures.perlin_noise(img_arr, 20*zoom, strength=255*1.5, position=position)
        ring_distance = 10
        ring_width = 2
        if zoom <= 1:
            ring_width = 4
        if zoom < 0.5:
            ring_distance = 20
        if zoom < 0.25:
            ring_distance = 20
            ring_width = 8
        if zoom < 0.125:
            ring_distance = 80
            ring_width = 32
        textures.height_rings(img_arr, ring_distance, ring_width)

        img_arr_back = img_arr.copy_to_host()
        img_arr_back = np.abs(img_arr_back)

        # img = Image.fromarray(img_arr_back).rotate(-90).resize(screen.get_size()).convert('RGB')
        img = Image.fromarray(img_arr_back).convert('RGB')
        img = pygame.image.fromstring(img.tobytes(), img.size, img.mode)
        img = pygame.transform.rotate(img, -90)
        img = pygame.transform.scale(img, [max(screen.get_size())]*2)

        screen.blit(img, (0, 0))

        pygame.display.update()
        screen.fill(0)


if __name__ == '__main__':
    main()
