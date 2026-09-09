import time

import numpy as np
import pygame.display
from PIL import Image
from numba import cuda

import textures
import utils


def main(screen=None):
    screen = screen or pygame.display.set_mode((700, 700), pygame.RESIZABLE)
    clock = pygame.time.Clock()
    img_arr = cuda.to_device(np.zeros([max(screen.get_size())]*2))
    # img_arr = cuda.to_device(np.zeros(list(reversed(screen.get_size()))))
    position = [0, 0]
    zoom = 1.0
    t = time.time()
    previous_mouse_pos = [0, 0]
    displacement = [0, 0]
    change_detected = True
    while True:
        dt = time.time()-t
        t = time.time()
        if dt != 0:
            print(1/dt)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit()
            if event.type == pygame.WINDOWRESIZED:
                change_detected = True
                # img_arr = cuda.to_device(np.zeros([min(1000, max(screen.get_size()))]*2))
                size = list(reversed(screen.get_size()))
                max_res = 1500
                if max(size) > max_res:
                    size[0] *= max_res/max(size)
                    size[1] *= max_res/max(size)
                size = list(map(int, size))
                img_arr = cuda.to_device(np.zeros(size))
            if event.type == pygame.MOUSEWHEEL:
                change_detected = True
                zoom *= 1 + event.y/10

        if pygame.mouse.get_pressed()[1]:
            displacement[0] = (previous_mouse_pos[0]-pygame.mouse.get_pos()[0])/zoom
            displacement[1] = -(previous_mouse_pos[1]-pygame.mouse.get_pos()[1])/zoom
            change_detected = True
        else:
            position[0] += displacement[0]
            position[1] += displacement[1]
            displacement = [0, 0]
            previous_mouse_pos = pygame.mouse.get_pos()

        if pygame.key.get_pressed()[pygame.K_w]:
            position[1] += dt*100
            change_detected = True
        if pygame.key.get_pressed()[pygame.K_s]:
            position[1] -= dt*100
            change_detected = True
        if pygame.key.get_pressed()[pygame.K_d]:
            position[0] += dt*100
            change_detected = True
        if pygame.key.get_pressed()[pygame.K_a]:
            position[0] -= dt*100
            change_detected = True
        if pygame.key.get_pressed()[pygame.K_EQUALS]:
            zoom *= (1+dt)
            change_detected = True
        if pygame.key.get_pressed()[pygame.K_MINUS]:
            zoom /= (1+dt)
            change_detected = True

        if change_detected:
            # utils.clear_gray_image(img_arr)
            textures.generate_height_map(img_arr,
                                         base_height=255*1.5,
                                         iterations=6,
                                         position=(position[0]+displacement[0],  position[1]+displacement[1]),
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
            # img_arr_back = img_arr_back.clip(0, 256)

            # img = Image.fromarray(img_arr_back).rotate(-90).resize(screen.get_size()).convert('RGB')
            img = Image.fromarray(img_arr_back).convert('RGB')
            img = pygame.image.fromstring(img.tobytes(), img.size, img.mode)
            img = pygame.transform.rotate(img, -90)
            img = pygame.transform.scale(img, [max(screen.get_size())]*2)

        try:
            screen.blit(img, (0, 0))
        except NameError:
            pass

        pygame.display.update()
        screen.fill(0)
        clock.tick(60)
        change_detected = False


if __name__ == '__main__':
    main()
