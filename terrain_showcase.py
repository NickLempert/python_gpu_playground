import threading
import time

import numpy as np
import pygame.display
from PIL import Image
from numba import cuda

import textures
import utils


class Terrain(threading.Thread):
    def __init__(self,
                 image_resolution: tuple[int, int] | list[int, int],
                 base_height=255*1.5,
                 iterations=6,
                 position: tuple[float, float] | list[float, float] = (0.0, 0.0),
                 zoom=1.0,
                 contrast=0.0):
        self.running = False
        self.images = []
        self.create_new_images(image_resolution)
        self.current_image = 0
        self.base_height = base_height
        self.iterations = iterations
        self.position = list(position)
        self.zoom = zoom
        self.contrast = contrast
        self.rendered_image = self.images[0].copy_to_host()
        self.updated = True
        self.sleep_time = 0.01
        super().__init__()

    def create_new_images(self, resolution):
        self.images = [cuda.to_device(np.zeros(resolution)) for _ in range(2)]

    def run(self):
        self.running = True
        t = time.perf_counter()
        while self.running:
            dt = time.perf_counter()-t
            if dt != 0:
                print(1/dt, 'fps 2')
            t = time.perf_counter()
            textures.generate_height_map(self.images[self.current_image],
                                         base_height=self.base_height,
                                         iterations=self.iterations,
                                         position=tuple(self.position),
                                         zoom=self.zoom,
                                         contrast=self.contrast)
            ring_distance = 10
            ring_width = 2
            if self.zoom <= 1:
                ring_width = 4
            if self.zoom < 0.5:
                ring_distance = 20
            if self.zoom < 0.25:
                ring_distance = 20
                ring_width = 8
            if self.zoom < 0.125:
                ring_distance = 80
                ring_width = 32
            textures.height_rings(self.images[self.current_image], ring_distance, ring_width)
            self.current_image = (self.current_image + 1) % len(self.images)
            self.updated = False
            cuda.synchronize()
            if dt != 0:
                time.sleep(self.sleep_time)

    def get_image(self):
        if not self.updated:
            self.rendered_image = self.images[self.current_image-1].copy_to_host()
            self.updated = True
        return self.rendered_image

    def join(self, timeout: float | None = None):
        self.running = False
        super().join(timeout)


def main(screen=None):
    screen = screen or pygame.display.set_mode((700, 700), pygame.RESIZABLE)
    clock = pygame.time.Clock()
    # img_arr = cuda.to_device(np.zeros([max(screen.get_size())]*2))
    # img_arr = cuda.to_device(np.zeros(list(reversed(screen.get_size()))))
    terrain = Terrain(list(reversed(screen.get_size())))
    terrain.start()
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
                terrain.join()
                exit()
            if event.type == pygame.WINDOWRESIZED:
                change_detected = True
                # img_arr = cuda.to_device(np.zeros([min(1000, max(screen.get_size()))]*2))
                size = list(reversed(screen.get_size()))
                max_res = 1920
                if max(size) > max_res:
                    size[0] *= max_res/max(size)
                    size[1] *= max_res/max(size)
                size = list(map(int, size))
                terrain.create_new_images(size)
                # img_arr = cuda.to_device(np.zeros(size))
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

        terrain.zoom = zoom
        terrain.position = (position[0]+displacement[0],  position[1]+displacement[1])

        if change_detected:
            # # utils.clear_gray_image(img_arr)
            # textures.generate_height_map(img_arr,
            #                              base_height=255*1.5,
            #                              iterations=6,
            #                              position=(position[0]+displacement[0],  position[1]+displacement[1]),
            #                              zoom=zoom,
            #                              contrast=0.0)
            # # textures.perlin_noise(img_arr, 20*zoom, strength=255*1.5, position=position)
            # ring_distance = 10
            # ring_width = 2
            # if zoom <= 1:
            #     ring_width = 4
            # if zoom < 0.5:
            #     ring_distance = 20
            # if zoom < 0.25:
            #     ring_distance = 20
            #     ring_width = 8
            # if zoom < 0.125:
            #     ring_distance = 80
            #     ring_width = 32
            img_arr_back = terrain.get_image()
            # textures.height_rings(img_arr, ring_distance, ring_width)

            # img_arr_back = img_arr.copy_to_host()
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
        change_detected = True


if __name__ == '__main__':
    main()
