import array

import pygame
import moderngl
import sys

CTX: moderngl.Context | None = None


def init():
    global CTX
    if CTX is None:
        CTX = moderngl.create_context()
        CTX.enable(moderngl.BLEND)
    return CTX


class FullScreenShader:
    def __init__(self, fragment_shader: str, vertex_shader: str | None = None):
        vertex_shader = vertex_shader or """
                #version 330 core
                in vec2 in_vert;
                out vec2 uvs;
                void main() {
                    gl_Position = vec4(in_vert, 0.0, 1.0);
                    uvs = in_vert * 0.5 + 0.5;
                }
                """
        self.program = CTX.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)

        self.quad_vertices = CTX.buffer(data=array.array('f', [
            -1.0, -1.0,
            1.0, -1.0,
            -1.0, 1.0,
            -1.0, 1.0,
            1.0, -1.0,
            1.0, 1.0,
        ]))

        self.vao = CTX.vertex_array(self.program, [(self.quad_vertices, '2f', 'in_vert')])

    def put_variables(self, **kwargs):
        for key, val in kwargs.items():
            self.program[key].value = val

    def render(self):
        self.vao.render(moderngl.TRIANGLES)


if __name__ == '__main__':

    FRAGMENT_SHADER = """"""
    with open('light_mouse.frag', 'r') as f:
        FRAGMENT_SHADER += f.read()
    pygame.init()
    display = pygame.display.set_mode((800, 600), pygame.OPENGL | pygame.DOUBLEBUF)
    init()
    clock = pygame.time.Clock()
    shader = FullScreenShader(FRAGMENT_SHADER)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        CTX.clear(0.0, 0.0, 0.0, 1.0)
        mouse_pos = pygame.mouse.get_pos()
        # shader.put_variables(resolution=display.get_size())
        shader.put_variables(mouse=[mouse_pos[0] / display.get_width(), 1 - mouse_pos[1] / display.get_height()])
        shader.put_variables(aspect_ratio=display.get_height() / display.get_width())
        shader.render()
        print(clock.get_fps())
        pygame.display.flip()
        clock.tick(60)
