import shader2d
from shader2d import init, CTX


class TerrainShader(shader2d.FullScreenShader):
    def __init__(self,
                 seed=0,
                 base_height=1.5,
                 iterations=6,
                 position: tuple[float, float] | list[float, float] = (0.0, 0.0),
                 zoom=1.0,
                 contrast=0.0):
        with open('terrain.frag', 'r') as f:
            super().__init__(f.read())
        self.put_variables(base_height=base_height,
                           iterations=iterations,
                           position=position,
                           zoom=zoom,
                           contrast=contrast,
                           seed=seed)



