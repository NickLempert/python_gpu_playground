import math

from numba import cuda


def gpu_program(compile_options=(), dimensions=2, auto_compile=True, default_shape_source=0):
    threadsperblock = tuple([32//dimensions]*dimensions)

    def __helper__(func):
        if auto_compile:
            func = cuda.jit(*compile_options)(func)

        def __out__(*args, shape=None):
            shape = shape or args[default_shape_source].shape
            blockspergrid = tuple((math.ceil(shape[i] / threadsperblock[i]) for i in range(dimensions)))
            func[blockspergrid, threadsperblock](*args)
        return __out__
    return __helper__
