import linecache
import math

from numba import cuda
# import numba_dpex as dpex
import inspect


def gpu_program(compile_options=(), dimensions=2, auto_compile=True, default_shape_source=0):
    threadsperblock = tuple([math.ceil(1024**(1/dimensions))]*dimensions)

    def __helper__(func):
        if auto_compile:
            func = cuda.jit(*compile_options)(func)

        def __out__(*args, shape=None, limit_threads=0):
            nonlocal threadsperblock
            if limit_threads:
                threadsperblock = tuple([math.ceil(limit_threads**(1/dimensions))]*dimensions)
            shape = shape or args[default_shape_source].shape
            blockspergrid = tuple((math.ceil(shape[i] / threadsperblock[i]) for i in range(dimensions)))
            func[blockspergrid, threadsperblock](*args)
        return __out__
    return __helper__


GENERATED_CODE_FAKE_FILENAME = 'GENERATED_CODE_FAKE_FILENAME.py'


def cuda_to_intel(function, imported_names: dict[str, any] = None, replace_call='cuda.grid('):
    imported_names = imported_names or {}
    code = inspect.getsource(function)[:]
    grid_call_found = code.find(replace_call)
    while grid_call_found != -1:
        after_grid_call_start = code[grid_call_found+len(replace_call):]
        closing_bracket = after_grid_call_start.index(')')
        axis_count = int(after_grid_call_start[:closing_bracket])
        replacement_calls = f'({", ".join((f"dpex.get_global_id({i})" for i in range(axis_count)))})'
        code = code[:grid_call_found] + replacement_calls + after_grid_call_start[closing_bracket+1:]
        grid_call_found = code.rfind(replace_call)

    namespace = {}
    namespace.update(imported_names)
    namespace.update(globals())

    exec(compile(code, GENERATED_CODE_FAKE_FILENAME, "exec"), namespace)

    linecache.cache[GENERATED_CODE_FAKE_FILENAME] = (
        len(code),
        None,
        code.splitlines(keepends=True),
        GENERATED_CODE_FAKE_FILENAME,
    )
    return namespace[function.__name__]


def polar_to_cartesian(rotation: float, distance=1.0):
    return math.cos(rotation)*distance, math.sin(rotation)*distance


def cartesian_to_polar(x: float, y: float):
    return math.atan2(y, x)

