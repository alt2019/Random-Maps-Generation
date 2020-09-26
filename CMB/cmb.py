"""
    This script implements random gaussian generation
    of Cosmic Microwave Background - like small-angle density
    maps with 3 different methods of computation accelerations:
    1) numba just-in-time compilation;
    2) numba just-in-time compilation plus parallelization;
    3) numba CUDA acceleration,
    and 2 ways:
    1) using numpy matrices;
    2) using raw nested for-loops.

    Distribution is defined as follows:
    F(x,y) = a[k1, k2] * cos(2 * pi / Nx * (k1 * x + k2 * y)) +
             b[k1, k2] * sin(2 * pi / Ny * (k1 * x + k2 * y)),
    x, y - coordinates on 2d grid (Nx * Ny),
    a, b - normally distributed random values of size (Kx * Ky)
        with mean equal 0,
    k1, k2 - coordinates on 2d grid (Kx * Ky) of polarization vector,
    satisfying as k1^2 + k2^2 = k^2
"""

__all__ = ['generate_cmb', 'draw_cmb_map', 'compare_methods']

import time
import math as m
import numpy as np
import matplotlib.pyplot as plt
from numba import jit, cuda, prange, set_num_threads


class Timer:
    def __init__(self, description):
        self.description = description

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, type, value, traceback):
        self.end = time.time()
        self.exec_time = self.end - self.start
        print(f"{self.description}: {self.exec_time} seconds")
        return None


###########################################################
# no optimizition
###########################################################
def compute(
        x: np.ndarray,
        y: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        Nx: int, Ny: int, Kx: int, Ky: int) -> np.ndarray:
    """
        This is the numba optimized kernel: it performs computations of
        2d CMB-like map

        Parameters
        ----------
        x: numpy.ndarray
            array of x coordinate points from 0.0 to 1.0
        y: numpy.ndarray
            array of y coordinate points from 0.0 to 1.0
        a: numpy.ndarray
            normally distributed random values array
        b: numpy.ndarray
            normally distributed random values array
        Nx: int
            Number of points over x
        Ny: int
            Number of points over y
        Kx: int
            Number of modes over "x"
        Ky: int
            Number of moder over "y"

        Return
        ------
        f: numpy.ndarray
            Result array that will be filled with computed values
    """
    f = np.zeros((Nx, Ny))
    for i in prange(0, Nx):
        for j in prange(0, Ny):
            for ik1 in range(0, 2 * Kx + 1):
                for jk2 in range(0, 2 * Ky + 1):
                    f[i, j] +=\
                        a[ik1, jk2] * np.cos(
                            (ik1 - Kx) * x[i] / x[-1] * 2 * np.pi
                            + (jk2 - Ky) * y[j] / y[-1] * 2 * np.pi) +\
                        b[ik1, jk2] * np.sin(
                            (ik1 - Kx) * x[i] / x[-1] * 2 * np.pi
                            + (jk2 - Ky) * y[j] / y[-1] * 2 * np.pi)
    return f
###########################################################


###########################################################
# only jit optimization
###########################################################
@jit
def compute_jit(
        x: np.ndarray,
        y: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        Nx: int, Ny: int, Kx: int, Ky: int) -> np.ndarray:
    """
        This is the numba optimized kernel: it performs computations of
        2d CMB-like map

        Parameters
        ----------
        x: numpy.ndarray
            array of x coordinate points from 0.0 to 1.0
        y: numpy.ndarray
            array of y coordinate points from 0.0 to 1.0
        a: numpy.ndarray
            normally distributed random values array
        b: numpy.ndarray
            normally distributed random values array
        Nx: int
            Number of points over x
        Ny: int
            Number of points over y
        Kx: int
            Number of modes over "x"
        Ky: int
            Number of moder over "y"

        Return
        ------
        f: numpy.ndarray
            Result array that will be filled with computed values
    """
    f = np.zeros((Nx, Ny))
    for i in prange(0, Nx):
        for j in prange(0, Ny):
            for ik1 in range(0, 2 * Kx + 1):
                for jk2 in range(0, 2 * Ky + 1):
                    f[i, j] +=\
                        a[ik1, jk2] * np.cos(
                            (ik1 - Kx) * x[i] / x[-1] * 2 * np.pi
                            + (jk2 - Ky) * y[j] / y[-1] * 2 * np.pi) +\
                        b[ik1, jk2] * np.sin(
                            (ik1 - Kx) * x[i] / x[-1] * 2 * np.pi
                            + (jk2 - Ky) * y[j] / y[-1] * 2 * np.pi)
    return f
###########################################################


###########################################################
# optimized with numpy matrices
###########################################################
def compute_Matr(
        x: np.ndarray,
        y: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        k1: np.ndarray,
        k2: np.ndarray) -> np.ndarray:
    """
        This is the numba optimized kernel with numpy matrix
        operations: it performs computations of 2d CMB-like map

        Parameters
        ----------
        x: numpy.ndarray
            array of x coordinate points
        y: numpy.ndarray
            array of y coordinate points
        a: numpy.ndarray
            normally distributed random values array
        b: numpy.ndarray
            normally distributed random values array
        Nx: int
            Number of points over x
        Ny: int
            Number of points over y
        Kx: int
            Number of modes over "x"
        Ky: int
            Number of moder over "y"

        Return
        ------
        res_arr: numpy.ndarray
            Result array that will be filled with computed values
    """
    res_arr = np.zeros((len(x), len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            arg_x = 2 * np.pi / x[-1] * x[i] * k1
            arg_y = 2 * np.pi / y[-1] * y[j] * k2
            res_arr[i, j] = \
                np.cos(arg_x) @ a @ np.cos(arg_y) - \
                np.sin(arg_x) @ a @ np.sin(arg_y) + \
                np.sin(arg_x) @ b @ np.cos(arg_y) + \
                np.cos(arg_x) @ b @ np.sin(arg_y)
    return res_arr
###########################################################


###########################################################
# optimized with numba.jit + numpy matrices
###########################################################
@jit(nopython=True)
def compute_jit_Matr(
        x: np.ndarray,
        y: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        k1: np.ndarray,
        k2: np.ndarray) -> np.ndarray:
    """
        This is the numba optimized kernel with numpy matrix
        operations: it performs computations of 2d CMB-like map

        Parameters
        ----------
        x: numpy.ndarray
            array of x coordinate points
        y: numpy.ndarray
            array of y coordinate points
        a: numpy.ndarray
            normally distributed random values array
        b: numpy.ndarray
            normally distributed random values array
        Nx: int
            Number of points over x
        Ny: int
            Number of points over y
        Kx: int
            Number of modes over "x"
        Ky: int
            Number of moder over "y"

        Return
        ------
        res_arr: numpy.ndarray
            Result array that will be filled with computed values
    """
    res_arr = np.zeros((len(x), len(y)))
    for i in range(len(x)):
        for j in range(len(y)):
            arg_x = 2 * np.pi / x[-1] * x[i] * k1
            arg_y = 2 * np.pi / y[-1] * y[j] * k2
            res_arr[i, j] = \
                np.cos(arg_x) @ a @ np.cos(arg_y) - \
                np.sin(arg_x) @ a @ np.sin(arg_y) + \
                np.sin(arg_x) @ b @ np.cos(arg_y) + \
                np.cos(arg_x) @ b @ np.sin(arg_y)
    return res_arr
###########################################################


###########################################################
# optimized with numba.jit + parallel mode + numpy matrices
###########################################################
@jit(nopython=True, parallel=True)
def compute_jit_parallel_Matr(
        a: np.ndarray,
        b: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        k1: np.ndarray,
        k2: np.ndarray) -> np.ndarray:
    """
        This is the numba optimized kernel with numpy matrix
        operations and parallelization: it performs computations
        of 2d CMB-like map

        Parameters
        ----------
        x: numpy.ndarray
            array of x coordinate points
        y: numpy.ndarray
            array of y coordinate points
        a: numpy.ndarray
            normally distributed random values array
        b: numpy.ndarray
            normally distributed random values array
        Nx: int
            Number of points over x
        Ny: int
            Number of points over y
        Kx: int
            Number of modes over "x"
        Ky: int
            Number of moder over "y"

        Return
        ------
        res_arr: numpy.ndarray
            Result array that will be filled with computed values
    """
    res_arr = np.zeros((len(x), len(y)))
    for i in prange(len(x)):
        for j in prange(len(y)):
            arg_x = 2 * np.pi / x[-1] * x[i] * k1
            arg_y = 2 * np.pi / y[-1] * y[j] * k2
            res_arr[i, j] =\
                np.cos(arg_x) @ a @ np.cos(arg_y) - \
                np.sin(arg_x) @ a @ np.sin(arg_y) + \
                np.sin(arg_x) @ b @ np.cos(arg_y) + \
                np.cos(arg_x) @ b @ np.sin(arg_y)
    return res_arr
###########################################################


###########################################################
# optimized with numba.jit + parallel mode
###########################################################
@jit(nopython=True, parallel=True)
def compute_jit_parallel_loop(
        x: np.ndarray,
        y: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        Nx: int, Ny: int, Kx: int, Ky: int) -> np.ndarray:
    """
        This is the numba optimized kernel: it performs computations of
        2d CMB-like map

        Parameters
        ----------
        x: numpy.ndarray
            array of x coordinate points from 0.0 to 1.0
        y: numpy.ndarray
            array of y coordinate points from 0.0 to 1.0
        a: numpy.ndarray
            normally distributed random values array
        b: numpy.ndarray
            normally distributed random values array
        Nx: int
            Number of points over x
        Ny: int
            Number of points over y
        Kx: int
            Number of modes over "x"
        Ky: int
            Number of moder over "y"

        Return
        ------
        f: numpy.ndarray
            Result array that will be filled with computed values
    """
    f = np.zeros((Nx, Ny))
    for i in prange(0, Nx):
        for j in prange(0, Ny):
            for ik1 in range(0, 2 * Kx + 1):
                for jk2 in range(0, 2 * Ky + 1):
                    f[i, j] +=\
                        a[ik1, jk2] * np.cos(
                            (ik1 - Kx) * x[i] / x[-1] * 2 * np.pi
                            + (jk2 - Ky) * y[j] / y[-1] * 2 * np.pi) +\
                        b[ik1, jk2] * np.sin(
                            (ik1 - Kx) * x[i] / x[-1] * 2 * np.pi
                            + (jk2 - Ky) * y[j] / y[-1] * 2 * np.pi)
    return f
###########################################################


###########################################################
# accelerated with numba.cuda.jit 1d grid
###########################################################
@cuda.jit
def compute_cuda_1dgrid(
        x: np.ndarray,
        y: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        Nx: int, Ny: int, Kx: int, Ky: int,
        f: np.ndarray) -> None:
    """
        This is the 1d CUDA kernel: it performs computations of
        2d CMB-like map

        Parameters
        ----------
        x: numpy.ndarray
            array of x coordinate points from 0.0 to 1.0
        y: numpy.ndarray
            array of y coordinate points from 0.0 to 1.0
        a: numpy.ndarray
            normally distributed random values array
        b: numpy.ndarray
            normally distributed random values array
        Nx: int
            Number of points over x
        Ny: int
            Number of points over y
        Kx: int
            Number of modes over "x"
        Ky: int
            Number of moder over "y"
        f: numpy.ndarray
            Result array that will be filled with computed values;
            is obtained by pointer
    """
    tx = cuda.threadIdx.x  # unique thread ID within a 1D block
    ty = cuda.blockIdx.x  # unique block ID within the 1D grid

    block_size = cuda.blockDim.x  # number of threads per block
    grid_size = cuda.gridDim.x    # number of blocks in the grid

    start = tx + ty * block_size
    stride = block_size * grid_size

    for i in range(start, Nx, stride):
        for j in range(0, Ny):
            # for ik1 in range(0, 2 * Kx + 1):
            #     for jk2 in range(0, 2 * Ky + 1):
            #         f[i, j] +=\
            #             a[ik1, jk2] * m.cos(
            #                 (ik1 - Kx) * x[i] / x[-1] * 2 * np.pi
            #                 + (jk2 - Ky) * y[j] / y[-1] * 2 * np.pi) +\
            #             b[ik1, jk2] * m.sin(
            #                 (ik1 - Kx) * x[i] / x[-1] * 2 * np.pi
            #                 + (jk2 - Ky) * y[j] / y[-1] * 2 * np.pi)

            # 2 loops are replaced by one
            for ik in range(0, (2 * Kx + 1) * (2 * Ky + 1)):
                ik1 = 0
                jk2 = 0
                f[i, j] +=\
                    a[ik1, jk2] * m.cos(
                        (ik1 - Kx) * x[i] / x[-1] * 2 * np.pi
                        + (jk2 - Ky) * y[j] / y[-1] * 2 * np.pi) +\
                    b[ik1, jk2] * m.sin(
                        (ik1 - Kx) * x[i] / x[-1] * 2 * np.pi
                        + (jk2 - Ky) * y[j] / y[-1] * 2 * np.pi)
                if ik1 != 2 * Kx + 1:
                    ik1 += 1
                else:
                    ik1 = 0
                    jk2 += 1


###########################################################
# accelerated with numba.cuda.jit 2d grid
###########################################################
@cuda.jit
def compute_cuda_2dgrid(
        x: np.ndarray,
        y: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        Nx: int, Ny: int, Kx: int, Ky: int,
        f: np.ndarray) -> None:
    """
        This is the 2d CUDA kernel: it performs computations of
        2d CMB-like map

        Parameters
        ----------
        x: numpy.ndarray
            array of x coordinate points from 0.0 to 1.0
        y: numpy.ndarray
            array of y coordinate points from 0.0 to 1.0
        a: numpy.ndarray
            normally distributed random values array
        b: numpy.ndarray
            normally distributed random values array
        Nx: int
            Number of points over x
        Ny: int
            Number of points over y
        Kx: int
            Number of modes over "x"
        Ky: int
            Number of moder over "y"
        f: numpy.ndarray
            Result array that will be filled with computed values;
            is obtained by pointer
    """
    start1, start2 = cuda.grid(2)
    stride1, stride2 = cuda.gridsize(2)

    for i in range(start1, Nx, stride1):
        for j in range(start2, Ny, stride2):
            # for ik1 in range(0, 2 * Kx + 1):
            #     for jk2 in range(0, 2 * Ky + 1):
            #         f[i, j] +=\
            #             a[ik1, jk2] * m.cos(
            #                 (ik1 - Kx) * x[i] / x[-1] * 2 * np.pi
            #                 + (jk2 - Ky) * y[j] / y[-1] * 2 * np.pi) +\
            #             b[ik1, jk2] * m.sin(
            #                 (ik1 - Kx) * x[i] / x[-1] * 2 * np.pi
            #                 + (jk2 - Ky) * y[j] / y[-1] * 2 * np.pi)

            # 2 loops are replaced by one
            for ik in range(0, (2 * Kx + 1) * (2 * Ky + 1)):
                ik1 = 0
                jk2 = 0
                f[i, j] +=\
                    a[ik1, jk2] * m.cos(
                        (ik1 - Kx) * x[i] / x[-1] * 2 * np.pi
                        + (jk2 - Ky) * y[j] / y[-1] * 2 * np.pi) +\
                    b[ik1, jk2] * m.sin(
                        (ik1 - Kx) * x[i] / x[-1] * 2 * np.pi
                        + (jk2 - Ky) * y[j] / y[-1] * 2 * np.pi)
                if ik1 != 2 * Kx + 1:
                    ik1 += 1
                else:
                    ik1 = 0
                    jk2 += 1


###########################################################
# accelerated with numba.cuda.jit 3d grid
###########################################################
@cuda.jit
def compute_cuda_3dgrid(
        x: np.ndarray,
        y: np.ndarray,
        a: np.ndarray,
        b: np.ndarray,
        Nx: int, Ny: int, Kx: int, Ky: int,
        f: np.ndarray) -> None:
    """
        This is the 3d CUDA kernel: it performs computations of
        2d CMB-like map

        Parameters
        ----------
        x: numpy.ndarray
            array of x coordinate points from 0.0 to 1.0
        y: numpy.ndarray
            array of y coordinate points from 0.0 to 1.0
        a: numpy.ndarray
            normally distributed random values array
        b: numpy.ndarray
            normally distributed random values array
        Nx: int
            Number of points over x
        Ny: int
            Number of points over y
        Kx: int
            Number of modes over "x"
        Ky: int
            Number of moder over "y"
        f: numpy.ndarray
            Result array that will be filled with computed values;
            is obtained by pointer
    """
    start1, start2, start3 = cuda.grid(3)
    stride1, stride2, stride3 = cuda.gridsize(3)

    for i in range(start1, Nx, stride1):
        for j in range(start2, Ny, stride2):
            # 2 loops are replaced by one
            for ik in range(start3, (2 * Kx + 1) * (2 * Ky + 1), stride3):
                ik1 = 0
                jk2 = 0
                f[i, j] +=\
                    a[ik1, jk2] * m.cos(
                        (ik1 - Kx) * x[i] / x[-1] * 2 * np.pi
                        + (jk2 - Ky) * y[j] / y[-1] * 2 * np.pi) +\
                    b[ik1, jk2] * m.sin(
                        (ik1 - Kx) * x[i] / x[-1] * 2 * np.pi
                        + (jk2 - Ky) * y[j] / y[-1] * 2 * np.pi)
                if ik1 != 2 * Kx + 1:
                    ik1 += 1
                else:
                    ik1 = 0
                    jk2 += 1


def generate_cmb(
        Nx: int, Ny: int, Kx: int, Ky: int, Nt: int = 4,
        x_start: float = 0.0, x_stop: float = 1.0,
        y_start: float = 0.0, y_stop: float = 1.0,
        mean_a: float = 0., sigma_a: float = 10.,
        mean_b: float = 0., sigma_b: float = 10.,
        acceleration_type: str = 'cuda',
        cu_threads_per_block: int = 256,
        cu_blocks_per_grid: int = 32) -> np.ndarray:
    """
        This function creates 2d CMB random gauss generated map
        with cuda acceleration.

        Parameters
        ----------
        Nx: int
            Number of points over x
        Ny: int
            Number of points over y
        Kx: int
            Number of modes over "x"
        Ky: int
            Number of moder over "y"
        Nt: int
            Number of threads to run in parallel on CPU
        x_start: float, optional, default = 0.0
            Start point of x direction
        x_stop: float, optional, default = 1.0
            Stop point of x direction
        y_start: float, optional, default = 0.0
            Start point of y direction
        y_stop: float, optional, default = 1.0
            Stop point of y direction
        mean_a: float, optional, default = 0.0
            Mean of 'a' random array
        sigma_a: float, optional, default = 10.0
            Standard deeviation of 'a' random array
        mean_b: float, optional, default = 0.0
            Mean of 'b' random array
        sigma_b: float, optional, default = 10.0
            Standard deeviation of 'b' random array
        acceleration_type: str, default "cuda"
            Type of acceleration. Available are:
                "cuda1d": cuda acceleration of nested for-loops,
                        additional parameters: cu_threads_per_block,
                        cu_blocks_per_grid, threads grid is 1d
                "cuda2d": cuda acceleration of nested for-loops,
                        additional parameters: cu_threads_per_block,
                        cu_blocks_per_grid, threads grid is 2d
                "cuda3d": cuda acceleration of nested for-loops,
                        additional parameters: cu_threads_per_block,
                        cu_blocks_per_grid, threads grid is 3d
                "jit+par": numba just-in-type compilation plus
                       parallelization into Nt processes of
                       nested for-loops
                "Matr+jit": numba just-in-type compilation of
                       matrix operations
                "Matr+jit+par": numba just-in-type compilation plus
                       parallelization into Nt processes of
                       matrix operations
                "no": no acceleration (raw nested for-loops)
        cu_threads_per_block: int, optional, default = 256
            Number of threads per block on cuda device
        cu_blocks_per_grid: int, optional, default = 32
            Number of blocs per grid on cuda device

        Default settings:
            2d grid:
                x coordinate: [0.0, 1.0]
                y coordinate: [0.0, 1.0]

        Return
        ------
        numpy.ndarray
            2d array - CMB random gauss generated map
    """
    x = np.linspace(x_start, x_stop, Nx, endpoint=True)
    y = np.linspace(y_start, y_stop, Ny, endpoint=True)

    a = np.random.normal(mean_a, sigma_a, size=(2 * Kx + 1, 2 * Ky + 1))
    b = np.random.normal(mean_b, sigma_b, size=(2 * Kx + 1, 2 * Ky + 1))

    # cu_threads_per_block = 32
    # cu_blocks_per_grid = 384

    if acceleration_type == 'cuda1d':
        f = np.zeros((Nx, Ny))
        with Timer('numba-cuda1d+nested_loops'):
            compute_cuda_1dgrid[cu_blocks_per_grid, cu_threads_per_block](
                x, y, a, b, Nx, Ny, Kx, Ky, f)
    elif acceleration_type == 'cuda2d':
        f = np.zeros((Nx, Ny))
        with Timer('numba-cuda2d+nested_loops'):
            compute_cuda_2dgrid[cu_blocks_per_grid, cu_threads_per_block](
                x, y, a, b, Nx, Ny, Kx, Ky, f)
    elif acceleration_type == 'cuda3d':
        f = np.zeros((Nx, Ny))
        with Timer('numba-cuda3d+nested_loops'):
            compute_cuda_3dgrid[cu_blocks_per_grid, cu_threads_per_block](
                x, y, a, b, Nx, Ny, Kx, Ky, f)
    elif acceleration_type == 'Matr':
        k1 = np.linspace(-Kx, Kx, 2 * Kx + 1, dtype=int)
        k2 = np.linspace(-Ky, Ky, 2 * Ky + 1, dtype=int)
        with Timer('numpy_matr'):
            f = compute_Matr(x, y, a, b, k1, k2)
    elif acceleration_type == 'Matr+jit':
        k1 = np.linspace(-Kx, Kx, 2 * Kx + 1, dtype=int)
        k2 = np.linspace(-Ky, Ky, 2 * Ky + 1, dtype=int)
        with Timer('numba-jit+matr'):
            f = compute_jit_Matr(x, y, a, b, k1, k2)
    elif acceleration_type == 'Matr+jit+par':
        set_num_threads(Nt)
        k1 = np.linspace(-Kx, Kx, 2 * Kx + 1, dtype=int)
        k2 = np.linspace(-Ky, Ky, 2 * Ky + 1, dtype=int)
        with Timer('numba-jit+par+matr'):
            f = compute_jit_parallel_Matr(a, b, x, y, k1, k2)
    elif acceleration_type == 'jit':
        with Timer('numba-jit+nested_loops'):
            f = compute_jit(x, y, a, b, Nx, Ny, Kx, Ky)
    elif acceleration_type == 'jit+par':
        set_num_threads(Nt)
        with Timer('numba-jit+par+nested_loops'):
            f = compute_jit_parallel_loop(x, y, a, b, Nx, Ny, Kx, Ky)
    elif acceleration_type == 'no':
        with Timer('no acceleration'):
            f = compute(x, y, a, b, Nx, Ny, Kx, Ky)
    else:
        raise 'Unknown acceleration type'
    return f
###########################################################


def draw_cmb_map(arr: np.ndarray) -> None:
    """
        This function draws 2d array

        Parameters
        ----------
        arr: np.ndarray
            Array to draw
    """
    plt.title('small-angle CMB distribution map')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.imshow(arr, cmap='viridis')
    plt.colorbar()
    plt.show()


def compare_methods():
    N_arr = np.linspace(10, 100, 10, endpoint=True, dtype=int)
    # N_arr = np.array([32, 64, 128, 256, 512, 1024, 2048])
    # N_arr = np.array([16, 32, 64, 128, 256])
    no_acc = np.zeros(N_arr.shape)
    jit_acc = np.zeros(N_arr.shape)
    jit_par_acc = np.zeros(N_arr.shape)
    np_acc = np.zeros(N_arr.shape)
    np_jit_acc = np.zeros(N_arr.shape)
    np_jit_par_acc = np.zeros(N_arr.shape)
    cuda1d_acc = np.zeros(N_arr.shape)
    cuda2d_acc = np.zeros(N_arr.shape)
    cuda3d_acc = np.zeros(N_arr.shape)
    K = 10
    for i, N in enumerate(N_arr):
        print(N)
        with Timer('no acceleration') as t:
            generate_cmb(N, N, K, K, acceleration_type='no')
        no_acc[i] = t.exec_time
        with Timer('jit acceleration') as t:
            generate_cmb(N, N, K, K, acceleration_type='jit')
        jit_acc[i] = t.exec_time
        with Timer('numpy.Matrix acceleration') as t:
            generate_cmb(N, N, K, K, acceleration_type='Matr')
        np_acc[i] = t.exec_time
        with Timer('numpy.Matrix jit acceleration') as t:
            generate_cmb(N, N, K, K, acceleration_type='Matr+jit')
        np_jit_acc[i] = t.exec_time
        with Timer('numpy.Matrix jit parallel acceleration') as t:
            generate_cmb(N, N, K, K,
                         acceleration_type='Matr+jit+par')
        np_jit_par_acc[i] = t.exec_time
        with Timer('jit parallel acceleration') as t:
            generate_cmb(N, N, K, K, acceleration_type='jit+par')
        jit_par_acc[i] = t.exec_time
        with Timer('cuda1d acceleration') as t:
            generate_cmb(N, N, K, K, acceleration_type='cuda1d')
        cuda1d_acc[i] = t.exec_time
        with Timer('cuda2d acceleration') as t:
            generate_cmb(N, N, K, K, acceleration_type='cuda2d')
        cuda2d_acc[i] = t.exec_time
        with Timer('cuda3d acceleration') as t:
            generate_cmb(N, N, K, K, acceleration_type='cuda3d')
        cuda3d_acc[i] = t.exec_time

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle('Comparison of different acceleration types', fontsize=20)
    ax1.plot(N_arr, no_acc, color='blue', label='no_acc', marker='o')
    ax1.plot(N_arr, jit_acc, color='green', label='jit_acc', marker='o')
    ax1.plot(N_arr, jit_par_acc, color='cyan', label='jit_par_acc', marker='o')
    ax1.plot(N_arr, np_acc, color='magenta', label='Matr_acc', marker='o')
    ax1.plot(N_arr, np_jit_acc, color='yellow',
             label='Matr_jit_acc', marker='o')
    ax1.plot(N_arr, np_jit_par_acc, color='black',
             label='Matr_jit_par_acc', marker='o')
    ax1.plot(N_arr, cuda1d_acc, color='red', label='cuda1d_acc',
             linestyle='dashed', marker='o')
    ax1.plot(N_arr, cuda2d_acc, color='blue', label='cuda2d_acc',
             linestyle='dashed', marker='o')
    ax1.plot(N_arr, cuda3d_acc, color='green', label='cuda3d_acc',
             linestyle='dashed', marker='o')
    ax1.set_xlabel('N grid points (N*N)', fontsize=15)
    ax1.set_ylabel('time, s', fontsize=15)
    ax1.legend()

    ax2.set_yscale('log')
    ax2.plot(N_arr, no_acc, color='blue', label='no_acc', marker='o')
    ax2.plot(N_arr, jit_acc, color='green', label='jit_acc', marker='o')
    ax2.plot(N_arr, jit_par_acc, color='cyan', label='jit_par_acc', marker='o')
    ax2.plot(N_arr, np_acc, color='magenta', label='Matr_acc', marker='o')
    ax2.plot(N_arr, np_jit_acc, color='yellow',
             label='Matr_jit_acc', marker='o')
    ax2.plot(N_arr, np_jit_par_acc, color='black',
             label='Matr_jit_par_acc', marker='o')
    ax2.plot(N_arr, cuda1d_acc, color='red', label='cuda1d_acc',
             linestyle='dashed', marker='o')
    ax2.plot(N_arr, cuda2d_acc, color='blue', label='cuda2d_acc',
             linestyle='dashed', marker='o')
    ax2.plot(N_arr, cuda3d_acc, color='green', label='cuda3d_acc',
             linestyle='dashed', marker='o')
    ax2.set_xlabel('N grid points (N*N)', fontsize=15)
    ax2.set_ylabel('time, log(s)', fontsize=15)
    ax2.legend()
    fig.set_size_inches(14.8 * 1.5, 8.33 * 1.5)  # sizes of 17-inch display
    plt.savefig('methods-comparison-smallN-03.png', dpi=200,
                orientation='landscape')
    plt.show()


if __name__ == '__main__':
    help(__name__)
    # N = 1024
    # K = 3
    # cmb_arr = generate_cmb(N, N, K, K, acceleration_type='no')
    # cmb_arr = generate_cmb(N, N, K, K, acceleration_type='Matr')
    # cmb_arr = generate_cmb(N, N, K, K, acceleration_type='Matr+jit')
    # cmb_arr = generate_cmb(N, N, K, K,
    #                        acceleration_type='Matr+jit+par')
    # cmb_arr = generate_cmb(N, N, K, K, acceleration_type='jit+par')
    cmb_arr = generate_cmb(N, N, K, K, acceleration_type='cuda2d')
    draw_cmb_map(cmb_arr)
    compare_methods()
