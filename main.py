from numba import cuda
import time
import math
import numpy as np

size = 500
"""Инициализация матриц"""
matrix_cpu1 = np.random.randint(0, 10, (size, size))
matrix_cpu2 = np.random.randint(0, 10, (size, size))
cpu_matrix_result = np.zeros((size, size), dtype=int)

matrix_gpu1 = cuda.to_device(matrix_cpu1)
matrix_gpu2 = cuda.to_device(matrix_cpu2)
gpu_matrix_result = cuda.device_array((len(matrix_cpu1), len(matrix_cpu2)))


"""Конфигурация ядра"""
threadsperblock = (32, 32)
blockspergrid_x = int(math.ceil(matrix_cpu1.shape[0] / threadsperblock[0]))
blockspergrid_y = int(math.ceil(matrix_cpu2.shape[1] / threadsperblock[1]))
blockspergrid = (blockspergrid_x, blockspergrid_y)


def cpu_matmul(A, B, C):
    """Умножение матриц CPU"""
    for i in range(size):
        for j in range(size):
            rez = 0
            for z in range(size):
                rez += A[i,z] * B[z,j]
            C[i,j] = rez

""""""
@cuda.jit
def gpu_matmul(A, B, C):
    """Умножение матриц GPU"""
    i, j = cuda.grid(2)
    if i < C.shape[0] and j < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[i, k] * B[k, j]
        C[i, j] = tmp

def main():

    print("CPU")
    start_time = time.time()
    cpu_matmul(matrix_cpu1, matrix_cpu2, cpu_matrix_result)
    print(" %s sec" % (time.time() - start_time))

    print("GPU")
    start_time = time.time()
    """Передача параметров ядра"""
    gpu_matmul[blockspergrid, threadsperblock](matrix_gpu1, matrix_gpu2, gpu_matrix_result)
    print(" %s sec" % (time.time() - start_time))

if __name__ == "__main__":
    main()
