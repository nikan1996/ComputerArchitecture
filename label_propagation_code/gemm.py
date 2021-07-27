from numba import njit
import numpy as np
import time


@njit()
def multiplication1(m0, m1, m2):
    result = m0
    for i in range(m1.shape[0]):
        for j in range(m2.shape[1]):
            for k in range(m2.shape[0]):
                # print(i,j,k)
                result[i][j] += m1[i][k] * m2[k][j]
    return result


@njit()
def dot(c: np.ndarray, a:np.ndarray, b:np.ndarray,i,j,block_size):
    C = c[i:i+block_size,j:j+block_size]
    k = b.shape[0]
    A = a[i:i+block_size,:]
    B =  b[:,j:j+block_size]
    for p in range(k):
        for i1 in range(block_size):
            for j1 in range(block_size):

                C[i1,j1] += A[i1,p]*B[p,j1]

    return C

@njit()
def multiplication1_block(m0, m1, m2, block_size):
    result = m0
    # 注意考虑末尾溢出的情况
    for i in range(0, m1.shape[0], block_size):
        for j in range(0, m2.shape[1], block_size):
                dot(m0,m1,m2,i,j,block_size)
    return result


@njit()
def outer(m0, m1, m2):
    result = m0
    for k in range(m2.shape[0]):
        for i in range(m1.shape[0]):
            for j in range(m2.shape[1]):
                # print(i,j,k)
                result[i][j] += m1[i][k] * m2[k][j]
    return result


def get_C_A_B(len):
    C = np.empty((len, len), dtype=int).reshape(len, len)
    A = np.empty((len, len), dtype=int).reshape(len, len)
    A.fill(1)
    B = np.empty((len, len), dtype=int).reshape(len, len)
    B.fill(2)
    return C, A, B


def exp3(len):
    # r = multiplication1(C, A, B)
    block_size = 8
    # 预热
    C, A, B = get_C_A_B(16)
    r = multiplication1(C, A, B)
    r = multiplication1_block(C, A, B, block_size=block_size)

    start = time.time()
    C, A, B = get_C_A_B(len)
    r = multiplication1(C, A, B)
    t2 = time.time() - start
    print('python multiplication time', t2)
    start = time.time()
    C, A, B = get_C_A_B(len)
    r = outer(C, A, B)
    t2 = time.time() - start
    print('python outer multiplication time', t2)
    start = time.time()
    C, A, B = get_C_A_B(len)
    r = multiplication1_block(C, A, B, block_size=block_size)
    t2 = time.time() - start
    print(f'python block multiplication time{t2},block size:{block_size}')
    return t2, C.nbytes * 2 / 8 / 1024


import matplotlib.pyplot as plt


def exp4():
    block_size = 8
    size = 256
    # 预热
    C, A, B = get_C_A_B(size)
    r = multiplication1(C, A, B)
    r = multiplication1_block(C, A, B, block_size=block_size)
    #
    tn = []
    sizes = [200, 400, 600, 800, 1000]
    x = [i for i in sizes]
    for size in sizes:
        C, A, B = get_C_A_B(size)
        start = time.time()
        multiplication1(C, A, B)
        t = time.time() - start
        tn.append(t)
    for i, block_size in enumerate(range(4, 9, 4)):
        tbs = []
        for size in sizes:
            C, A, B = get_C_A_B(size)
            start = time.time()
            multiplication1_block(C, A, B, block_size=block_size)
            tb = time.time() - start
            tbs.append(tb)
        print(tbs)
        for i in range(len(tn)):
            print('speedup:', tn[i] / tbs[i])
        plt.plot(x, tbs, label=f'block_{block_size}x{block_size}')
    plt.title('GEMM TEST')
    print(tn)

    # plt.plot(x, tn, label=f'normal')
    # plt.ylabel("Multiplication Time(s)")
    # plt.xlabel("Matrix Size")
    # plt.legend()
    # plt.xticks(x)
    # plt.yticks(tn)

    plt.show()


if __name__ == '__main__':
    # exp3(256)
    # exp3(512)
    # exp3(786)
    # exp3(64)
    exp4()
    exp4()
    # exp4()
    # exp4()
