import time
import numpy as np
from numba import njit
# def gemm_multiplication1(C,A,B):
#     """
#     进行一个gemm的加速 C = C + A*B
#     :param m1:
#     :param m2:
#     :return:
#     """
#     for i in range(n):
#         for j in range()

@njit()
def multiplication1(m0,m1, m2):
    result = m0
    for i in range(m1.shape[0]):
        for j in range(m2.shape[1]):
            for k in range(m2.shape[0]):
                # print(i,j,k)
                result[i][j] += m1[i][k] * m2[k][j]
    return result
def multiplication3(m0,m1,m2):
    result = m0
    result +=np.dot(m1,m2)
    return result
@njit()
def multiplication_outer(m0,m1, m2):
    result = m0
    for k in range(m2.shape[0]):
        for i in range(m1.shape[0]):
            for j in range(m2.shape[1]):
                # print(i,j,k)
                result[i][j] += m1[i][k] * m2[k][j]
    return result
def multiplication3_blockm1(m0,m1,m2):
    result = m0
    m11,m12 = np.split(m1,2)
    # m21,m22 = np.split(m2,2,axis=1)
    t = np.vstack((np.dot(m11,m2),np.dot(m12,m2)))
    result +=t
    return result

def exp(len):
    n = len**2
    C =np.empty((len,len)).reshape(len,len)
    print(C.nbytes/1024/1024/8,'MB')
    A = np.empty((len,len)).reshape(len, len)
    A.fill(1)
    B = np.empty((len,len)).reshape(len, len)
    B.fill(2)
    start = time.time()
    r = multiplication1(C,A,B)
    print('njit multiplication time', time.time() - start)
    start = time.time()
def exp2(len):
    cache = 8192*1024*8
    cache2 = 4096*1024*8
    n = len**2
    C =np.empty((len,len)).reshape(len,len)
    if C.nbytes*2 < cache+cache2:
        print('缓存命中')
    else:
        print('缓存失效')
    print(C.nbytes/1024/1024/8,'MB')
    A = np.empty((len,len)).reshape(len, len)
    A.fill(1)
    B = np.empty((len,len)).reshape(len, len)
    B.fill(2)
    start = time.time()
    r = multiplication3(C,A,B)
    print('numpy multiplication time', time.time() - start)
    start = time.time()
    r = multiplication3_blockm1(C,A,B)
    print('numpy multiplication3_blockA time', time.time() - start)

def exp3(len):
    cache = 8192 * 1024 * 8
    cache2 = 4096 * 1024 * 8
    cache1 = 32 *1024*8

    n = len ** 2
    C = np.empty((len, len),dtype=int).reshape(len, len)
    print('C shape',C.shape,'C nbytes:',C.nbytes/1024/8,'KB')
    if C.nbytes * 2 < cache1:
        print('l1缓存命中')
    else:
        print('l1缓存失效')
    print(C.nbytes /1024/8, 'KB')
    A = np.empty((len, len),dtype=int).reshape(len, len)
    A.fill(1)
    B = np.empty((len, len),dtype=int).reshape(len, len)
    B.fill(2)
    # r = multiplication1(C, A, B)
    start = time.time()
    r = multiplication1(C, A, B)
    t2 = time.time() - start
    print('python multiplication time', t2)
    start = time.time()
    r = multiplication_outer(C, A, B)
    t2 = time.time() - start
    print('python outer multiplication time', t2)

    return t2,C.nbytes*2/8/1024

if __name__ == '__main__':
    cache3 = 8192
    cache2 = 4096
    cache1 = 32 *1024*8
    len = 10
    C = np.empty((len, len),dtype=int).reshape(len, len)
    A = np.empty((len, len),dtype=int).reshape(len, len)
    A.fill(1)
    B = np.empty((len, len),dtype=int).reshape(len, len)
    B.fill(2)
    multiplication1(C,A,B)
    multiplication_outer(C,A,B)
    # len = 2500
    # exp2(len)
    import math
    # len1 = int(math.sqrt(cache1/2/4))
    # print(len1)
    # exp3(len1-5)
    # exp3(len1+30)
    len1 = 257
    t0,b0 = exp3(len1)
    t,b =exp3(len1+1)
    t1,b1 = exp3(len1-1)
    print('l1 cache speed up:',t0/t)
    import matplotlib.pyplot as plt
    x = [b1,b0,b]
    y = [t1,t0,t]
    plt.ylabel("Multiplication Time(s)")
    plt.xlabel("Matrix Size(KB)")
    plt.plot(x,y)
    plt.xticks(x)
    plt.show()
