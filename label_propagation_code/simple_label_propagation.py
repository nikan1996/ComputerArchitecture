import time
from multiprocessing import shared_memory,pool
import psutil

import numpy as np
import pandas
from sklearn.model_selection import train_test_split
import math
import multiprocessing
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages


def read_shape_sets(file):
    """

           x      y  label
    0  15.55  28.65      2
    1  14.90  27.55      2
    2  14.45  28.35      2
    3  14.15  28.80      2
    4  13.75  28.05      2
   shape:(787, 3)
    """
    data = pandas.read_csv(file, sep='\t', names=['x', 'y', 'label'])
    # print(data.head())
    # print(data.shape)
    return data


def construct_matrix_F_P(labeled_data, unlabeled_data, alpha=1):
    """
    构建矩阵F (NXC)
    F = [YL;YC]
    :return:
    """

    all_labels = pandas.unique(labeled_data['label'])
    all_labels = sorted(all_labels)
    # print(all_labels)
    # 合并一下
    new_data = pandas.concat([labeled_data, unlabeled_data])
    # print(new_data.shape)

    matrix = new_data.to_numpy()
    # print(matrix)
    n = new_data.shape[0]
    c = len(all_labels)
    # nxc
    matrix_F = np.full((n, c), 0, dtype=float)
    # nxn
    # 构造F矩阵，有label的一项赋值为1 其他为0
    for index, row in enumerate(matrix):
        label = int(row[2])
        if label != 0:
            # label值比索引大1，所以要减一下1
            matrix_F[index][label - 1] = 1
    # 构造P矩阵，根据欧几里得距离进行权重计算
    matrix_P = np.full((n, n), 0, dtype=float)
    for index, row in enumerate(matrix):
        weight = 0
        for c_index, c_row in enumerate(matrix):
            # 计算转移概率，
            x = row[0]
            y = row[1]
            c_x = c_row[0]
            c_y = c_row[1]
            euclidean_distance = (x - c_x) ** 2 + (y - c_y) ** 2
            weight_i_j = math.exp(-euclidean_distance / (alpha * alpha))
            # print(weight_i_j)
            matrix_P[index, c_index] = weight_i_j
            weight += weight_i_j
        matrix_P[index] /= weight
    # print(matrix_P.shape)

    return matrix_F, matrix_P

    # labeled_matrix = np.multiply()


def label_propagation(F, P, labeled_number):
    pul = P[labeled_number:, :labeled_number]
    # print(pul.shape)
    puu = P[labeled_number:, labeled_number:]
    # print(puu.shape)
    fu = np.array(F[labeled_number:])
    yl = np.array(F[:labeled_number])
    # print(fu.shape)

    start = time.time()
    propagation = True
    last_fu = fu
    round = 0
    while propagation is True:
        round += 1
        fu = np.dot(puu, fu) + np.dot(pul, yl)
        loss = np.mean(last_fu - fu)
        if loss == 0:
            propagation = False
            continue
        last_fu = fu

    print(f'标签传播花费时间:{time.time() - start:.3f}s,传播轮次{round}')
    return fu


def speedup_label_propagation(F, P, labeled_number, each_i=5):
    pul = P[labeled_number:, :labeled_number]
    puu = P[labeled_number:, labeled_number:]
    print(puu.nbytes / 1024 / 1024 / 8)
    fu = np.array(F[labeled_number:])
    yl = np.array(F[:labeled_number])
    start = time.time()
    propagation = True
    last_fu = fu
    round = 0
    t = np.linalg.matrix_power(puu, 0)
    puu2 = np.dot(puu, puu)
    puu_cache = {'puu2': puu2}
    b = t
    t111 = time.time()
    for i in range(1, each_i):
        if i == 1:
            b += puu
        else:
            if 'puu' + str(i) == 'puu2':
                b += puu_cache['puu' + str(i)]
            else:
                puu_cache['puu' + str(i)] = np.dot(puu_cache.get('puu' + str(i - 1)), puu)
                b += puu_cache['puu' + str(i)]
    a = np.dot(puu_cache.get('puu' + str(each_i - 1)), puu)
    c = np.linalg.multi_dot([b, pul, yl])
    while propagation is True:
        round += 1
        fu = np.dot(a, last_fu) + c
        loss = np.mean(last_fu - fu)
        # print(loss)
        if loss == 0:
            propagation = False
        last_fu = fu
    propagation_time = time.time() - start
    print(f'标签传播花费时间:{propagation_time:.3f}s,传播轮次{round}')
    return np.copy(last_fu), propagation_time

def create_shm_nparray(matrix):
    shm = shared_memory.SharedMemory(create=True, size=matrix.nbytes)
    b = np.ndarray(shape=matrix.shape, dtype=matrix.dtype, buffer=shm.buf)
    b[:] = matrix[:]
    return b.shape, b.dtype, shm.name, shm, b


def nparray_from_shm(shape, dtype, name):
    shm_matrix = shared_memory.SharedMemory(name=name)
    matrix = np.ndarray(shape=shape, dtype=dtype, buffer=shm_matrix.buf)
    return matrix, shm_matrix


def shm_close(shms):

    for shm in shms:
        shm.close()
        shm.unlink()

def _propa(a_args, fu_args, c_args,start,end,affinity_core):
    p = psutil.Process()
    print(f"{p}, affinity {p.cpu_affinity()}", flush=True)

    a,shm_a =nparray_from_shm(*a_args)
    a = np.copy(a)

    fu,shm_fu =nparray_from_shm(*fu_args)
    _fu = np.copy(fu[:,start:end])
    c,shm_c =nparray_from_shm(*c_args)
    c = np.copy(c[:,start:end])
    print('sizes a fu c',a.shape,_fu.shape,c.shape)
    propagation = True
    last_fu = _fu
    t = time.time()
    while propagation is True:
        tmp_fu = np.dot(a, last_fu) + c
        loss = np.mean(last_fu - tmp_fu)
        if loss == 0:
            break
        last_fu = tmp_fu
        # print('单次速度:',time.time()-f)
    print('分块并行t:',time.time()-t)

    # 修改shm内容
    _fu[:] = last_fu
    print(last_fu.shape)

    shm_close([shm_fu,shm_c,shm_a])
    return
def _paraller_propa(a,fu,c,worker=2):
    x = 1024*1024*8
    print('a',a.nbytes/x)
    print('fu',fu.nbytes/x)
    print('c',c.nbytes/x)
    # puu bytes 4.490756988525391 MB
    # 进行竖切， 每个woker传播不同的label
    # 比如worker1 负责传播label 1-5 , worker2 传播label 6-10, ...
    worker_pool = pool.Pool(2)
    worker_pool.apply(sum,args=([1,2],))
    a_args = create_shm_nparray(a)
    fu_args = create_shm_nparray(fu)
    c_args = create_shm_nparray(c)
    shape = fu.shape
    label_numer = shape[1]
    slice_index=  label_numer//2
    start = 0
    end = start+slice_index
    end2 = label_numer
    # future1 = worker_pool.apply_async(_propa, args=(a_args[:3], fu_args[:3], c_args[:3], 0, label_numer, 1))

    future1 = worker_pool.apply_async(_propa,args=(a_args[:3],fu_args[:3],c_args[:3],start,end,1))
    # fu2 = _propa(a_args[:3], fu_args[:3], c_args[:3], end, end2, 2)
    future2 = worker_pool.apply_async(_propa,args=(a_args[:3],fu_args[:3],c_args[:3],end,end2,2))
    t = time.time()
    print('wait...')

    fu1 = future1.get()
    print('end2',time.time()-t)
    propagation_time = time.time()-t
    c = np.copy(fu_args[-1])
    shm_close([a_args[-2],fu_args[-2],c_args[-2]])
    return c,propagation_time

def parallel_label_propagation(F, P, labeled_number, each_i=5):
    # existing_shm = shared_memory.SharedMemory(name='np1')
    pul = P[labeled_number:, :labeled_number]
    puu = P[labeled_number:, labeled_number:]
    # print('puu bytes', puu.nbytes / 1024 / 1024 / 8, 'MB')
    fu = np.array(F[labeled_number:])
    yl = np.array(F[:labeled_number])
    start = time.time()

    b = np.linalg.matrix_power(puu, 0)
    puu2 = np.dot(puu, puu)
    puu_cache = {'puu2': puu2}

    for i in range(1, each_i):
        if i == 1:
            b += puu
        else:
            if 'puu' + str(i) == 'puu2':
                b += puu_cache['puu' + str(i)]
            else:
                s = time.time()
                puu_cache['puu' + str(i)] = np.dot(puu_cache.get('puu' + str(i - 1)), puu)
                b += puu_cache['puu' + str(i)]
                # print('dot time', time.time() - s)
    a = np.dot(puu_cache.get('puu' + str(each_i - 1)), puu)
    c = np.linalg.multi_dot([b, pul, yl])
    # 优化初始化时间

    fu,propagation_time= _paraller_propa(a, fu, c)
    # fu = _propa(a,fu,c)
    print(f'并行标签传播花费时间:{propagation_time:.3f}s')
    return fu, propagation_time


def main():
    """
    标签传播算法是一个较为简单的聚类算法，
    数据分为 有标签的数据和无标签的数据，有标签的数据把他们的标签传播到附近无标签的数据中，这就是标签传播。
    首先我们需要构建一个图，或者说关系矩阵。
    假设有 N个数据，其中L个是有标签数据，U个是无标签数据 ，N=L+U
    矩阵F是 NXC的矩阵，C是标签，在这里一个数据只会有一个标签，或者多个也是可以的
    矩阵F格式：
       C1 C2 C3
    L1 0  1  0
    L2 0  0  1
    U1 0  0  0
    U2 0  0  0
    NXN的概率矩阵P，行i 列j.
    Fij表示 第i个数据和第j个数据的传播概率，
    传播概率公式：Wij = Exp(-||xi-xj||2 / a平方)
    alpha 是超参数。
    通过
    F = PF 乘积后得到的F 是标签传播后的矩阵，主要是要把Unlabeled的数据标签进行更新，labeled的数据并不需要做更新。
    这个F需要执行多次，如何判断收敛呢？
    """
    filename = 'data/Aggregation.txt'
    df = read_shape_sets(filename)
    # test_size 70% 因为 我认为 标签传播算法只需要一小部分有标签的数据，其它数据可以通过传播得到标签。
    train, test = train_test_split(df, test_size=0.7,random_state=7)

    test = test.assign(label=0)
    labeled_number = train.shape[0]
    print('label_number', labeled_number)
    F, P = construct_matrix_F_P(labeled_data=train, unlabeled_data=test)
    speedup_label_propagation(F, P, labeled_number, each_i=100)


def main2():
    filename = 'data/Aggregation.txt'
    df = read_shape_sets(filename)
    train, test = train_test_split(df, test_size=0.7)
    test = test.assign(label=0)
    labeled_number = train.shape[0]
    print('label_number', labeled_number)
    F, P = construct_matrix_F_P(labeled_data=train, unlabeled_data=test)
    final_f = speedup_label_propagation(F, P, labeled_number)
    # final_f = label_propagation(F, P, labeled_number)

    # UxC的一个矩阵
    labels = np.argmax(final_f, axis=1) + 1
    print(labels)
    print(test.head())
    test_row = test.index
    for i, label in enumerate(labels):
        test.at[test_row[i], 'label'] = label
    print(test.head())
    print('end')
    colors = ['black', 'red', 'cyan', 'navy', 'tomato', 'pink', 'orange']
    # 来个坐标点图
    # plt.plot(train['x'].to_numpy(),train['y'].to_numpy(),linestyle='None',  markerfacecolor='b',
    #         markeredgecolor='b',color='r',markersize = 11.0)
    x = train['x'].to_numpy()
    y = train['y'].to_numpy()
    c = test['label'].to_numpy() - 1
    fig, ax = plt.subplots()

    for _x, _y, _c in zip(x, y, c):
        ax.scatter(_x, _y, c=colors[_c], s=60)
        # ax.annotate(_c, (_x, _y))

    x = test['x'].to_numpy()
    y = test['y'].to_numpy()
    c = test['label'].to_numpy() - 1

    for _x, _y, _c in zip(x, y, c):
        ax.scatter(_x, _y, c=colors[_c], s=4)
        # ax.annotate(_c, (_x, _y))

    plt.show()
    # parallel_label_propagation(F, P, labeled_number)


def expirements():
    filenames = ['data/Aggregation.txt', 'data/Compound.txt', 'data/D31.txt',
                 'data/flame.txt', 'data/jain.txt',
                 'data/pathbased.txt', 'data/R15.txt', 'data/spiral.txt']
    # filenames = ['data/Aggregation.txt']
    for f in filenames:
        asd(f, 0.6)


def asd(filename, test_size=0.7):
    colors = ['black', 'red', 'cyan', 'navy', 'tomato', 'pink', 'orange']
    for name, hex in matplotlib.colors.cnames.items():
        if name not in colors:
            colors.append(name)
    colors.pop(0)

    df = pandas.read_csv(filename, sep='\t', names=['x', 'y', 'label'])
    # 来个坐标点图
    # plt.plot(train['x'].to_numpy(),train['y'].to_numpy(),linestyle='None',  markerfacecolor='b',
    #         markeredgecolor='b',color='r',markersize = 11.0)

    fig, ax = plt.subplots(1, 3, sharey=True)
    x = df['x'].to_numpy()
    y = df['y'].to_numpy()
    c = df['label'].to_numpy() - 1
    for _x, _y, _c in zip(x, y, c):
        ax[0].scatter(_x, _y, c=colors[_c], s=60)
        # ax.annotate(_c, (_x, _y))
    ax[0].set_title(' Origin')
    # parallel_label_propagation(F, P, labeled_number)

    train, test = train_test_split(df, test_size=test_size)
    test = test.assign(label=0)
    labeled_number = train.shape[0]
    print('label_number', labeled_number)
    F, P = construct_matrix_F_P(labeled_data=train, unlabeled_data=test)
    final_f = speedup_label_propagation(F, P, labeled_number)

    # UxC的一个矩阵
    labels = np.argmax(final_f, axis=1) + 1
    test_row = test.index
    for i, label in enumerate(labels):
        test.at[test_row[i], 'label'] = label
    # 显示训练
    x = train['x'].to_numpy()
    y = train['y'].to_numpy()
    c = train['label'].to_numpy() - 1
    ax[1].set_title('After lp')
    for _x, _y, _c in zip(x, y, c):
        ax[1].scatter(_x, _y, c=colors[_c], s=60)

    x = test['x'].to_numpy()
    y = test['y'].to_numpy()
    c = test['label'].to_numpy() - 1
    for _x, _y, _c in zip(x, y, c):
        ax[1].scatter(_x, _y, c=colors[_c], s=60)

    x = train['x'].to_numpy()
    y = train['y'].to_numpy()
    c = train['label'].to_numpy() - 1
    ax[2].set_title('After lp smaller test point')
    for _x, _y, _c in zip(x, y, c):
        ax[2].scatter(_x, _y, c='black', s=60)

    x = test['x'].to_numpy()
    y = test['y'].to_numpy()
    c = test['label'].to_numpy() - 1
    for _x, _y, _c in zip(x, y, c):
        ax[2].scatter(_x, _y, c=colors[_c], s=10)

    fig.savefig(filename + str(test_size) + '.png')

    # fig.show()


if __name__ == '__main__':
    import os

    # os.environ["OMP_NUM_THREADS"] = "1"
    # os.environ["MKL_NUM_THREADS"] = "1"
    # os.environ["OPENBLAS_NUM_THREADS"] = "1"
    # os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    # os.environ["NUMEXPR_NUM_THREADS"] = "1"
    # main()
    # 标签传播花费时间:0.607s,传播轮次385
    # 标签传播花费时间:0.615s,传播轮次395
    # main2()
    # expirements()
    import os

    # os.environ["OMP_NUM_THREADS"] = "1"
    # os.environ["MKL_NUM_THREADS"] = "1"
    # os.environ["OPENBLAS_NUM_THREADS"] = "1"
    # os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    # os.environ["NUMEXPR_NUM_THREADS"] = "1"

    filename = 'data/D31.txt'
    df = read_shape_sets(filename)
    # test_size 70% 因为 我认为 标签传播算法只需要一小部分有标签的数据，其它数据可以通过传播得到标签。
    train, test = train_test_split(df, test_size=0.7)

    test = test.assign(label=0)
    labeled_number = train.shape[0]
    print('label_number', labeled_number)
    F, P = construct_matrix_F_P(labeled_data=train, unlabeled_data=test)
    # speedup_label_propagation(F, P, labeled_number,each_i=5)
    parallel_label_propagation(F, P, labeled_number, each_i=5)
