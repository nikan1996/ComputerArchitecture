import csv
import gc
import math
import pickle
import sys
from concurrent.futures._base import FIRST_COMPLETED

import numpy as np
import pandas as pd
import time

from numba import njit

from patch_shared_memory import _SharedMemory_init

from concurrent.futures import ProcessPoolExecutor, wait
import multiprocessing
from multiprocessing import shared_memory

# shared_memory.SharedMemory.__init__ = _SharedMemory_init

from multiprocessing import Process
import os

import multiprocessing, multiprocessing.shared_memory


def read_rating_as_matrix(filename):
    """
    获取ratings.csv里的评分表，输出用户对电影的评分矩阵。
    矩阵格式：

          M1 M2 M3
    User1 2  3   4
    User2 2  3   4
    User3 1  2   3
    :return:
    """
    full_data = pd.read_csv(filename, dtype={
        'userId': np.int32, 'movieId': np.int, 'rating': np.float, 'timestamp': np.int
    })
    # print('前五位数据:', full_data.head())
    # print('总数据量:', len(full_data))
    user_length = full_data['userId'].nunique()
    # print('user数量:', user_length)
    new_data = pd.DataFrame()
    d = [[]]
    movie_data = pd.read_csv('ml-latest-small/movies.csv')
    # print(movie_data['movieId'].head())
    movie_length = movie_data['movieId'].nunique()
    # print('movie数量:', movie_length)

    movie_id_map = {}
    index_movie_id_map = {}
    for index, data in movie_data.iterrows():
        if int(data['movieId']) not in movie_id_map:
            movie_id_map[int(data['movieId'])] = index
        index_movie_id_map[index] = int(data['movieId'])
    # 构造矩阵
    empty_matrix = np.empty((user_length, movie_length))
    empty_matrix.fill(np.nan)
    user_id_map = {}
    index_user_id_map = {}
    for index, data in full_data.iterrows():
        # print(data['userId'],movie_id_map[data['movieId']])
        # userId和矩阵index偏移是1，因为userId从1开始算
        # movieId需要额外映射来获取真实movieId。
        if int(data['userId']) not in user_id_map:
            user_id_map[int(data['userId'])] = int(data['userId']) - 1
            index_user_id_map[int(data['userId']) - 1] = int(data['userId'])

        empty_matrix[user_id_map[int(data['userId'])], movie_id_map[int(data['movieId'])]] = data['rating']
    return empty_matrix, user_id_map, movie_id_map, index_user_id_map, index_movie_id_map


@njit
def init_pu_qi_b1_b2_u(matrix, factor_num):
    # pu: user_number x k
    user_number, movie_number = matrix.shape
    u = np.nanmean(matrix)
    print('users:', user_number, 'movies:', movie_number, 'k:', factor_num)
    # self.pu = np.random.rand(user_number, self.factor_num)
    # self.qi = np.random.rand(movie_number, self.factor_num)

    # random_value = 0.24
    pu = np.random.normal(0, .1, (user_number, factor_num))
    qi = np.random.normal(0, .1, (movie_number, factor_num))
    # pu = np.random.rand(user_number, factor_num)
    # qi = np.random.rand(movie_number, factor_num)

    # pu = np.random.normal(0, random_value, (user_number, factor_num))
    # qi = np.random.normal(-random_value, random_value, (movie_number, factor_num))
    b1 = np.zeros(user_number)
    b2 = np.zeros(movie_number)

    for u_index in range(user_number):
        value = np.nanmean(matrix[u_index])
        if np.isnan(value):
            value = u
            # pu[u_index] = np.mean(pu)
        b1[u_index] = value - u

    for m_index in range(movie_number):
        value = np.nanmean(matrix[:, m_index])
        if np.isnan(value):
            value = u
            # qi[m_index] = np.mean(qi)

        b2[m_index] = value - u
    return pu, qi, b1, b2, u


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


@njit
def _run_block(m, pu, qi, b1, b2, u, factor_num, lr, reg, start_row, start_column):
    num = 0
    loss = 0

    for u_i, row in enumerate(m):
        for m_i, rating in enumerate(row):
            if np.isnan(rating):
                continue
            _user_index = u_i
            _movie_index = m_i

            rating = m[_user_index, _movie_index]
            # 梯度下降
            num += 1
            user_index = start_row + _user_index
            movie_index = start_column + _movie_index
            b1i = b1[user_index]
            b2j = b2[movie_index]
            pred = u + b1i + b2j
            # 用循环可以加速numba njit 比dot快
            for factor in range(factor_num):
                pred += pu[user_index, factor] * qi[movie_index, factor]
            # piqj = np.dot(pu[user_index],qi[movie_index])

            eij = rating - pred
            # loss在这里计算略微有点耗时
            loss += (eij) ** 2 + \
                    reg * (
                            b1i ** 2 + b2j ** 2 +
                            np.linalg.norm(pu[user_index]) ** 2 +
                            np.linalg.norm(qi[movie_index] ** 2)
                    )

            b1[user_index] = b1i + lr * (eij - reg * b1i)
            b2[movie_index] = b2j + lr * (eij - reg * b2j)
            # puf = pu[user_index]
            # qif = qi[movie_index]
            # pu[user_index] +=lr * (eij * qif - reg * puf)
            # qi[movie_index]+= lr * (eij * puf - reg * qif)
            # 展开for循环比直接用dot快一些
            for f in range(factor_num):
                puf = pu[user_index, f]
                qif = qi[movie_index, f]
                pu[user_index, f] += lr * (eij * qif - reg * puf)
                qi[movie_index, f] += lr * (eij * puf - reg * qif)

    loss = loss / num
    # loss = 1

    return loss


def block_gradient_descent(b, block, factor_num, lr, reg, pu, qi, b1, b2, u):
    # print(f'开始block:{block["id"]}')
    start = time.time()
    start_row, start_column = block['start']
    end_row, end_column = block['end']
    matrix, shm_matrix = nparray_from_shm(*b)

    pu, shm_pu = nparray_from_shm(*pu)
    qi, shm_qi = nparray_from_shm(*qi)
    b1, shm_b1 = nparray_from_shm(*b1)
    b2, shm_b2 = nparray_from_shm(*b2)
    m = matrix[start_row:end_row + 1, start_column:end_column + 1]

    # 计算Loss并梯度下降
    # start = time.time()
    loss = _run_block(m, pu, qi, b1, b2, u, factor_num, lr, reg, start_row, start_column)

    shms = [shm_matrix, shm_pu, shm_qi, shm_b1, shm_b2]

    shm_close(shms)

    return loss, block['id']


def paralle_fit(matrix, epochs=1, lr=0.005, reg=0.02, factor_num=50, threshold=0.45, thread_num=1, save_model_flag=True,
                model_name=None):
    worker_pool = ProcessPoolExecutor(max_workers=thread_num)

    pu, qi, b1, b2, u = init_pu_qi_b1_b2_u(matrix, factor_num)
    blocks = calculate_patitioning_size_for_each_block(*matrix.shape, thread_num)
    matrix_shm_args = create_shm_nparray(matrix)
    pu_shm_args = create_shm_nparray(pu)
    qi_shm_args = create_shm_nparray(qi)
    b1_shm_args = create_shm_nparray(b1)
    b2_shm_args = create_shm_nparray(b2)
    epoch_times = []
    for epoch in range(epochs + 1):
        # numpy matrix
        start_epoch = time.time()
        new_blocks = blocks.copy()
        futures = []
        total_loss = 0
        locked_regions = {}
        used_blocks = set()
        while new_blocks or futures:
            start_round = time.time()
            for block in new_blocks:
                locked_rows = []
                locked_columns = []
                for region in locked_regions:
                    locked_rows.append(region[0])
                    locked_columns.append(region[0])
                block_region = block['id']
                if block_region[0] not in locked_rows and block_region[1] not in locked_columns:
                    future = worker_pool.submit(block_gradient_descent, matrix_shm_args[:3], block, factor_num, lr, reg,
                                                pu_shm_args[:3], qi_shm_args[:3], b1_shm_args[:3], b2_shm_args[:3], u)
                    futures.append(future)
                    locked_regions[block['id']] = 1
                    # print('lock:', block_region)
                    used_blocks.add(block['id'])

            if new_blocks and len(used_blocks) > 0:
                new_blocks = [block for block in new_blocks if block['id'] not in used_blocks]
                used_blocks = set()
                # print(len(used_blocks))
            # if not new_blocks:
            #     print('new block 为空,epoch结束')
            start_round = time.time()
            if futures:
                wait(futures, return_when=FIRST_COMPLETED)

                f = []
                done_futures = []
                for i in futures:
                    if i.done():
                        done_futures.append(i)
                    else:
                        f.append(i)
                futures = f
                if done_futures:
                    for f in done_futures:
                        loss, block_region = f.result()
                        total_loss += loss
                        locked_regions.pop(block_region)
                        # print('pop:',block_region)

        total_loss /= (thread_num ** 2)
        epoch_time = time.time() - start_epoch
        epoch_times.append(epoch_time)
        print('epoch', epoch, f'elapsed time:{epoch_time:0.3f}s,loss:{total_loss}')
        if total_loss <= threshold:
            print('loss收敛，提前退出训练')
            break
        # save_time = time.time()
        if save_model_flag and epoch % 50 == 0:
            if model_name:
                save_model(pu_shm_args[-1], qi_shm_args[-1], b1_shm_args[-1], b2_shm_args[-1], u, name=model_name)
            else:
                save_model(pu_shm_args[-1], qi_shm_args[-1], b1_shm_args[-1], b2_shm_args[-1], u)
        # print(f'save model time:{time.time() - save_time}s')
    shm_close([matrix_shm_args[-2],
               pu_shm_args[-2],
               qi_shm_args[-2],
               b1_shm_args[-2],
               b2_shm_args[-2]])
    del matrix_shm_args
    del pu_shm_args
    del qi_shm_args
    del b1_shm_args
    del b2_shm_args

    return sum(epoch_times[1:]) / (epochs - 1)


def test_rmse(matrix, user_id_map, movie_id_map, index_user_id_map, index_movie_id_map, pu, qi, b1, b2, u):
    '''
    test the model and return the value of rmse
    '''
    rmse = 0.0
    num = 0
    user_ids, movie_ids = np.asarray(~np.isnan(matrix)).nonzero()
    for user_index, movie_index in zip(user_ids, movie_ids):
        rating = matrix[user_index, movie_index]
        num += 1
        user_id = index_user_id_map[user_index]
        movie_id = index_movie_id_map[movie_index]
        true_movie_index = movie_id_map.get(movie_id, None)
        true_user_index = user_id_map.get(user_id, None)
        # print('用户id:',user_id,'电影id:',movie_id,f'真实id:{true_user_index},{true_movie_index}')
        # print(np.dot(pu[true_user_index], qi[true_movie_index]))
        pr = u + b1[true_user_index] + b2[true_movie_index] + np.dot(pu[true_user_index], qi[true_movie_index])
        rmse += (rating - pr) ** 2

    rmse = (rmse / num) ** 0.5
    return rmse


def calculate_patitioning_size_for_each_block(matrix_row, matrix_column, thread_num):
    # 16个线程， ceil(610/16) = 39 ,每个块处理39个用户,最后一个块有25个用户
    # 16个线程， ceil(9742/16) = 609, 每个块有609个电影，最后一个块607个电影
    # 把矩阵分成当成16*16的块进行并行化运算
    # 理论上每个核只需要处理16个块就可以完成程序

    block_row = math.ceil(matrix_row / thread_num)
    reminder_row = matrix_row - block_row * (thread_num - 1)
    block_column = math.ceil(matrix_column / thread_num)
    reminder_column = matrix_column - block_column * (thread_num - 1)
    blocks = []

    row = 0
    column = 0

    for i in range(1, thread_num + 1):
        final_row_step = block_row
        if i == thread_num:
            final_row_step = reminder_row

        for j in range(1, thread_num + 1):
            start = (block_row * (i - 1), block_column * (j - 1))

            if j == thread_num:
                # 增加一个 reminder_column的长度
                end = (start[0] + final_row_step - 1, start[1] + reminder_column - 1)
            else:
                # 增加一个 block_column的长度
                end = (start[0] + final_row_step - 1, start[1] + block_column - 1)
            id = (i, j)
            block = {
                'id': id,
                'start': start,
                'end': end
            }
            blocks.append(block)
    new_block = []
    for i in range(0, thread_num):
        for j in range(0, thread_num):
            index = (i + j) % thread_num + j * thread_num
            _block = blocks[index]
            new_block.append(_block)
            # print(index)

    assert len(blocks) == thread_num * thread_num
    return new_block


def save_model(pu, qi, b1, b2, u, name='matrix_factorization.pkl'):
    '''
    save the model
    '''
    data_list = [pu, qi, b1, b2, u]
    f = open(name, 'wb')
    pickle.dump(data_list, f)
    f.close()


def load_model(name='matrix_factorization.pkl'):
    '''
    reload the model from local disk
    '''

    f = open(name, 'rb')
    data_list = pickle.load(f)
    f.close()
    pu, qi, b1, b2, u = data_list
    return pu, qi, b1, b2, u


def main():
    thread_num = 6
    print('thread_num:', thread_num)

    start = time.time()
    matrix, user_id_map, movie_id_map, index_user_id_map, index_movie_id_map = read_rating_as_matrix(
        'ml-latest-small/ml_train2.csv')
    # print(matrix)
    # test_matrix, test_user_id_map, test_movie_id_map, test_index_user_id_map, test_index_movie_id_map = read_rating_as_matrix(
    #     'ml-latest-small/ml_test2.csv')

    # print('读取rating.csv耗时：' + str(int((time.time() - start) * 1000)) + 'ms')
    paralle_fit(matrix, epochs=10000, factor_num=100, threshold=0.05, thread_num=thread_num, save_model_flag=True)


def rmse():
    matrix, user_id_map, movie_id_map, index_user_id_map, index_movie_id_map = read_rating_as_matrix(
        'ml-latest-small/ml_train2.csv')
    test_matrix, test_user_id_map, test_movie_id_map, test_index_user_id_map, test_index_movie_id_map = read_rating_as_matrix(
        'ml-latest-small/ml_test2.csv')
    pu, qi, b1, b2, u = load_model()
    rmse_train = test_rmse(matrix, user_id_map, movie_id_map, index_user_id_map, index_movie_id_map, pu, qi, b1, b2, u)
    print('rmse_train:', rmse_train)
    rmse_test = test_rmse(test_matrix, user_id_map, movie_id_map, test_index_user_id_map, test_index_movie_id_map, pu,
                          qi, b1, b2, u)
    print('rmse_test:', rmse_test)


import matplotlib.pyplot as plt


def experiment1():
    # 作图函数
    matrix, user_id_map, movie_id_map, index_user_id_map, index_movie_id_map = read_rating_as_matrix(
        'ml-latest-small/ml_train2.csv')
    test_matrix, test_user_id_map, test_movie_id_map, test_index_user_id_map, test_index_movie_id_map = read_rating_as_matrix(
        'ml-latest-small/ml_test2.csv')
    fig, ax = plt.subplots()  # Create a figure containing a single axes.
    ax.set_title('factor:100,total epoch:50')
    ax.set_ylabel('Each Epoch Time(s)')
    ax.set_xlabel('Thread Number')
    x = []
    y = []
    for i in range(1, 11):
        t = paralle_fit(matrix, epochs=50, factor_num=100, threshold=0.05, thread_num=i, save_model_flag=False)
        x.append(i)
        y.append(t)
    #     # print((i,time))
    ax.set_xticks(x)
    ax.plot(x, y)
    fig.show()
    # pu, qi, b1, b2, u = load_model()

    # rmse_test = test_rmse(test_matrix, user_id_map, movie_id_map, test_index_user_id_map, test_index_movie_id_map, pu,
    #                       qi, b1, b2, u)

def save_exp2_model():
    matrix, user_id_map, movie_id_map, index_user_id_map, index_movie_id_map = read_rating_as_matrix(
            'ml-latest-small/ml_train2.csv')
    for i in range(50, 500, 50):
        t = paralle_fit(matrix, epochs=300, factor_num=i, threshold=0.05, thread_num=4, save_model_flag=True,
                           model_name=f'models/factor{i}.pkl')
def experiment2():
    # 作图函数
    # 保存model
    #  matrix, user_id_map, movie_id_map, index_user_id_map, index_movie_id_map = read_rating_as_matrix(
    #         'ml-latest-small/ml_train2.csv')
    #     for i in range(50, 500, 50):
    #         t = paralle_fit(matrix, epochs=300, factor_num=i, threshold=0.05, thread_num=6, save_model_flag=True,
    #                            model_name=f'models/factor{i}.pkl')
    matrix, user_id_map, movie_id_map, index_user_id_map, index_movie_id_map = read_rating_as_matrix(
        'ml-latest-small/ml_train2.csv')
    test_matrix, test_user_id_map, test_movie_id_map, test_index_user_id_map, test_index_movie_id_map = read_rating_as_matrix(
        'ml-latest-small/ml_test2.csv')
    fig, ax = plt.subplots()  # Create a figure containing a single axes.
    ax.set_title('RMSE with 300 epochs')
    ax.set_ylabel('RMSE')
    ax.set_xlabel('Factor Number')
    x = []
    y = []
    for i in range(50, 500, 50):
        x.append(i)
        pu, qi, b1, b2, u = load_model(f'models/factor{i}.pkl')
        print(pu.nbytes)
        rmse_test = test_rmse(test_matrix, user_id_map, movie_id_map, test_index_user_id_map, test_index_movie_id_map,
                              pu,qi, b1, b2, u)
        print(rmse_test)
        y.append(rmse_test)

    ax.set_xticks(x)
    ax.plot(x, y)
    fig.show()
    # pu, qi, b1, b2, u = load_model()

    # rmse_test = test_rmse(test_matrix, user_id_map, movie_id_map, test_index_user_id_map, test_index_movie_id_map, pu,
    #                       qi, b1, b2, u)

def exp_speed_up():
    matrix, user_id_map, movie_id_map, index_user_id_map, index_movie_id_map = read_rating_as_matrix(
        'ml-latest-small/ml_train2.csv')
    test_matrix, test_user_id_map, test_movie_id_map, test_index_user_id_map, test_index_movie_id_map = read_rating_as_matrix(
        'ml-latest-small/ml_test2.csv')
    t1 = paralle_fit(matrix, epochs=100, factor_num=100, threshold=0.05, thread_num=1, save_model_flag=False)
    fig, ax = plt.subplots()  # Create a figure containing a single axes.
    ax.set_title('Speedup')
    ax.set_ylabel('Ratio')
    ax.set_xlabel('Threadnumber')
    x = [1]
    y = [1]
    for i in range(2,10):
        x.append(i)
        t = paralle_fit(matrix, epochs=100, factor_num=100, threshold=0.05, thread_num=i, save_model_flag=False)
        y.append(t1/t)
    ax.set_xticks(x)
    ax.plot(x, y)
    fig.show()

if __name__ == '__main__':
    exp_speed_up()
    # experiment2()
    # experiment1()

    # rmse()
    # matrix, user_id_map, movie_id_map, index_user_id_map, index_movie_id_map = read_rating_as_matrix(
    #     'ml-latest-small/ml_train2.csv')
    # calculate_patitioning_size_for_each_block(610,9742,16)
    # calculate_patitioning_size_for_each_block(610,9742,4)
