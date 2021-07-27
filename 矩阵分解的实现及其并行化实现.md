# 作业：矩阵分解的实现及其并行化实现

<center>2009853Z-II20-0017 倪侃</center>

[toc]

矩阵分解做推荐系统是来自netflix比赛里的一个方法“https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf“，通过矩阵分解的方法，可以取得比其他方法更好的效果。矩阵分解的思路来源于SVD分解。

## 简介

笔记本 CPU:AMD 4800U(8核16线程，l2 cache 4096k,l3 cache 8192k) 内存(16GB DDR4 2666MHZ) Windows10+Python3.8.5

采用了movielens的最小的数据集（10k），对ratings进行预测。

最终并行版本可以做到一个迭代在40毫秒左右。

优化技巧：

1. numba jit加速。
2. shared memory减少并行通信开销。
3. 尽量让矩阵运算在cache之内。
4. 减少进程的空闲时间。

![image-20210407020820377](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20210407020820377.png)



## 代码逻辑

![image-20210407014655259](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20210407014655259.png)

![image-20210407014708903](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20210407014708903.png)

整块代码做的事情：

1.读取评分表 

2.构建用户，电影偏置矩阵，用户隐因子矩阵，电影隐因子矩阵

3.梯度下降使得损失函数收敛。

读取评分表中：评分表的 电影ID和矩阵索引需要做一个映射 保证更新矩阵的正确性。

梯度下降可以设置一个适当的学习率，正则项需要包含在loss内。

未优化的版本大概一个epoch迭代需要5秒左右。隐因子数量基本不影响速度，略微影响rmse ， 在netflix那篇论文有一张图![image-20210407020322750](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20210407020322750.png)

说明了RMSE和因子数量的关系，可以看到因子数量给个50到200左右就差不多了。如果给得多，增加运算量，实际效果也不一定好。

## 并行化实现

并行化参考了DSGD的棋盘分割block方法。

![image-20210407014910045](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20210407014910045.png)

为什么要分割？ 避免数据同步的问题 ， 通过分块，保证 BU BI P Q 矩阵的局部更新不会出现冲突，

无需进入竞争条件（race condition)。所以我就采用了这个方法。如果我的线程是16个，那么我就把整个矩阵分成16*16的格子，进行排列后，每一轮我都会并行执行16个格子的矩阵做梯度下降。

另外我做了优化，通过程序实现的更细粒度的行列锁，能让线程空闲时间减少。提高线程的利用率。

当然Python实现的程序存在进程开销和进程通信问题。

我用了和标签传播一样的方法：1. 预热进程 2.使用shared memory传递numpy数组。这里由于windows python的bug导致 内存泄露，找了个链接修复了一下python的问题。

## 运行结果

RMSE总体能到0.88-0.9，训练集测试集70%-30%分开。

![image-20210407021816591](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20210407021816591.png)

![RMSE with 300 epochs](C:\Users\admin\PycharmProjects\ComputerArchitecture\sgd\RMSE with 300 epochs.png)

因子数量越多和RMSE越小，但也不是因子数量越多越好的，上面的图训练了300个epochs.

![threadnumber-epoch_time](C:\Users\admin\PycharmProjects\ComputerArchitecture\sgd\threadnumber-epoch_time.png)

线程数量和每个epoch花费时间的图，大概可以看到在我的电脑上Thread Number为6的时候 迭代速度是最快的，只要40毫秒多一些。 理论上如果没有进程通信的开销，计算是核越多速度越快的。核越多，同步的开销越大。

### 并行加速比

![image-20210407033115609](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20210407033115609.png)

可以看到大概最快是2.2倍左右的样子。

这是已经优化的情况。

如果基准是最初版本的话，加速比大概能达到100倍。

## 源码

见压缩包内文件夹。