# 作业：标签传播的实现及其并行化实现

<center>2009853Z-II20-0017 倪侃</center>

[toc]



标签传播是一个半监督学习算法。假如我有一批数据，其中一小部分有标签，而另外一大部分没有标签。我可以利用标签传播让没有标签的数据带上标签。那么LP算法的核心思想就是：传播。就像COVID-19一样，变种很多，传播随着人口密度和戴口罩比例决定传播概率。

LP算法通过构造概率转移矩阵来传播标签的概率。通过多次传播以后，没有标签的数据会被它周围的带标签的数据传播到相对应的矩阵。 这里标签不是直接传播的，而是通过概率传播，一个迭代只传播一点点概率，通过多次迭代，概率累加，最终数据的标签取决于概率最大的那个标签。

那么总体来说算法达到收敛的时候，传播就可以完成，什么时候收敛呢？ 在F矩阵不变的时候，那么就收敛了。（即F=PF的迭代是可以收敛的，在概率转移的时候最终总能达到一个不变的状态。）

## 实验

采用 http://cs.joensuu.fi/sipu/datasets/ 的 Shape sets 数据进行验证分类验证，包括 http://cs.joensuu.fi/sipu/datasets/Aggregation.txt 等 8 个数据。

我切分数据集  70%为测试集，即无label的数据，30%为训练集，即有label的数据。

下面每个图片里都有三张子图，第一张是原数据绘制结果，第二张是标签传播后的结果绘制，第三张是黑点为训练集、其它小点为测试集的传播标签结果，第三张图片体现了第二张图得传播过程。

Aggregation.txt

![Aggregation.txt](C:\Users\admin\PycharmProjects\ComputerArchitecture\label_propagation\data\Aggregation.txt.png)

Compound.txt

 ![Compound.txt](C:\Users\admin\PycharmProjects\ComputerArchitecture\label_propagation\data\Compound.txt.png)

D31.txt

 ![D31.txt](C:\Users\admin\PycharmProjects\ComputerArchitecture\label_propagation\data\D31.txt.png)

flame.txt

 ![flame.txt](C:\Users\admin\PycharmProjects\ComputerArchitecture\label_propagation\data\flame.txt.png)

jain.txt

 ![jain.txt](C:\Users\admin\PycharmProjects\ComputerArchitecture\label_propagation\data\jain.txt.png)

pathbased.txt

 ![pathbased.txt](C:\Users\admin\PycharmProjects\ComputerArchitecture\label_propagation\data\pathbased.txt.png)

R15.txt

 ![R15.txt](C:\Users\admin\PycharmProjects\ComputerArchitecture\label_propagation\data\R15.txt.png)

spiral.txt

 ![spiral.txt](C:\Users\admin\PycharmProjects\ComputerArchitecture\label_propagation\data\spiral.txt.png)

所有的结果如图所示。可以发现 标签传播的效果还是很好的，经过一定数量的迭代后就会收敛（几百次到几千次不等，取决于散点的密度和数量）。传播后的存在边界上有几个点和原始值标签不一样，属于可理解范围内。

 

## 代码逻辑

参考作业中的公式

 ![image-20210406133948806](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20210406133948806.png)

Fu = puu*fu + pul*yl 是一个逐步迭代的方法，我最开始用了这个公式。

后来我采用了公式(1)进行迭代。遇到了几个问题。

问题1：如何确定公式中n的取值？

n是一个可选值，通过逐步迭代法  我发现大部分收敛的次数在几百次左右，当n比较大的时候，puu(n)运算会比较慢，原因可能是cache需要puu*n的大小，而超过我的cpu l3 cache的时候，矩阵运算就需要频繁从内存中取值了，大大降低速度。

问题2： Puu需要求得2次方到n次方的值，运算复杂度较高？ 

用缓存，缓存每一次方计算的结果， 比如缓存了 1,2,3,4次方的puu， 那么算puu 5次方的时候 直接 用缓存的 puu4和puu做一次乘积就可以得到puu5， 极大加快了运算效率。

问题3： ![image-20210406134232462](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20210406134232462.png)式子放在哪计算？

因为迭代过程只会更新fu，那么我们只需要在最开始计算一次该式子即可。无需在迭代中重复计算。

运行结果展示：

![image-20210406135446239](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20210406135446239.png)



## 并行化

### PYTHON多进程

对公式1加速，我考虑到几个点

1. l1/l2/l3 cache的利用

   如何更好地切割矩阵适配cache 。

2. 分割收敛流程。

   在标签传播中就是不停地对fu进行迭代，那么我能不能 分别收敛fu？ 如果我有n个进程，我把fu一分为n，每个进程收敛属于它自己的那一部分，在最后汇聚到主进程中。我就对所有的fu收敛完毕了。

   block分割方法： 

   1. 横向分割
   2. 纵向分割

   多进程这一块主要是为了利用多CPU而做的分割。

![image-20210407005412238](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20210407005412238.png)

fu如果做二等分可以是 [Y1;Y2]， 那么我在实现的时候可以让 Y1 = ... ,Y2 = ... 做并行化更新。

代码使用了python来写（windows10平台）。

由于python多线程受限于GIL锁的困扰导致无法利用。

所以考虑使用多进程来编写。编写的时候发现python多进程也是一件很复杂的事情。

遇到的问题：

1. 多进程初始化开销很大
2. 多进程通信成本很大

我的几个解决方案：

1. 多进程预热，提前创建好进程
2. 使用sharedmemory存放numpy 数组，其中windows python的shared memory存在内存泄露的bug，我自己做了补丁（参考自https://bugs.python.org/issue40882）

#### 多进程简单实验结果

由于Python多进程开销较大，简单地实验了二分并行的情况

数据 data/D31.txt是数据量最大的一个文本，适合作为测试加速比的样本。

![image-20210407011356285](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20210407011356285.png)

截图中的加速比：1.41。加速比在1.4-1.5波动。这个加速比包括了进程通信的开销。

效果不是特别好，但是可以发现有一定的加速。我还输出了本机的l2 l3 cache看了一下，这个数据量是能读入cache的。其它的数据比D31还要小得多，但是跑得有点快，可能还没进程通信的开销大。

怎么选择并行粒度：

原则：并行后的通信开销+分割任务开销+并行执行时间<串行执行时间。 作tiling的时候

### GEMM

如何选择Blocked Matrix Multiplication的Block大小？让Block运算块适配CPU的Cache，减少Cache Miss。



我进行了矩阵乘法的单独测试。 可以试一下在Cache Miss的时候 使用GEMM计算的加速比。

![Matrix_cache](C:\Users\admin\PycharmProjects\ComputerArchitecture\Matrix_cache.png)

在进行矩阵乘AXB的时候，我发现两个矩阵大小和为64KB（A =32KiB B=32Kib) 的时候运算速度明显要低于63.5和64.5的。

原因：已知我的电脑是zen3 ，具有32KiB的l1 cache 

![image-20210407175819352](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20210407175819352.png)

当一个矩阵小于cache size的时候，它可以直接存储进入l1 cache，当矩阵大于cache size的时候 它可以存入l2 cache，如果矩阵正好等于cache size ， 两个矩阵会不停轮流被读入l1 cache，造成开销。(为什么小于的时候不会被轮流读入呢？ 猜测是zen3 cache控制策略问题)

tvm有教怎么优化tvm的gemm 从而达到Numpy的gemm性能

https://tvm.apache.org/docs/tutorials/optimize/opt_gemm.html#sphx-glr-tutorials-optimize-opt-gemm-py

https://zhuanlan.zhihu.com/p/75203171

https://github.com/flame/how-to-optimize-gemm/wiki

这几篇文章写得很好，学到了很多的优化技巧。

![image-20210408142031404](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20210408142031404.png)

牛人优化的方案很多，我还在学习阶段。

![image-20210408150306456](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20210408150306456.png)

上图code通过把最内层的k放到最外层，用外积的方法做矩阵乘积，比内积能更好地利用cache。

接下来简单实现一下 Blocked Matrix Multiplication的算法。

实现以后对200-1000大小的矩阵做了一次测试，可以看到分块以后计算乘积确实变快了。

随着矩阵大小增大，加速效果越发地明显。

实际block size怎么选择？可以使用auto tun 地技术，动态调整参数，直到调出最好的参数为止。

![image-20210408165440515](C:\Users\admin\AppData\Roaming\Typora\typora-user-images\image-20210408165440515.png)

![gemm test](C:\Users\admin\PycharmProjects\ComputerArchitecture\gemm test.png)



## 源码

见附件。