# Lecture 4：自注意力机制（Self Attention）

#### Lecture 4：自注意力机制（Self-Attention） <a href="#lecture4-zi-zhu-yi-li-ji-zhi-selfattention" id="lecture4-zi-zhu-yi-li-ji-zhi-selfattention"></a>

> Lectured by HUNG-YI LEE (李宏毅)
>
> Recorded by Yusheng zhao（[yszhao0717@gmail.com](mailto:yszhao0717@gmail.com)）

***

**引入**

> * 致力于解决输入为一组向量的深度学习任务。
>
> <img src="https://s1.328888.xyz/2022/05/03/hJGl4.png" alt="" data-size="original">
>
> 例如👇）——作业一自然语言处理
>
> <img src="https://s1.328888.xyz/2022/05/03/hJ9PT.png" alt="" data-size="original">
>
> 作业二——声音讯息👇
>
> <img src="https://s1.328888.xyz/2022/05/03/hJLh2.png" alt="" data-size="original">
>
> 作业三——图👇，每个节点都可以是一个向量，包含了人物的社交信息
>
> <img src="https://s1.328888.xyz/2022/05/03/hJXYM.png" alt="" data-size="original">
>
> 分子也可以看作“Graph”：（这里化学元素每个原子用one-hot表示）
>
> <img src="https://s1.328888.xyz/2022/05/03/hJfM7.png" alt="" data-size="original">
>
> ***
>
> * 输出的形式多样：
>   *   每个vector各自输出一个label：比方说文字处理中的_词性标注_、_语音音素分类_、_社交网络标签_
>
>       <img src="https://s1.328888.xyz/2022/05/03/hJ8FC.png" alt="" data-size="original">
>   *   一整个sequence输出一个label：比方说：_情感分析_、_语者辨认_、_给一个graph输出一个label_
>
>       <img src="https://s1.328888.xyz/2022/05/03/hJ2pZ.png" alt="" data-size="original">
>   *   模型自己决定输出label的数量——_**seq2seq**_任务，例如：_翻译_、_完整的语音辩识_
>
>       <img src="https://s1.328888.xyz/2022/05/03/hJhGg.png" alt="" data-size="original">

***

**模型一：Sequence Labeling**

即上文输入和输出数目一样多的情形。

注意到，对于单个输入vector要关注它的上下文信息。但是，以某个vector为中心，为了cover整个sequence，开一个一定尺寸的窗口输入全连接层中——参数巨多而且有可能overfitting。**Self-attention**被用来化解这个困难。

![](https://s1.328888.xyz/2022/05/03/hJ4l1.png)

> FC = Fully Connected Layer

Self-attention考虑到整个sequence的信息，有多少输入self-attention就有多少输出。模型中经常使用FC和Self-attention交替使用。[Attention is all you need](https://arxiv.org/abs/1706.0376212)

![](https://s1.328888.xyz/2022/05/03/hi2oe.png)

Self-Attention的每个输出都考虑了所有的输入。以为例：

*   首先根据找到跟相似的向量（找到整个sequence里面哪些部分是重要的哪些部分和同一个level、决定回归结果数值或者来分类结果class所需要的信息）；每一个向量和关联性的值用表示

    <img src="https://s1.328888.xyz/2022/05/03/hirmt.png" alt="" data-size="original">

    这个Self-attention的module怎样自动决定两个向量的相关性？以下给出计算两个向量相关性的模组。

    <img src="https://s1.328888.xyz/2022/05/03/hiiH1.png" alt="" data-size="original">

    上述比较常见的做法——_**Dot-product**_：输入的这两个向量（需要计算关联度）分别乘上两个不同的矩阵和，得到两个向量和，然后这两个向量做element-wise相乘，得到。

    上述另一种计算方式——_**Additive**_：同样地把这两个向量（需要计算关联度）分别乘上两个不同的矩阵和（_inner-product_），得到两个向量和；再然后把这个两个向量串起来，扔进激活函数，然后通过一个Transform再得到。（是随机初始化的，然后训练出来的）

    在本文中用到的方法默认为左边的_**Dot-product**_
* 在**Self-attention**中，分别用和、、求得对应的——**attention score**。求法如下：

![](https://s1.328888.xyz/2022/05/03/hiArO.png)

* 自己与自己计算关联性：

![](https://s1.328888.xyz/2022/05/03/hi4ZP.png)

再把算出来的通过处理：

> ：

不一定要用Softmax，只要是激活函数，有人用Relu效果也很好。

*   Extract information based on attention scores.根据这些去抽取出整个sequence中比较重要的咨询。

    把乘上得到新的向量。然后，再把每个乘上，然后再把它们加起来。

![](https://s1.328888.xyz/2022/05/03/hiq9m.png)

上边谁的值最大，谁的那个Attention的分数最大，谁的那个就会dominant你抽出来的结果。举例说：上述中如果支计算出来的值最大，那么就最接近。

**相似度计算方法**

在做attention的时候，我们需要计算query（）和某个key（）的分数（相似度），常用方法有：

* 点乘：
* 矩阵相乘
* 计算余弦相似度：
* 串联方式：把和拼接起来，
* 多层感知机：

**总结**

**Self-attention**就是一排input的vector得到相同数量的output的vector。计算中涉及到三个Transform矩阵是network的参数，是学习（learn）得来的，可以看作是带有权重的，以下认为是self-attention的矩阵**运算**。

![](https://s1.328888.xyz/2022/05/03/hiDWA.png)

每一个self-attention层只有一个矩阵。

然后为了得到得分，计算内积👇

![](https://s1.328888.xyz/2022/05/03/hitUR.png)

![](https://s1.328888.xyz/2022/05/03/hidkS.png)

同理👇

![](https://s1.328888.xyz/2022/05/03/hi5qi.png)

不是唯一的选项，也可以用其他激活函数。

接下来👇

![](https://s1.328888.xyz/2022/05/03/hiWmv.png)

这一串操作全是**矩阵**运算，不用加循环体，方便编程。把上述过程可以精简为👇

![](https://s1.328888.xyz/2022/05/03/hizo0.png)

称之为**Attention Matrix**。在整个**Self-attention**中输入是，输出是，其中又只有是未知的，需要透过训练集（training data）学习得到。

_**self-attention进阶版——Multi-head Self-attention**_

为什么我们需要多一点的head呢？——关系中蕴含着不同种类的关联性，以下 `2-head`为例：

![](https://s1.328888.xyz/2022/05/03/hip2J.png)

我们需要两种不同的相关性，所以需要产生两种不同的head，都有两个，另外一个位置做相似处理。head 1和head 2相对独立，分开做，如上图，只和运算。

**缺陷**——self-attention少了关于位置（上下文）的资讯，因此一下介绍相关的完善方法。

**Positional Encoding——把位置的咨询塞进self-attention**

* Each position has a unique positional vector （为每一个位置设定一个vector，不用的位置就有专属的一个vector）
* 把加到上：![](https://s1.328888.xyz/2022/05/03/hi07F.png)
* 这样子的Positional Encoding是**hand-crafted**的，人设的问题包括：可能sequence的长度超过人设的范围。在[Attention is all you need](https://arxiv.org/abs/1706.0376212)中这个代表位置的vector是透过一个规则产生的：一个神奇的sin、cos的function
* Positional Encoding任然是一个尚待研究的问题，可以创造一个新的产生办法，可以**learn from data**

[这篇论文](https://arxiv.org/abs/2003.0922930)讨论了Positonal Encoding的生成方法。

**Many applications of Self-attetntion**

* [_**Transformer**_](https://arxiv.org/abs/1706.03762)
* [_**BERT**_](https://arxiv.org/abs/1810.04805)
* 不仅在NLP领域，self-attention也可以用来语音识别（Speech）：[Transformer-Transducer](https://arxiv.org/abs/1910.12977)。文章中，self-attention被做了小小的改动。语音是一条非常长的sequence，由于时间序列下，为了描述一个时间段的语音信号，向量维数非常大，如果sequence的长度为，为了计算_Attention Matrix_，需要做次的`inner product`，算力和memory的压力很大。_**Truncated Self-attention**_被设计用来在只看一个小的范围（范围由人设定）而非整句话，以加快运算速度。

![](https://s1.328888.xyz/2022/05/03/hiIJy.png)

*   _**self-attention for Image**_：

    一张图片可以看作是一个vector的set

    <img src="https://s1.328888.xyz/2022/05/03/hiRWk.png" alt="" data-size="original">

    例如上图：每个位置的pixel都可以看作是一个三维的vector，所以这张图片是一个的vectors set。self-attention处理图片的工作的例子：[Self-Attention GAN](https://arxiv.org/abs/1805.08318)、[DEtection Transformer(DETR)](https://arxiv.org/abs/2005.12872)
*   _**self-attention for Graph**_：

    在Graph里，每个**node**看作一个向量（保存有相关的信息）；另外，graph里还有**edge**的信息。哪些node相连——哪些node有关联性：因此，邻接矩阵表示了在做self-attention的计算时，只需要计算相连的node之间的关联性就好了。

    <img src="https://s1.328888.xyz/2022/05/03/himq3.png" alt="" data-size="original">

    没有相连的nodes之间就不用计算attention score了，可设置为0，因为这些可能是domain knowledge暗示下的这种nodes间没有关系。

    由此，提出了一个很fancy的network：_**Graph Neural Network (GNN)图神经网络**_。老师表示水很深，把握不住，感兴趣可以另外自行学习。

**Self-attention和CNN的比较**

CNN可以看作是一种简化版的Self-attention，它只关注于receptive field；而self-attention则关注整张图像。self-attention看作是复杂化的CNN，用attention找到有关联性的pixel，仿佛是network自动learn且决定自己的“receptive field”（不再是人工划定）

[On the Relationship between Self-Attention and Convolutional Layers](https://arxiv.org/abs/1911.03584)用数学的方式严谨的证明CNN是self-attention的一个特例。self-attention设定特定的参数就可以做到和CNN一样的事情。

由于self-attention相较于CNN更加flexible，为了避免过拟合，需要更多的数据才能达到更好的效果。而CNN在训练资料较少时表现相对较好，因为随着数据增多，CNN并没有得到更多好处。

![](https://s1.328888.xyz/2022/05/03/hiYnd.png)

[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929)这篇文章用self-attention处理影像，它把一张图像拆成个patch，把每个patch当作一个word处理。（当然这个数据集量一般研究者很难搜集到，这篇文章来自Google）

**Conformer**：一个CNN和Self-attention的混合体。

**Self-attention和RNN的比较**

_**RNN：Recurrent Neuroal Network（循环神经网络）**_和self-attention一样都是处理input是一个sequence的状况，在第一个RNN里扔进input第一个vector，然后output一个东西hidden layerFCprediction，对于第二个RNN需要input第一个吐出来的东西以及input第二个vector再output东西，以此类推，如下图👇

![](https://s1.328888.xyz/2022/05/03/hiceQ.png)

当然，RNN可以是双向的。两者不同的地方：对于RNN而言，距离较远的两个vector，如果前者不被memory一直记忆到输入处理后者的网络，两个向量很难产生关联性；而再attention里，输入向量是平行的，输出向量是平行的，只要match到，就可以产生任意两个向量的关联性。——天涯若比邻，aha

所以目前来看attention优于RNN，许多RNN架构都改为attention了。进一步了解两者关系：[Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236)，attention加一点东西就会变成RNN。

**延展**

Self-attention有非常多的变形：[Long Range Arena: A Benchmark for Efficient Transformers](https://arxiv.org/abs/2011.0400642)、[Efficient Transformers: A Survey](https://arxiv.org/abs/2009.06732)

![](https://s1.328888.xyz/2022/05/03/hiww4.png)

由于self-attention最大的问题就是运算量大，所以未来相关的问题很多关于如何变形以减少运算量，提高运算速度。如何使attention越来越好，也是未来尚待研究的问题。
