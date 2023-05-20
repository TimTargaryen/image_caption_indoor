# Introduction to Image caption

<center>黄奕铭，2022.07</center>

> 做一些**图像描述**的入门任务，data 2 text范式
>
> show&tell中的NIC和show、attend&tell的AttentionNIC以及自己的多功能attention已经实现，源码在ipynb中
>
> 最后还有一点对前沿和新idea的想法

## 综述

### 文本生成的一些metrics

* BLEU-n系列, 源自NLP中的n-gram模型，对句子进行滑窗处理: 
  $$
  BLEU_n(pred, GT) = \frac{推理结果以n为滑窗得出与真值相同的子串数}{真值以n为滑窗得出的子串总数}
  $$
缺点是，容易导向短句子，可以加长度惩罚因子
  
* n召回率 recall@n，衡量系统的查全能力：
  $$
  Recall_k = \frac{m}{w_r}=\frac{检索出的正样本}{所有正样本}
  $$
  
* 准确率：
  $$
  precision = \frac{m}{w_{infer}}=\frac{生成的整确词语数}{推理出的总词语数}
  $$

* METERO，用连续块定义的惩罚系数，乘上召回率与精确率的调和平均
$$
Metero = (1-Pen)\frac{P\cdot R}{\alpha P + (1-\alpha)R}, \\\\
  Pen = \lambda(\frac{chunks}{m})^\theta,chunks\ is\ the \ num \ of\ subsequent \ chunk \\\alpha ,\ \lambda, \ \theta  \space is\space hyperparameter
$$

* ROUGE系列

  * ROUGE-N 其是就是n召回率

  * ROUGE-L, 最长公共子序列召回和精准的调和，允许跳词:
    $$
    R_{lcs} = \frac{LCS}{m}\\P_{lcs} = \frac{LCS}{n}\\ F_{lcs} = \frac{(1 + \beta^2)R_{lcs}P_{lcs}}{R_{lcs+}\beta^2P_{lcs}}
    $$
    
  * ROUGE-W，对L跳词的情形赋更低的权，连续则权更高

  * ROUGE-S，ROUGE-N 中n元组允许跳词的情形

* CIDER

  * 前置知识TF-IDF，TF词频-IDF逆向文件频率，目的抑制语段中一些无法真正呈现语义的词：
    $$
    TF = \frac{文章中该词的出现频率}{文章总词数}\\
    IDF = log(\frac{语料库中的文档总数}{包含该词的文档数+ 1})\\
    TFIDF= TF \cdot IDF
    $$

  * 而CIDER的目的在于，给定一个参考语料库（将每个句子视为一个文档），而句子变为n-gram的向量, 然后计算候选句和参考句的余弦相似程度：
    $$
    S_x = [word_1...word_n], S_{GT} = [word_1...word_n]\\
    CIDER =AVG(\frac{TFIDF(S_x)\cdot TFIDF(S_{GT})}{||TFIDF(S_x)||\cdot||TFIDF(S_{GT})||})_{all}
    $$

* SPICE

  * ECCV2016中提出，通过scene graph预训练模型的生成计算三元组G，c为生成的标题，O()为目标类别函数, 生成目标类别，E()为关系函数，生成目标间关系的二元组，K()属性函数，生成目标与属性间的二元组
    $$
    scene\ graph = G(c) = <O(c),E(c),K(c)>\\ = <O_x,<O_x,R_{xy},O_y>,<O_x,attr>>
    $$
    然后将候选句子和生成句子的三元组转化为由一、二、三元组组成的集合，从集合意义上算召回率和精确率，最后算其调和平均（设该转化为T

    ),  即：
    $$
    P(c) = \frac{|T(G(c)) \cap T(G(S))|}{T(G(c))}
    \\R(c) = \frac{|T(G(c)) \cap T(G(S))|}{T(G(c))}
    \\SPICE = \frac{2P(c)\cdot R(C)}{P(C) + R(c)}
    $$

### 公开datasets（包含VQA）

* Flickr30k\9k\8k，节选自网络相簿

  ![image-20220706181130997](indoor.assets/image-20220706181130997.png)

* MSCOCO，Common Object in Context 微软的大型CV数据集，有不同任务的子集，其中2015captioning为Image Caption任务的数据集

  ![image-20220706173812645](indoor.assets/image-20220706173812645.png)

* VQA2017

  ![image-20220706172634447](indoor.assets/image-20220706172634447.png)

* VisualGenome数据集，2016年发布的大型语义理解数据集，其中有VQA任务的子集，并对其中的词与全部都映射到了WordNet的synset中

  ![image-20220706171857082](indoor.assets/image-20220706171857082.png)

### 最新综述

### Image-caption in transformer age

* 趋势是从cnn-rnn到tranformer， 大（预训练）模型，自监督
* IC(image caption)很多关键技术继承自机器翻译，尤其是encoder-decoder架构和attention机制
* 一个难题是编码器解码器异构的cnn-rnn liked的模型很难training end2end，冻结梯度的训练方法并不能很好地学到语义，因此有了一些解决思路
  * 提取更好的、语义更高的特征，如faster RCNN
  * 用attention去narrow 两种数据间分布的gap
  * 对齐两种数据、用scene-graph、GCN、树结构等等方法
* 而同构，即都用transformer的架构会更好端到端训练，可以做的地方在于attention的k、v、q是怎么与前面的模块连接的
* 在预训练大模型的时代，IC的很多实现都倾向于“pre-train + fine tune”或者对大数据集上训练好的模型做“prompt + predict”去下游的IC任务

## 经典之作

### Show&Tell

* basic idea:

  > * maximize the likelihood $ P(Sentence|Image) $
  >
  > * CNN, as decoder->embedding Image to a fixed-length vector, a kind of middle representation->RNN, as encoder
  >
  > * 相对机器翻译任务就是"(instead of an input sentence
  >
  >   in the source language)"
  >
  
* 一些思考：
  
  * 在这篇文章出的时候，只能用rnn处理定长的表示，而由于自然语言中对context的依赖，所以要最大化已知前面t- 1个滑窗的ground truth和图片张量的情况下第t个滑窗与ground truth相同的可能性，即RNN
  
    ![image-20220629080850536](indoor.assets/image-20220629080850536.png)
  
  * LSTM解决长程依赖问题（gradient对于bptt学习），原因在于输入门系数$i_t$稍小加上遗忘门系$f_t$稍大（可以特化不同门间忘掉的系数的比例，相当于周期性忘掉，超参）
  
    ![image-20220629082820681](indoor.assets/image-20220629082820681.png)
  
* NIC网络架构的设计：

  ![image-20220629084141451](indoor.assets/image-20220629084141451.png)

  * 训练时，输入图片和句子，输出概率，以概率和one-hot的词向量相乘求和的负值做为损失，不断使平均预测对one-hot的可能行提升
  * 推理则有两种，一种为推理结果不断填充，即先用$x_{-1}$推理出$p_0$然后通过乘we权重矩阵的方式做一次词嵌入，成为下个lstm单元的输入；第二种为对k大小的候选句子集进行搜索，得到$\sum p$最大的，这种方法实验效果好（**问题是**，**k大小的句子集是怎么选的？**）

* 实验与复现

  * metrics 有：1. 人去agreement 2. BLEU 3.推荐系统里的召回率
  * 这篇文章实验的策略是：
    * 大数据时才比特征工程强；
    * cnn 的backbone要pretrained
    * 随机梯度下降且learning rate固定
    * 512维的词嵌入 word2vec，从新闻语料库中得到的$W_e$
  * 结果的讨论：
    * 14年sota，但比人差
    * 数据量大，迁移效果就会好
    * 尽管生成的句子很novel，但是BLEU分数尚可
    * 用词向量，就是为了词之间的similarity（相较BoW），14年

### Show, Attend & Tell

>这篇文章，在结论和开头，一个insight都是可解释性，模式的对应、attention的算出的ROI
>
><img src="indoor.assets/image-20220630162524632.png" alt="image-20220630162524632" style="zoom:50%;" />
>
>![image-20220630162451578](indoor.assets/image-20220630162451578.png)
>
>另一个insight在于用attention去搞定salient feature，即 contribution中提到的“where” and “what”，在related works提到另一篇工作是用目标检测的思路去做的，但是attention潜在地做了这件事，

* 具体methods

  * 提取图像特征时，仅提取到feature map的级别，不用线性层进行再提取，这样可以选择性的通过权重进行focus

  * 变种的lstm与attention结合

    ![image-20220630180617542](indoor.assets/image-20220630180617542.png)

    ![image-20220630213323880](indoor.assets/image-20220630213323880.png)

    * 相较一般lstm多了z的部分
    * y在训练时为one hot的词乘上嵌入矩阵就会变成词向量

    * 其中小z是由图像做attention计算得到，即通过打分得到分布，分布就是注意力的核心，然后乘上值，而文章的提出硬性随机注意力和软性确定注意力，就在$f_{att}$这个打分函数上；
    * 

    ![image-20220630213732493](indoor.assets/image-20220630213732493.png)

    * $z_t$是每个隐状态和同一个也是唯一一个图像的中间表示通过单头                 attention算出来的

    * 和show&tell一样, "we use a deep output layer to compute the output word probability",每个词是通过image的中间表示和之前产生的单词，lstm算出的隐状态算出来的

    * 网络结构如下

      ![image-20220701115003172](indoor.assets/image-20220701115003172.png)

  * 打分函数，硬性随机与确定：

    * 硬性：

      * 之所以说是硬性，那就是查询向量s 为0,1组成的向量，乘上值向量后，就只会进行选中的功能，s的多项式分布由fatt$\alpha$决定
    
        ![image-20220701120758394](indoor.assets/image-20220701120758394.png)
    
      * 在目标函数上就是优化正确one-hot词向量在给定middle representation下的概率
        $$
        argmax \ log \ p(y|a)
        $$
    
      * 学习方法上就是用蒙特卡罗搜索梯度，通过$\alpha$得出分布，算所有的s，将该偏导进一步转化，为最下面的式子，变成强化学习规则形式，便可以用reward去学习![image-20220701130051024](indoor.assets/image-20220701130051024.png)
    
        ![image-20220701130320717](indoor.assets/image-20220701130320717.png)
    
    * 软性
    
      * 之所以是软性，是因为算的是一定location下的期望
    
      * 这里直接算的是正则化后的几何平均近似$p(y|a)$期望说明了优化目标是该边缘概率分布
    
        ![image-20220701182521685](indoor.assets/image-20220701182521685.png)
    
      * 优化目标如下，惩罚项的目的是为了尽可能的让fatt打的分$\alpha$尽量接近平均数![image-20220701183044493](indoor.assets/image-20220701183044493.png)

* 实验与结果讨论：

  * backbone 用的是vggnet，没做finetuning

  * 在常见数据集上sota了，但比nic没有大幅提升

  * 文章做了很多如下的可视化，说明了attention的可解释性，即选择location

    ![image-20220701185517202](indoor.assets/image-20220701185517202.png)

  * 对于开源的实现，是tenano写的，基本上是tensor级别的实现

### 经典baseline：Bottom up&top down

>* bottom-up: cnn架构自底而上，“proposes a set of salient image regions”，用目标检测的思路做了attention的效果
>* top-down：attention提取自顶而下
>

* basic idea
  
  * 先用bottom up做通用任务，然后用top down去做context相关的具体任务
  
  * 总体目的是用目标检测的bottom up方法把先做一次“注意力”，关注重点区域，然后用attention + lstm去做top-down
  
  * 用通用的bottom-up的cnn-based目标检测架构，提取bounding box的坐标以及对应的类，做了hard attention的效果
  * 然后用context specific的top-down架构或者VAQ对应模型去做特定的任务
  
* specific methodology
  * 使用Faster RCNN将图像编码为K x 2048的一组向量，包含了分类框坐标与框中对应物的类别信息，用cnn 从底向上提取了目标检测意义下的注意力结果
  
  * 然后将该组向量传入top down LSTM结构时，将其与前一个Language lstm单元的隐状态，可学习的词嵌入矩阵与one hot词向量的乘积拼接为一个新的向量，送入top down单元计算出隐状态；
  
    将计算的隐状态做查询向量和图片向量组进行attention计算，算出$\hat{v}$送入language LSTM（第二层lstm）
  
  * 而language LSTM则将$\hat{v}, h_{t}^1$拼接成的向量做为输入，将引状态做一次softmax得出词向量y在前面所有词做为前提的条件概率
  
    ![image-20220705092812098](indoor.assets/image-20220705092812098.png)
  
  * 训练方法
  
    * 一般的，优化交叉熵
    * 而针对CIDER指标，最大化CIDER的平均期待
    * 训练时，采用SCST的强化学习方法，使用对抗方法，使网络中的上文词汇尽量贴合
    
  * VQA视频问答任务，即top down部分换为GRU编码问题，训练时，换上one hot对应的分数向量做为 label
  
    ![image-20220706042250575](indoor.assets/image-20220706042250575.png)
  
    * 大致上，将question 嵌入后用GRU算出隐状态，将其与提取的图像特征拼接，做一次top down的attention然后将结果再和算出的隐状态做元素乘积，经过激活函数，算出候选的one hot词向量（不同维对应不同词）的分数

* 实验
  * 首先，对coco和genome数据集中的公共部分做了区分，确保不会包含重复的部分
  * 其次，对faster RCNN中不清晰的类进行了清洗、对同义的类进行了合并
  * 用了之前的一个标题分割的工作做的validation
  * 对于图像理解使用的是resnet101，对于VQA任务使用的是resnet200 
  * 结果：在两个任务的test服务器上，均sota了



### Transformer的起源：《Attention is All You Need》

> transformer的时代就是自此开启，纯attention 架构，催生了后来的许做著名工作，如NLP中的gpt、bert、CV中的ViT、DETR、swin、跨模态的CLIP等

* 模型全架构， encoder-decoder范式，堆叠attention

  ![image-20220709162515716](indoor.assets/image-20220709162515716.png)

* 基本流程：
  1. 将已经词嵌入的词向量组进行位置编码（cos、sin周期性放缩）

  2. ![image-20220712183503776](indoor.assets/image-20220712183503776.png)

  其中左半边为encoder单元为堆叠的self-attention，经过堆叠的六个单元得到提取的特征做为key-value值分别送入不同的位置的decoder

  3. 根据输出概率向量解码为目标语言
  
* 论文中的idea：

  1. encoder 选择self-attention的原因是计算复杂度，数据并行，长程依赖，本文也给出了几种范式下时间复杂度、序列操作时间、长程依赖的对比

     ![image-20220712185627229](indoor.assets/image-20220712185627229.png)

  2. positional encoding的意义是attention学不到上下文的依赖关系，所以要周期缩放embedding后的词向量

  3. mask矩阵的目的在于推理时只能和之前token进行计算

  4. key-value的多头注意力是在计算query和key的相似度

  5. 打分函数选择缩放点积的原因是，词嵌入的feature维很大512->2048(中间层)，除$d_k^{\frac{1}{2}}$后softmax后的结果不会梯度过小

* 实验

  * 用英-德、英-法翻译数据集做的
  * 用8台P100训练12小时得到了base model，训练3.5天得到了最大的模型
  * sota：英德BLEU：28.4 英法：41.0

### Dual-Level

> *文章的结论和介绍的核心insight是 目标检测出的region与图像的grid结合，而全文都在做的则是视觉不同信号位置的编码、它们之间的交互*

* basic idea

  1. 目标识别出的region有object-level的信息利于语义理解
  2. 检测出的目标缺乏全局信息
  3. 但这正好能被网格特征图所捕捉
  4. 同时这也会引来问题，形状的相似会让grid和region在attention中得到高注意力，但实际上并没有真正语义上的相似
  5. 所以用LCCA解决4的问题，用DWSA提取两种表示，用CRA的attention去将两种特征在一定相关性意义下嵌入

* 具体的方法：

  1. 用GPE去编码1d的grids信息，让其知道中心与角落间图像绝对的位置信息差异

  2. 用RPE编码region的bounding box坐标信息，做到同样的目的

  3. 对于相对性位置信息的编码，将网格也视为bounding box，然后用以下公式编码信息

     ![image-20220712232734091](indoor.assets/image-20220712232734091.png)

  4. LCCA通过之前编码时的bounding box用region是否与grid相交建立了无向但可以指向自己的图，对于该图图节点建了对应的领接矩阵的权重矩阵W，该权重矩阵用于学习节点至节点的关系，整合位置信息，在注意力计算中

     ![image-20220713001057995](indoor.assets/image-20220713001057995.png)
     
     ![image-20220713095710928](indoor.assets/image-20220713095710928.png)
     
     用上面的公式训练出结点至节点的关系，原文说做了图级的softmax
     
  5. 训练，和前面的工作一致，最大化已知前面n个词做为条件的分类正确该率
  
     ![image-20220713101332918](indoor.assets/image-20220713101332918.png)
  
     对于CIDER指标，用了强化学习的SCST方法训练
  
* 实验细节：

  * coco上训练
  * backbone 用的vg数据集上预训练的faster rcnn
  * offline 和 online都sota了
  * 做了消融实验展示了CRA在不同超参下的表现、以及只有transfromer没有CRA的结果，证明了CRA的重要性
  * 也对LCCA做了消融实验
  * 做了Transformer和DLCT的可视化对比

## 前沿

### CLIP

* prompt + 对比学习

  ![image-20220713122507999](indoor.assets/image-20220713122507999.png)

* 大模型，训练对个人不可能

* zero shot + linear probe

* ![image-20220504104437381](indoor.assets/image-20220504104437381.png)

  速度有待提升

* ![image-20220504104028322](indoor.assets/image-20220504104028322.png)

* ![image-20220713122723277](indoor.assets/image-20220713122723277.png)

  计算效率 + 对比学习 + 生成caption

* clip不是为了few shot而设计的

### CLIP的应用DALL E 1&2

* 用CLIP做ranking&用CLIP的图像编码器

  ![image-20220713142500985](indoor.assets/image-20220713142500985.png)

  

  ![image-20220713140538821](indoor.assets/image-20220713140538821.png)



## 总结

Image Caption任务的范式就是下图的encoder-decoder, 双塔模型

![image-20220713141247541](indoor.assets/image-20220713141247541.png)

* 基本上挑战就是以下两点：
  * 视觉和语言的信息如何提取得更具高层语义
  * 视觉与语言的信息的中间表示如何更有解释性地对齐、融合：
    * 是让图变成句子？BUTD与Dual-Level用attention干的都是这件事
    * 还是句子变为图？



## 想到的、可能的idea

1.  让语义变的更高层：

   * 对于图像信息的提取能不能用语义分割、实例分割的级别？
   * 对于语言信息的提取能不能做到现在NLP前沿中根据语法、词法树、三元组等等策略做到信息抽取的级别？
   * 因此，图片与语言中间表示的结合（attention-based）可以做得更加可解释？

2.  跟随最新的多模态工作与预训练模型做相关工作

   * 对CLIP做关于新数据集的linear probe 然后加一些可解释性？
   * 对预训练模型、预训词嵌入模型做剪枝、矩阵分解 、蒸馏+ fine tune做trade off提性能？
   * 训练上、非bp的？强化学习、生成式的？

3. 其他表示：

   * 图神经网络？
   * 引入知识图谱 + 可解释？