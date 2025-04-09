
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from docutils.nodes import target
from multipledispatch.dispatcher import source
from nltk.classify.tadm import encoding_demo
# from jieba.lac_small.predict import batch_size
from sympy.stats.sampling.sample_numpy import numpy
from torch.autograd import Variable
import math
import torch
from torch.nn.functional import embedding, dropout
import numpy as np
import copy

from transformers.models.ctrl.modeling_ctrl import EncoderLayer


class Embeddings(nn.Module):
    """Embeddings表面这是两个共享参数的embedding层"""
    def __init__(self,  d_model,vocab_size,):
    # d_model is the embedding size
    #vocab_size:词表大小
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding( vocab_size,d_model)
        self.d_model = d_model
    def forward(self, x):
        """embedding层是首层
        :param x:代表输入给模型的文本通过词汇映射后的tensor"""
        #* math.sqrt(self.d_model)可快速收敛系数
        return self.lut(x) * math.sqrt(self.d_model)

class  PositionalEncoding(nn.Module):
    def __init__(self, d_model,dropout, max_len=5000):
        """:param d_model:embedding的维度
            :param dropout:dropout的比率
            :param max_len:序列的最大长度"""
        super(PositionalEncoding, self).__init__()
        #实例化nn预定义的dropout层
        self.dropout = nn.Dropout(p=dropout)
        #初始化位置编码矩阵,大小为max_len  * d_model
        pe=torch.zeros(max_len,d_model)
        #初始化绝对位置矩阵,词汇的绝对位置用其索引表示
        #扩充一列匹配形状
        positon = torch.arange(0, max_len).unsqueeze(1)
        # print("position===", positon.shape)
        # 1X (d_model/2)形状


        # 绝对位置矩阵初始化之后，接下来就是考虑如何将这些位置信息加⼊到位置编码矩阵中，
        # 最简单思路就是先将max_len x 1的绝对位置矩阵， 变换成max_len x d_model形状，然后覆盖原来的初始位置编码矩阵即可，
        # 要做这种矩阵变换，就需要⼀个1xd_model形状的变换矩阵div_term，我们对这个变换矩阵的要求除了形状外，
        # 还希望它能够将⾃然数的绝对位置编码缩放成⾜够⼩的数字，有助于在之后的梯度下降过程中更快的收敛.这样我们就可以开始初始化这个变换矩阵了.
        # ⾸先使⽤arange获得⼀个⾃然数矩阵， 但是细⼼的同学们会发现， 我们这⾥并没有按照预计的⼀样初始化⼀个1*d_model的矩阵，
        # ⽽是有了⼀个跳跃，只初始化了⼀半即1xd_model/2 的矩阵。 为什么是⼀半呢，其实这⾥并不是真正意义上的初始化了⼀半的矩阵，
        # 我们可以把它看作是初始化了两次，⽽每次初始化的变换矩阵会做不同的处理，第⼀次初始化的变换矩阵分布在正弦波上， 第⼆次初始化的变换矩阵分布在余弦波上，
        # 并把这两个矩阵分别填充在位置编码矩阵的偶数和奇数位置上，组成最终的位置编码矩阵.


        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))  # 跳跃其实是初始化两次,后面的值用于放缩参数
        # print(torch.exp(torch.arange(0, 512, 2) * -(math.log(10000) / 512))) #没有后面的* -(math.log(10000) / 512)值会变inf
        print('div_term==', div_term)
        print('div_term==', div_term.shape) #div_term== torch.Size([256])
        print('sin部分==', torch.sin(positon*div_term))
        print('sin部分==', torch.sin(positon*div_term).shape) # sin部分== torch.Size([60, 256])
        pe[:, 0::2] = torch.sin(positon * div_term)  # 偶数位置用sin处理后的值填充
        # print('cos部分==', torch.cos(positon*div_term))
        # print('cos部分==', torch.cos(positon*div_term).shape)# cos部分== torch.Size([60, 256])
        pe[:, 1::2] = torch.cos(positon * div_term)  # 奇数部分用cos处理后的值填充

        # pe现在还只是⼀个⼆维矩阵，要embedding的输出（⼀个三维张量）相加，就必须拓展⼀个维度，所以这⾥使⽤unsqueeze拓展维度.
        #1*max_len*d_model
        pe = pe.unsqueeze(0)

        '''
        peregister.buffer()  :对模型有帮助,但不是超参数或者参数,且不随优化过程修改
        如这里的pe位置编码矩阵
        '''
        #buffer:对模型有帮助,但不是超参数或者参数,且不随优化过程修改
        self.register_buffer('pe', pe)
        #初始化绝对位置矩阵,词汇的绝对位置用其索引表示
        #
        #参数形状为1,代表矩阵拓展的位置,使向量变成max_len * 1 的矩阵
    def forward(self, x):
        print('x.size===', x.size(1))  # x.size=== 4
        print('x.size===', x.shape)  # torch.Size([2, 4, 512])
        print('pe.shape===', self.pe[:, :x.size(1)].shape)  # pe.shape=== torch.Size([1, 4, 512])
        #初始化时max_len=5000,但一个句子通常不会有5000个词汇,因此这里截取:x.size(1)个列,即句子单词数,前面才是有用的信息
        #调整位置参数的大小,再将位置信息加入x中,
        #位置编码不参与更新
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)

        return self.dropout(x)

def emb_test():
    # embedding演示
    # 8代表的是词表的大小, 也就是只能编码0-7
    # 如果是10， 代表只能编码0-9  这里有11出现所以尚明embedding的时候要写成12
    embedding = nn.Embedding(12, 3)
    input = torch.LongTensor([[1, 2, 3, 4], [4, 11, 2, 1]])
    print(embedding(input))

    # padding_idx代表的意思就是和给定的值相同， 那么就用0填充， 比如idx=0那么第二,四行都是0
    embedding=nn.Embedding(10,3,padding_idx=0)
    input = torch.LongTensor([[0, 2, 0, 5]])
    print(embedding(input))

def subsequent_mask(size):
    x=torch.tensor([i for i in range(size**2)])
    x=x.resize(1,size,size)
    '''
    torch.triu(input, diagonal=0, *,out=None) -> Tensor
    返回input 张量，对应对角线 (diagonal）取值的结果。 其余位置为0
    '''
    print('====\n', np.triu(x, k=1))
    print('====\n', np.triu(x, k=-1))
    print('====\n', x-np.triu(x, k=-1))


    subsequent_mask=np.triu(x,k=1).astype('uint8') #上半三角
    #下三角矩阵,确保前面的单词在处理时不会收到后面单词的信息

    return torch.from_numpy(1-subsequent_mask) #求下半三角

def attention(query, key, value, mask=None, dropout=None):
    """注意力机制的实现"""
    #q的最后一维一般就是embedding层大小
    d_k = query.size(-1)
    #交换 key的后两维从而实现矩阵乘法,再/sqrt(d_k)
    score=torch.matmul(query, key.transpose(-2,-1))/math.sqrt(d_k)

    #是否使用掩码张量
    if mask is not None:
        #用mask_fill方法,将掩码张量与score张量每个位置进行比较,如果为0则用-1e9替换
        score=score.masked_fill(mask==0,-1e9)

    # 对scores的最后⼀维进⾏softmax操作, 使⽤F.softmax⽅法, 第⼀个参数是softmax对象, 第⼆个是⽬标维度.
    # 这样获得最终的注意⼒张量
    p_attn=F.softmax(score,dim=-1)

    if dropout is not None:
        p_attn=dropout(p_attn)
    # softmax结果与value张量相乘, 得到最后的输出
    return torch.matmul(p_attn,value),p_attn

def transpose_view_demo():
    x=torch.tensor([[1,2,3],[4,5,6]])
    print(x.view(3,-1)) #展开顺序不变
    print(x.transpose(1,0).contiguous()) #列变行,行变列
    #contiguous用于让数据变回连续,使得后续可以执行view()等操作

def clones(module,N):
    """克隆模块"""
    # n:克隆的个数
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MultiHeadedAttention(nn.Module):
    def __init__(self, head, embedding_dim, dropout=0.1):
        """

        :param head:头的参数
        :param embedding_dim:embedding维度
        :param dropout: dropout概率
        """
        super(MultiHeadedAttention,self).__init__()
        # assert embedding_dim % head == 0,否则无法embedding
        assert embedding_dim % head == 0
        #每个头获得的维度
        self.d_k=embedding_dim // head
        self.head=head
        self.embedding=embedding_dim
        #4个相同线性层分别用于q,k,v,最终输出
        #多头的线性层不改变形状,因此输入形状=输出形状
        self.liners=clones(nn.Linear(embedding_dim,embedding_dim),4)
        self.attn=None
        self.dropout=nn.Dropout(p=dropout)
    def forward(self,query,key,value,mask=None):
        if mask is not None:
            #mask表示掩码矩阵,用于防止output提前进入
            mask=mask.unsqueeze(0)
        #获取batch_size
        batch_size=query.size(0)
        #zip将网络层和输入数据连接,再用view和transpose进行维度变换
        #q,k,v的维度为 batch_size * seq_len * embedding_dim 其中 embedding_dim = head * d_k
        #model(x).view(batch_size,-1,self.head,self.d_k).transpose(1,2),向量转换为 batch_size* head * num_head * d_k(embedding dim)
        #for model,x in zip 部分只用到了liners的前三个项 ,再用model 对 x 进行处理
        #transpose(1,2) 使得seq_len维度和embedding维度相邻,这样子注意力机制才能 得到词义与句子的关系
        #transposeqkv将qkv切分成注意力头个子矩阵，并转化成适合并行计算的形式
        q,k,v=[model(x).view(batch_size,-1,self.head,self.d_k).transpose(1,2)
               for model,x in zip(self.liners,(query,key,value))]
        print('-=-=', q.shape)
        print('-=-=', k.shape)
        print('-=-=', v.shape)
        #各个头的输出传至注意力层
        x,self.attn=attention(q,k,v,mask=mask,dropout=self.dropout)
        #注意力层输出结果要调整回原样,即transpose(1,2),
        #前面将1,2两个维度进行过转置,这里转回来再用contiguous()使得底层恢复连续存储,否则无法view调整形状
        x=x.transpose(1,2).contiguous().view(batch_size,-1,self.head*self.d_k)
        #最后通过最后一个线性层输出
        return self.liners[-1](x)

class PositionwiseFeedForward(nn.Module):#前馈神经网络(前馈全连接层)
    def __init__(self,d_model,d_ff,dropout=0.1):
        """

        :param d_model:线性层输入维度
        :param d_ff:第一个liner的输出,也是第二个liner的输入
        :param dropout:
        """
        super(PositionwiseFeedForward,self).__init__()
        self.w_1=nn.Linear(d_model,d_ff)
        self.w_2=nn.Linear(d_ff,d_model)

        self.dropout=nn.Dropout(dropout)
    def forward(self,x):
        #liner -> relu -> dropout -> liner 数据路径
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class   LayerNorm(nn.Module):

    def __init__(self,features,eps=1e-6):
        """
        :param features: 输入数据的维度,等价于前面的d_model或者embedding_dim
        :param eps: 用于防止分母为0
        """
        super(LayerNorm,self).__init__()
        #a2和b2也是模型的参数,需要训练,因此封装在nn.Parameter中
        self.a2=nn.Parameter(torch.ones(features))
        self.b2=nn.Parameter(torch.zeros(features))
        self.eps=eps


    def forward(self,x):
        """
        :param x:上一层的输出
        :return:维度与x一致
        """
        #求最后一个维度的均值和标准差,保持输出维度和输入维度一致
        mean = x.mean(-1, keepdim=True) #求均值,(4,1)->(4,1),如果keepdim=False,则 (4,1)->(4)
        std=x.std(-1, keepdim=True)
        #规范化操作 (x-mean)/标准差 ,+ self.eps和 + self.b2都是为了防止/0和0结果的出现

        #最后对结果乘以我们的缩放参数，即a2，*号代表同型点乘，即对应位置进⾏乘法操作，加上位移参数b2.返回即可.
        norm=(x - mean) / (std + self.eps)
        #y=a*x+b
        return self.a2 * norm + self.b2

class SublayerConnection(nn.Module):
    def __init__(self,size,dropout=0.1):
        """
        :param size: embedding_dim
        :param dropout:
        """
        super(SublayerConnection,self).__init__()
        #规范化
        self.norm=LayerNorm(size)
        self.dropout=nn.Dropout(dropout)
    def forward(self,x,sublayer):
        """

        :param x: 上一个层或者子层输入
        :param sublayer: 子层连接中的子层函数(本层的处理,在transformer中可能是多头注意力机制或前馈神经层
        :return: x+子层函数的输出
        """
        return x+self.dropout(sublayer(self.norm(x)))

if __name__ == '__main__':
    # emb_test()
    # div_term = torch.exp(torch.arange(0, 128, 2) * -(math.log(10000.0) / 128))  # 跳跃其实是初始化两次,后面的值用于放缩参数
    # print(torch.exp(torch.arange(0, 128, 2) * -(math.log(10000.0) / 128)),torch.exp(torch.arange(0, 128, 2) * -(math.log(10000.0) / 128)).shape)

    x = Variable(torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]]))
    emb = Embeddings(512,1000)
    emb_res=emb(x)
    print("embr:", emb_res)
    d_model=512
    dropout=0.1
    max_len=60


    # embedding维度是512维
    d_model = 512

    # 置0比率为0.1
    dropout = 0.1
    # 句子最大长度
    max_len=60
    x = emb_res
    pe = PositionalEncoding(d_model, dropout, max_len)
    pe_result = pe(x) #加上位置编码后后的输入
    # print(pe_result)
    print(pe_result) # torch.Size([2, 4, 512])

    # 可视化,展示sin和cos函数
    # plt.figure(figsize=(15, 5))
    # pe=PositionalEncoding(20,0)
    # y=pe(Variable(torch.zeros(1,100,20)))
    # plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())
    # plt.legend(['dim %d'%p for p in [4,5,6,7]])
    # plt.show()

    #掩码演示
    # sm=subsequent_mask(5)
    # print(sm)

    q=k=v=pe_result
    # print('q.shape',q.shape)
    attn,p_attn=attention(q,k,v)
    print('attn',attn,attn.shape)
    print('p_attn',p_attn,p_attn.shape)
    #reshape() 函数用于改变数组或张量的形状，但不改变元素的排列顺序
    #transpose() 函数用于交换数组或张量的轴的顺序，从而改变元素的排列顺序。
    transpose_view_demo() #transpose和view的区别

    #多头注意力机制demo
    head=8
    embedding_dim=512
    dropout=0.2
    q=k=v=pe_result #q,k,v的维度都是[2,4,512]
    print("q.shape",q.shape)
    mask=Variable(torch.zeros(8,4,4))
    mha=MultiHeadedAttention(head,embedding_dim,dropout)
    mha_result,mha_p_attn=mha(q,k,v,mask)
    print("mha_result,",mha_result,"\nmha_result.shape",mha_result.shape) #torch.Size([4, 512])

    #前馈全连接层demo
    d_model=512
    d_ff=64
    dropout=0.2
    ff=PositionwiseFeedForward(d_model,d_ff,dropout)
    #处理多头注意力传出的结果
    ff_result=ff(mha_result)
    print(mha_result.shape) #torch.Size([4, 512])
    print(ff_result.shape) #torch.Size([ 4, 512])
    # 前馈全连接层不对输入张量的形状进行改变

    #正则化层LayerNorm
    features = d_model = 512 #形状和前馈全连接层传出的结果一样
    eps = 1e-6
    ln=LayerNorm(d_model)
    #处理前馈全连接层传出的结果
    ln_result=ln(ff_result)
    print(ln_result,ln_result.shape) #torch.Size([4, 512])

    #残差连接中的子层连接(SublayerConnection
    size=512
    dropout=0.2
    head=8
    d_model=512
    x=pe_result
    mask=Variable(torch.zeros(8,4,4))
    self_attn=MultiHeadedAttention(head,embedding_dim,dropout) #多头自注意力输出
    sublayer=lambda x: self_attn(x,x,x,mask) #子层函数,把给出的layer放入q,k,v中,实现自注意力机制
    sc=SublayerConnection(size,dropout)
    sc_result=sc(x,sublayer) #残差结果即上一层的输出pe_result+子层函数的输出
    print(sc_result,sc_result.shape) #torch.Size([4, 512])

    class EncoderLayer(nn.Module):
        def __init__(self, size, self_attn, feed_forward, dropout): #初始化编码器层
            """

            :param size: embedding维度大小
            :param self_attn: 多头自注意力层的实例化对象
            :param feed_forward: 前馈全连接层实例化对象
            :param dropout: dropout率
            """
            super(EncoderLayer, self).__init__()
            self.self_attn = self_attn
            self.feed_forward = feed_forward
            self.sublayer = clones(SublayerConnection(size, dropout), 2) #克隆两个子层连接(残差网络
            self.size = size
        def forward(self, x, mask): #前向传播
            """

            :param x: 上一层输出
            :param mask: 掩码张量
            :return:
            """

            x=self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask)) #第一层连接,输入x,调用自注意力函数
            return self.sublayer[1](x, self.feed_forward) #第二个子层连接,输入x,子层函数
    #编码器层encoder layer
    print("编码器层encoder layer")
    size=512
    head=8
    d_model=512
    d_ff=64
    x=pe_result#位置信息输出即编码层的输入
    dropout=0.2
    self_attn=MultiHeadedAttention(head, d_model) #多头注意力层实例化
    ff=PositionwiseFeedForward(d_model, d_ff, dropout) #前馈连接层
    mask=Variable(torch.zeros(8,4,4)) #掩码张量
    encoder_layer=EncoderLayer(d_model,self_attn, ff, dropout) #编码层实例化
    encoder_layer_result=encoder_layer(x, mask) #编码层前向传播



    #编码器encoder
    class Encoder(nn.Module):
        def __init__(self,layer,N):
            """

            :param layer: encoder_layer
            :param N: num_layers
            """
            super(Encoder,self).__init__()
            #N个编码器层,用深拷贝避免指向同一个层
            self.layers=clones(layer,N)
            self.N=N
            self.norm=LayerNorm(d_model) #编码器层后进行层归一化
        def forward(self,x,mask):
            """

            :param x:input
            :param mask: 掩码
            :return: encoder output
            """
            for layer in self.layers:
                x=layer(x,mask)

            return self.norm(x)
    #编码器
    print("编码器encoder")
    x=pe_result#位置信息输出即编码层的输入
    size = 512
    head = 8
    d_model = 512
    d_ff = 64
    c = copy.deepcopy
    attn = MultiHeadedAttention(head, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    dropout = 0.2
    #这些对象一定要做深拷贝,避免多个layer指向同一位置
    layer = EncoderLayer(size, c(attn), c(ff), dropout) #编码层实例化

    # 编码器中编码器层的个数N
    N = 8
    mask = Variable(torch.zeros(8, 4, 4))

    encoder = Encoder(layer, N)
    encoder_result = encoder(x, mask)
    print("encoder_result",encoder_result)
    print("encoder_result.shape",encoder_result.shape)

    class DecoderLayer(nn.Module):
        #包括三个子层
        def __init__(self,size,self_attn,src_attn,feed_forward,dropout):
            """

            :param size: embedding大小
            :param self_attn: 自注意力层
            :param src_attn: 普通注意力层
            :param feed_forward: 前馈全连接层
            :param dropout: 
            """
            super(DecoderLayer,self).__init__()
            self.size = size
            self.self_attn = self_attn
            self.src_attn = src_attn
            self.feed_forward = feed_forward
            self.sublayer = clones(SublayerConnection(size,dropout),3) #子层连接,残差网络实现
        def forward(self,x,memory,src_mask,tgt_mask):
            """

            :param x: 来自上一层的输入
            :param memory: 来自encoder的语义存储变量
            :param src_mask: 源数据掩码张量
            :param tgt_mask: 目标数据掩码张量
            :return:
            """
            x=self.sublayer[0](x,lambda x:self.self_attn(x,x,x,tgt_mask)) #自注意力层
            x=self.sublayer[1](x,lambda x:self.src_attn(x,memory,memory,src_mask))# 普通注意力层,encoder传来的信息memory
            x=self.sublayer[2](x,self.feed_forward) #前馈全连接层
            return x
    print("decoder layer")
    x=pe_result
    memory=encoder_result
    mask=Variable(torch.zeros(8,4,4))
    source_mask=target_mask=mask

    head=8
    size=512
    d_model=512
    d_ff=64
    dropout=0.2
    c=copy.deepcopy
    self_attn=MultiHeadedAttention(head,d_model,dropout)#自注意力层
    src_attn=MultiHeadedAttention(head,d_model,dropout)    #普通注意力层
    ff=PositionwiseFeedForward(d_model,d_ff,dropout) #前馈全连接层
    de_layer=DecoderLayer(size,c(self_attn),c(src_attn),c(ff),dropout)
    de_layer_result=de_layer(x,memory,source_mask,target_mask)
    print("de_layer_result.shape",de_layer_result.shape,"\nde_layer",de_layer_result)
    class Decoder(nn.Module):
        def __init__(self, layer, N):
            super(Decoder, self).__init__()
            self.layers = clones(layer, N)
            self.norm = LayerNorm(layer.size)



        def forward(self, x, memory, source_mask, target_mask):
            for layer in self.layers:
                x = layer(x, memory,  source_mask, target_mask)
            return self.norm(x)

    print("解码器decoder")
    x = pe_result  # 位置信息输出即解码器的输入
    size = 512
    head = 8
    d_model = 512
    d_ff = 64
    c = copy.deepcopy
    self_attn = MultiHeadedAttention(head, d_model, dropout)  # 自注意力层
    src_attn = MultiHeadedAttention(head, d_model, dropout)  # 普通注意力层
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)  # 前馈全连接层
    de_layer = DecoderLayer(size, c(self_attn), c(src_attn), c(ff), dropout)
    dropout=0.2
    # 这些对象一定要做深拷贝,避免多个layer指向同一位置
    # 编码器中编码器层的个数N
    N = 8
    mask = Variable(torch.zeros(8, 4, 4))
    source_mask = target_mask = mask
    memory=encoder_result
    decoder = Decoder(de_layer, N)
    decoder_result = decoder(x, memory,source_mask, target_mask)
    print("decoder_result", decoder_result)
    print("decoder_result.shape", decoder_result.shape)

    # 生成器
    class Generator(nn.Module):
        def __init__(self, d_model, vocab_size):
            """

            :param d_model: embedding维度
            :param vocab_size: 词表大小
            """
            super(Generator, self).__init__()
            #用线性层改变维度大小,最后用softmax输出概率
            self.project= nn.Linear(d_model, vocab_size)
        def forward(self, x):
            #先用liner转换为vocab_size大小的结果,最后用softmax输出
            #用log_softmax不影响最终结果
            return F.log_softmax(self.project(x), dim=-1)

    # 生成器(根据解码器输出生成output
    d_model=512
    vocab_size=1000
    gen = Generator(d_model, vocab_size)
    x=decoder_result
    output = gen(x)
    print("output.shape", output.shape,"output", output)

    #编码器-解码器
    class EncoderDecoder(nn.Module):
        def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
            """
            :param encoder: 编码器
            :param decoder: 解码器
            :param src_embed: source 源数据embedding函数
            :param tgt_embed: target目标数据embedding函数
            :param generator: 输出类别generator
            """
            super(EncoderDecoder, self).__init__()
            self.encoder = encoder
            self.decoder = decoder
            self.src_embed = src_embed
            self.tgt_embed = tgt_embed
            self.generator = generator

        def encode(self,source,source_mask):

            return self.encoder(self.src_embed(source),source_mask)

        def decode(self,memory,source_mask,target,tgt_mask):
            #decoder的 forward(self,  x, memory(编码器的输出), source_mask, target_mask):
            return self.decoder(self.tgt_embed(target),memory,source_mask,tgt_mask)

        def forward(self, src, tgt, src_mask, tgt_mask):
            """

            :param src: 源数据
            :param tgt: 目标数据
            :param src_mask: 源数据掩码
            :param tgt_mask: 目标数据掩码
            :return:
            """

            return self.generator(self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask))

    vocab_size=1000
    d_model=512
    #上面实现的编码器和解码器
    encoder=encoder
    decoder=decoder
    #嵌入层
    source_embd=nn.Embedding(vocab_size,d_model)
    target_embd=nn.Embedding(vocab_size,d_model)
    #上面实现的generator
    generator=gen
    #上面实现的模型
    encoder_decoder=EncoderDecoder(encoder,decoder,source_embd,target_embd,generator)
    # 假设源数据与目标数据相同, 实际中并不相同
    source = target = Variable(torch.LongTensor([[100, 2, 421, 508], [491, 998, 1, 221]]))

    # 假设src_mask与tgt_mask相同，实际中并不相同
    source_mask = target_mask = Variable(torch.zeros(8, 4, 4).long())

    ed = EncoderDecoder(encoder, decoder, source_embd, target_embd, generator)
    ed_result = ed(source, target, source_mask, target_mask)
    print('ed_result.shape ==', ed_result.shape,'ed_result', ed_result)

#TODO 回来查报错
# def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):

