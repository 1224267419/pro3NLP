from io import open
import unicodedata
import re
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from openpyxl.styles.builtins import output
from torch import optim
import time
import matplotlib.pyplot as plt


#自动定义cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 起始标志
SOS_token = 0
# 结束标志
EOS_token = 1

class Lang:
    def __init__(self, name):
        #语言类别
        self.name = name
        #词语对应数值
        self.word2index = {}
        #数值对应词语,0代表起始标志，1代表结束标志(上面有
        self.index2word = {0: "SOS", 1: "EOS"}
        #当前已有连哥哥词汇
        self.n_word=2
    def addSentence(self, sentence):
        """
        将句子转换为对应的数值序列
        :param sentence: 句子
        :return: 序列
        """
        for word in sentence.split(' '):#将句子按照空格分割，并遍历
            self.addWord(word)#调用addword方法
    def addWord(self, word):
        """添加单词"""
        if word not in self.word2index:#如果单词不在词典中才添加
            self.word2index[word] = self.n_word#将单词添加到词典中
            self.index2word[self.n_word] = word#将单词向量添加到对应的词典中
            self.n_word += 1#词汇数加1

print("Lang类演示")
# a="i am your father"
# lang=Lang("en")
# lang.addSentence(a)
# print(lang.word2index)
def unicodeToAscii(s):
    """#将unicode编码转换为ascii编码,去除一些重音标志"""
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )
def normalizeString(s):
    """#将字符串转换为小写，去除一些特殊字符和两端的空白符"""
    s=unicodeToAscii(s.lower().strip())
    s=re.sub(r"([.!?])", r" \1", s)#在标点符号前后添加空格.
    # \1代表正则表达式中的分组引用
    # 去除非字母和标点符号的字符
    s=re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


a="i am your Father!@#$%"
print(normalizeString(a))

#文件读入,拆分句子和词语对
path="./data/eng-fra.txt"
def readLangs(lang1, lang2,path):#读取语言文件
    lines=open(path, encoding='utf-8').read().strip().split('\n')#读取文件并按行分割
    #以\t划分子列表(eng和fra对应的语句对),对读取到的每一行进行标准化处理
    pairs=[[normalizeString(s) for s in l.split('\t')] for l in lines]
    input_lang=Lang(lang1)#创建输入语言对象
    output_lang=Lang(lang2)#创建输出语言对象
    return input_lang, output_lang, pairs

lang1="eng"
lang2="fra"#创建输入语言和输出语言对象

input_lang,output_lang,pairs=readLangs(lang1, lang2,path)#读取语言文件并创建语言对象和语句对
# print("input_lang",input_lang)
# print("output_lang",output_lang)
# print("pairs",pairs) #语言对

#设置最大句子长度
MAX_LENGTH=10

#选择带有指定前缀的语言特征数据作为训练数据(数据筛选)
eng_prefixes=(
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

def filterPair(p):#筛选语句对
    """p[0]即英文字符串,p[1]即法文字符串,二者场合都均要小于MAX_LENGTH且
    p[0]要以eng_prefixes中的前缀开头"""
    return (len(p[0].split(' '))<MAX_LENGTH and
            len(p[1].split(' '))<MAX_LENGTH and
            p[0].startswith(eng_prefixes))

def filterPairs(pairs):#筛选出所有满足条件的语句对
    return [pair for pair in pairs if filterPair(pair)]

#整合数据预处理部分
def prepareData(lang1, lang2, path, reverse=False):
    """读取语言文件并创建语言对象和语句对,将其添加进"""
    #获得两个对应的语言类以及语言对列表
    input_lang, output_lang, pairs = readLangs(lang1, lang2, path)
    # 筛选出所有满足条件的语句对
    pairs=filterPairs(pairs)
    for pair in pairs:
        #添加语句,填充词表
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    return input_lang, output_lang, pairs

input_lang, output_lang, pairs=prepareData('eng', 'fra', path)
print(random.choice(pairs))
print(input_lang.n_word)
print(output_lang.n_word)

#pair2tensor
def tensorFromPair(pair):
    def tensorFromSentence(lang, sentence):
        """
        句子转换为张量
         :param lang: Lang的实例化对象
         :param sentence: 待转换的句子"""
        # 句子分割遍历每一个词汇,再用word2index方法找到其对应的索引
        indexes = [lang.word2index[word] for word in sentence.split(' ')]
        # 句子末尾添加EOS_token
        indexes.append(EOS_token)
        # 将索引列表转换为 n*1  张量
        return torch.tensor(indexes, dtype=torch.long,device=device).view(-1, 1)
    """将语言对转换为tensor"""
    #处理源语言和目标语言
    input_tensor=tensorFromSentence(input_lang, pair[0])
    target_tensor=tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

#随机抽一个处理一下
pairs_tensor=tensorFromPair(random.choice(pairs))
print(pairs_tensor[0].shape)

#编码器encoder
class EncoderRNN(nn.Module):
    """

    """
    def __init__(self, input_size, hidden_size):
        """

        :param input_size: 原语言的词表大小
        :param hidden_size: 隐藏层大小,即词嵌入维度,同时也是GRU的输出尺寸
        """
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size) #embedding层
        self.gru = nn.GRU(hidden_size, hidden_size) #GRU层,这里输入只有1个词,因此layer=1
    def forward(self, input, hidden):

        input=input.to(device)
        hidden=hidden.to(device)
        self.embedding=self.embedding.to(device)
        self.gru=self.gru.to(device)

        output= self.embedding(input) #将输入的单词索引转换为词嵌入向量
        #编码器每次只以一个词作为输入,因此词汇映射后的尺寸为 1*n
        output=output.view(1, 1, -1) #将词嵌入向量转换为1*1*n的形状,因为torch中的GRU的输入是3维的(seq_len, batch, input_size)
        output=output.to(torch.device("cuda:0"))
        output, hidden = self.gru(output, hidden) #将词嵌入向量传入GRU层
        return output, hidden
    def initHidden(self):
        return torch.zeros(1,1,self.hidden_size,device=device)

#实例化GRU网络
input_size, hidden_size=20,25
input=pairs_tensor[0][0] #英语第一个词
print(input.shape)
encoder=EncoderRNN(input_size, hidden_size)
output, hidden=encoder(input,torch.zeros(1,1,hidden_size))
print(output.shape,hidden.shape)
#保存encoder的输出,用于后续decoder解码
encoder_outputs=output


#基于GRU的解码器实现
class DecoderRNN(nn.Module):
    """

    """
    def __init__(self,  hidden_size,output_size):
        """
        :param hidden_size: 解码器输入尺寸,也是隐藏层大小
        :param output_size: 输出层大小,即目标语言词表大小
        """
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size,hidden_size ) #embedding层
        self.gru = nn.GRU(hidden_size, hidden_size,1) #GRU层,这里输入只有1个词,因此layer=1
        self.out= nn.Linear(hidden_size, output_size) #全连接层
        self.softmax = nn.LogSoftmax(dim=1) #softmax层
    def forward(self, input, hidden):
        self.embedding=self.embedding.to(device)
        self.gru=self.gru.to(device)
        self.softmax=self.softmax.to(device)
        self.out=self.out.to(device)
        hidden=hidden.to(device)
        input=input.to(device)
        output = self.embedding(input).view(1, 1, -1)
        # 然后使用relu函数对输出进行处理，根据relu函数的特性, 将使Embedding矩阵更稀疏，以防止过拟合
        output = F.relu(output)
        # 接下来, 将把embedding的输出以及初始化的hidden张量传入到解码器gru中
        output, hidden = self.gru(output, hidden)
        # 因为GRU输出的output也是三维张量，第一维没有意义，因此可以通过output[0]来降维
        # 再传给线性层做变换, 最后用softmax处理以便于分类
        output = self.softmax(self.out(output[0]))
        return output, hidden
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device) #初始化隐藏状态

hidden_size=23
output_size = 10
decoder=DecoderRNN(hidden_size, output_size).to(device)
#法语第一个词
input=pairs_tensor[1][0]
hidden=decoder.initHidden()
output,hidden=decoder(input,hidden)
print('output.shape',output.shape,'hidden.shape',hidden.shape)
# output.shape torch.Size([1, 10]) hidden.shape torch.Size([1, 1, 23])

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        """

        :param hidden_size: GRU输入大小
        :param output_size:  decoder输出大小
        :param dropout_p:  dropout概率
        :param max_length: 句子最大长度
        """
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        #embedding层,atten不改变输出尺寸,因此embedding层的输出尺寸即hidden_size,输入大小即output_size(即词表大小
        self.embedding = nn.Embedding(output_size, hidden_size).to(device)
        #这里用注意力机制的第一种模式,拼接Q,K做线性变换再做softmax,因此输入维度是hidden_size*2
        self.attn=nn.Linear(self.hidden_size*2,self.max_length).to(device)

        #拼接embedding后的input和bmm的输出结果(2*hidden_size)
        self.attn_combine=nn.Linear(self.hidden_size*2,self.hidden_size).to(device)

        self.gru=nn.GRU(self.hidden_size,self.hidden_size).to(device)
        self.dropout=nn.Dropout(self.dropout_p).to(device)
        #输出层
        self.out=nn.Linear(self.hidden_size,output_size).to(device)

    def forward(self, input, hidden, encoder_outputs):
        """

        :param input:输入
        :param hidden:上一decoder的隐藏层传入
        :param encoder_outputs:
        :return:
        """
        #embedding层
        input=input.to(device)
        hidden=hidden.to(device)
        encoder_outputs=encoder_outputs.to(device)

        embedded=self.embedding(input).view(1,1,-1)
        embedded=self.dropout(embedded)

        #Q,K拼接做lianer再做softmax
        attn_weight=F.softmax(#拼接embedded和hidden
            self.attn(torch.cat((embedded[0],hidden[0]),1)),dim=1
        )
        ##bmm计算注意力权重,
        attn_applied=torch.bmm(attn_weight.unsqueeze(0),encoder_outputs.unsqueeze(0))

        output=torch.cat((embedded[0],attn_applied[0]),1)#拼接
        attn_combine=torch.relu(self.attn_combine(output))
        output,hidden=self.gru(attn_combine.unsqueeze(0),hidden)#GRU
        output=torch.log_softmax(self.out(output[0]),dim=1)#输出层
        #返回解码器结果
        return output,hidden,attn_weight

    # 这个部分有问题,事实上初始化Decoder的hidden应使用Encoder输出的hidden而非自己init
    def initHidden(self):
        return torch.zeros(1,1,self.hidden_size,device=device)


hidden_size = 25
output_size = 10 #词表长度,这里设为10
#取出一个法文单词
input=pairs_tensor[1][0]
encoder_outputs=torch.randn(10,hidden_size)
decoder_attn=AttnDecoderRNN(hidden_size,output_size,dropout_p=0.1)
hidden=decoder_attn.initHidden()
#解码
output_word,hidden,attn_weights=decoder_attn(input,hidden,encoder_outputs)
print("output_word",output_word)

#teacher_forcing 强制修正上一步输出结果为正确,从而避免错错误积累

def train(input_tensor,target_tensor,encoder,decoder,encoder_optimizer,decoder_optimizer,criterion,teacher_forcing_ratio,max_length=MAX_LENGTH):
    """
    即使是带attention的版本也可以用相同的train函数,因为多出来的attn_weights参数其实在训练过程中无作用
    :param input_tensor: 源语言tensor
    :param target_tensor: 目标语言tensor
    :param encoder:
    :param decoder:
    :param encoder_optimizer: 优化器
    :param decoder_optimizer:
    :param criterion: loss_function
    :param max_length: 语句最大长度
    :param teacher_forcing_ratio: 老师修正概率
    :return:
    """

    #初始化隐藏层
    encoder_hidden=encoder.initHidden()

    #编码器和解码器的优化器梯度归零
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    #获取文本tensor的长度
    input_length=input_tensor.size(0)
    target_length=target_tensor.size(0)

    #初始化encoder_output,二维tensor用于存储每一个字符的encoder结果
    encoder_outputs=torch.zeros(max_length,encoder.hidden_size,device=device)

    loss=0

    #遍历encoder_tensor索引
    for ei in range(input_length):
        #根据索引取出对应的单词tensor,放进encoder中进行encode
        encoder_output,encoder_hidden=encoder(input_tensor[ei],encoder_hidden)
        #将encoder_output存入encoder_outputs中,[0,0] 即 [1,1,hidden_size]_>[hidden_size]
        encoder_outputs[ei]=encoder_output[0,0]

    #初始化decoder输入,即[SOS_token],初始字符
    decoder_input=torch.tensor([[SOS_token]],device=device)

    #初始化解码器的hidden,即encoder输出的hidden
    decoder_hidden=encoder_hidden

    #根据随机数与teacher_forcing_ratio比较,决定本次训练是否使用teacher_forcing
    #如果设置teacher_forcing_ratio=1,则必定使用teacher_forcing
    use_teacher_forcing=(random.random()<teacher_forcing_ratio)

    #触发teacher_forcing
    if use_teacher_forcing:
        for i in range(target_length):
            #attention中的q=decoder_input,k=decoder_hidden,v=encoder_outputs
            #数据进入decoder,下一轮的input即本轮的output
            decoder_input,decoder_hidden,decoder_attention=decoder(decoder_input,decoder_hidden,encoder_outputs)
            #将decoder_output与target_tensor[i]进行对比,计算loss
            loss+=criterion(decoder_input,target_tensor[i])
            decoder_input=target_tensor[i]  # Teacher forcing,强制更换下一轮的输入
    else:
        for i in range(target_length):
            #attention中的q=decoder_input,k=decoder_hidden,v=encoder_outputs
            #数据进入decoder,下一轮的input即本轮的output
            decoder_output,decoder_hidden,decoder_attention=decoder(decoder_input,decoder_hidden,encoder_outputs)
            #将decoder_output与target_tensor[i]进行对比,计算loss
            loss+=criterion(decoder_output,target_tensor[i])
            # decoder_input=target_tensor[i]  # 不进行Teacher forcing

            #将decoder_output中概率最大的输出作为下一轮的输入
            topv,topi=decoder_output.topk(1)
            if topi.squeeze().item()==EOS_token:#如果模型预测输出为EOS_token(结束符),则退出循环
                break
            #这里.detach使得decoder_input与原来的decoder_output分离,相当于全新的外界输入
            #相当于固定写法
            decoder_input=topi.squeeze().detach() #将decoder_input更新为模型预测的输出

    #反向传播
    loss.backward()
    #用optimizer直接更新参数
    encoder_optimizer.step()
    decoder_optimizer.step()
    #返回平均loss
    return loss.item()/target_length
def time_since(since): #计时
    s=time.time() - since
    return 'since  %d minutes %d seconds' % (s//60, s%60)

teacher_forcing_ratio=0.5

def trainIters(encoder,decoder,n_iters,print_every=5000,plot_every=100,lr=0.1):
#def trainIters(encoder,decoder,encoder_optimizer,decoder_optimizer,criterion,n_iters,print_every=1000,plot_every=100,lr=0.1):

    """

    :param encoder:
    :param decoder:
    :param encoder_optimizer:
    :param decoder_optimizer:
    :param criterion:
    :param n_iters: 迭代步数
    :param print_every: 日志打印间隔
    :param plot_every: loss打印间隔
    :return:
    """
    since=time.time()

    plt_losses=[]

    print_loss_total=0 #打印间隔的loss
    plot_loss_total=0 #绘制间隔的loss
    #优化器和损失函数
    encoder_optimizer=optim.SGD(encoder.parameters(),lr=lr)
    decoder_optimizer=optim.SGD(decoder.parameters(),lr=lr)
    criterion=nn.NLLLoss()
    #训练
    for iter in range(1,n_iters+1):
        #每次训练随机选一条语句
        training_pair=tensorFromPair(random.choice(pairs))
        #获取输入和输出
        input_tensor,target_tensor=training_pair
        loss=train(input_tensor,target_tensor,encoder,decoder,encoder_optimizer,decoder_optimizer,criterion,iter)

        #loss++
        print_loss_total +=loss
        plot_loss_total += loss
        #到了打印间隔
        if iter%print_every==0:

            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0#损失归0
            #打印 耗时
            print('耗时:%s (迭代步数:%d 当前迭代步数百分比:%.2f%%) 当前平均损失: %.4f' % (time_since(since), iter, iter / n_iters * 100, print_loss_avg))
        if iter%plot_every==0:
            #plt_losses加上损失值
            plt_losses.append(plot_loss_total / plot_every)
            plot_loss_total = 0#损失归0

    #绘制损失曲线
    plt.figure()
    plt.plot(plt_losses)
    plt.savefig("./img/s2s_loss.png")
    plt.show()


hidden_size=256
encoder1=EncoderRNN(input_lang.n_word,hidden_size).to(device)
attn_decoder1=AttnDecoderRNN(hidden_size, output_lang.n_word, dropout_p=0.1).to(device)
n_iters=75000

#迭代训练
#模型保存位置
PATHENCODER='./model/seq2seq_eng2fra_encoder.pth'
PATHDECODER='./model/seq2seq_eng2fra_decoder.pth'
PATH_ATTN_DECODER='./model/seq2seq_eng2fra_attn_decoder1.pth'
trainIters(encoder1,attn_decoder1,n_iters=n_iters)
# 训练
# 调用trainIters进行模型训练，将编码器对象encoder1，解码器对象attn_decoder1，迭代步数，日志打印间隔传入其中
# trainIters(encoder1, attn_decoder1, n_iters, print_every=print_every)
# #保存模型
# torch.save(encoder1.state_dict(), PATHENCODER)
# torch.save(attn_decoder1.state_dict(),PATH_ATTN_DECODER)
#
# #读取模型
# encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
# attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
#
# encoder1.load_state_dict(torch.load(PATHENCODER))
# # rnn_1=rnn_1.load_state_dict(torch.load(PATHRNN)) #错误做法,返回_IncompatibleKeys这个对象
# attn_decoder1.load_state_dict(torch.load(PATH_ATTN_DECODER))