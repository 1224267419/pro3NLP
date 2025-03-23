from io import open
import unicodedata
import re
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from openpyxl.styles.builtins import output
from torch import optim

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
        output=output.view(1, 1, -1).to(device) #将词嵌入向量转换为1*1*n的形状,因为torch中的GRU的输入是3维的(seq_len, batch, input_size)
        output, hidden = self.gru(output, hidden) #将词嵌入向量传入GRU层
        return output, hidden
#实例化GRU网络
input_size, hidden_size=20,25
input=pairs_tensor[0][0] #英语第一个词
print(input.shape)
encoder=EncoderRNN(input_size, hidden_size)
output, hidden=encoder(input,torch.zeros(1,1,hidden_size))
print(output.shape,hidden.shape)


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

