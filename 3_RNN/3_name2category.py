from io import open
import glob
import os
import string
import unicodedata
import random
import time
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        """ 分别代表RNN输入维度 , 输入最后一维尺寸,输出尺寸, RNN层数"""
        super(RNN, self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.n_layers= n_layers
        #RNN层,输入维度为input_size,输出维度为hidden_size,层数为n_layers
        self.rnn=nn.RNN(input_size, hidden_size, n_layers)
        #线性层,用于将RNN输出维度转化为指定维度
        self.linear=nn.Linear(hidden_size, output_size)
        #softmax层,用于输出类别(在最后一个维度上进行softmax,输出结果维度和最后一个维度大小相同
        self.softmax=nn.LogSoftmax(dim=-1)
    def forward(self, input, hidden):
        #扩充维度,从(1,57) ->(1,1,57)
        input=input.unsqueeze(0)
        #RNN层,输入为input和hidden,输出为output和hidden
        rr,hn=self.rnn(input,hidden)
        #rnn返回的结果通过线性变换和sofrmanx返回,同时返回hn作为后续rnn的输入
        return self.softmax(self.linear(rr[0])),hn
    def initHidden(self):
        """初始化隐藏层"""
        return torch.randn(self.n_layers,1,self.hidden_size)




def unicodeToAscii(s): #消除重音
#Mn Me和Mc 是附加符号的标记
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn' #如果出现Mn,说明没有规范化
                        and c in all_letters)
def readLines(filename):
    """读取文件中的每一行数据并组成列表"""
    # strip()去除空白符 ,split('\n')按行分割
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

def lineToTensor(line):
    '''人名转化为对应onehot tensor
    :param line:人名'''
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        #字符串在all_letters中的索引置1
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor
if __name__ == '__main__':
    # 第二步: 对data文件中的数据进行处理，满足训练要求.
    # 获取所有常用字符包括字母和常用标点
    all_letters = string.ascii_letters + " .,;'"
    n_letters = len(all_letters)
    # print(all_letters) #常用字符:a-z A-Z .,;'


    # #消除重音
    #  s = "Ślusàrski"
    #  print(s)
    #  a = unicodeToAscii(s)
    #  print(a)

    data_path='./data/names/'
    # res=readLines(data_path+'french.txt')
    # print(res[:20])

    # 构建的category_lines形如：{"English":["Lily", "Susan", "Kobe"], "Chinese":["Zhang San", "Xiao Ming"]}
    category_lines={}
    # 所有国家类别
    all_category=[]
    #正则表达式遍历所有文件
    # 读取指定路径下的txt文件， 使用glob，path中可以使用正则表达式
    # glob资料：https://blog.csdn.net/qq_17753903/article/details/82180227
    for filename in glob.glob(data_path+'*.txt'):
        #取出最后一个文件名,即对应的名字 *.txt部分,文件名按.进行切割,最后取出的即国家名 *的部分
        category=os.path.splitext(os.path.basename(filename))[0]
        #国籍装入all_category中
        all_category.append(category)
        #读取文件内容,组成名字列表
        lines=readLines(filename)
        category_lines[category]=lines #字典,键为category,值为lines
    print(all_category) #打印国家类别数
    print(category_lines['Irish'][:10]) #打印前十个瑞士名字

    #onehot编码将名字转换为张量
    print(lineToTensor('Jones')) #将名字转换为张量
    print(lineToTensor('Jones').size()) #打印张量大小 torch.Size([len, 1, 57]),len为名字长度
