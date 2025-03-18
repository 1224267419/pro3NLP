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
import time

from openpyxl.styles.builtins import output
from xgboost.dask import predict


def time_since(since): #计时
    return time.time() - since


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
        #logsoftmax仅仅只是多了个求对数的操作
        self.softmax=nn.LogSoftmax(dim=-1)
    def forward(self, input, hidden):
        #扩充维度,从(1,57) ->(1,1,57)
        input=input.unsqueeze(0)
        #RNN层,输入为input和hidden,输出为output和hidden
        rr,hn=self.rnn(input,hidden)
        #rnn返回的结果通过线性变换和sofrmanx返回,同时返回hn作为后续rnn的输入
        return self.softmax(self.linear(rr)),hn
    def initHidden(self):
        """初始化隐藏层"""
        return torch.randn(self.n_layers,1,self.hidden_size)

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        """ 分别代表RNN输入维度 , 输入最后一维尺寸,输出尺寸, RNN层数"""
        super(LSTM, self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.n_layers= n_layers
        #RNN层,输入维度为input_size,输出维度为hidden_size,层数为n_layers
        self.lstm=nn.LSTM(input_size, hidden_size, n_layers)
        #线性层,用于将RNN输出维度转化为指定维度
        self.linear=nn.Linear(hidden_size, output_size)
        #softmax层,用于输出类别(在最后一个维度上进行softmax,输出结果维度和最后一个维度大小相同
        self.softmax=nn.LogSoftmax(dim=-1)
    def forward(self, input, hidden,c):
        #扩充维度,从(1,57) ->(1,1,57)
        input=input.unsqueeze(0)
        #RNN层,输入为input和hidden,输出为output和hidden=hidden+细胞状态c
        rr,(hn,c)=self.lstm(input,(hidden,c))
        #rnn返回的结果通过线性变换和sofrmanx返回类别结果,同时返回hn作为后续rnn的输入
        return self.softmax(self.linear(rr)),(hn,c)
    def initHidden(self):
        """初始化隐藏层"""
        c=hidden=torch.randn(self.n_layers,1,self.hidden_size)
        return hidden,c

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        """ 分别代表RNN输入维度 , 输入最后一维尺寸,输出尺寸, RNN层数"""
        super(GRU, self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.n_layers= n_layers
        #RNN层,输入维度为input_size,输出维度为hidden_size,层数为n_layers
        self.gru=nn.GRU(input_size, hidden_size, n_layers)
        #线性层,用于将RNN输出维度转化为指定维度
        self.linear=nn.Linear(hidden_size, output_size)
        #softmax层,用于输出类别(在最后一个维度上进行softmax,输出结果维度和最后一个维度大小相同
        self.softmax=nn.LogSoftmax(dim=-1)
    def forward(self, input1, hidden):
        #扩充维度,从(1,57) ->(1,1,57)
        input1=input1.unsqueeze(0)
        #RNN层,输入为input和hidden,输出为output和hidden
        rr,hn=self.gru(input1, hidden)
        #rnn返回的结果通过线性变换和sofrmanx返回,同时返回hn作为后续rnn的输入
        return self.softmax(self.linear(rr)),hn
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

def categoryFromOutput(output,all_categories):
    "根据输出得到类别"
    top_n, top_i = output.topk(1)
    #largest：如果为True，按照大到小排序； 如果为False，按照小到大排序 k：指明是得到前k个数据以及其index。
    #topk最常用的场合就是求一个样本被网络认为前k个最可能属于的类别。
    #top_i对象中取出索引值
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

def randomTrainingExample(all_categories,category_lines):
    #随机从字典中选择一个类别
    category = random.choice(all_categories)
    #从选择的类别中随机选一个名字
    line = random.choice(category_lines[category])
    category_tensor=torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category,line,category_tensor,line_tensor

def trainRNN(rnn,category_tensor,line_tensor,learning_rate=0.005):
    # 构建传统的RNN训练函数
    # 定义损失函数为nn.NLLLoss，因为RNN的最后一层是nn.LogSoftmax, 两者的内部计算逻辑正好能够吻合.
    criterion = nn.NLLLoss()

    #初始化rnn参数
    hidden= rnn.initHidden()
    #梯度归0
    rnn.zero_grad()
    #rnn遍历所有的字符,从而得到输出结果
    for i in range(line_tensor.size()[0]): #遍历每一个字母
        output,hidden=rnn(line_tensor[i],hidden)
    #计算损失
    loss = criterion(output.squeeze(0), category_tensor)
    #反向传播
    loss.backward()
    #更新参数
    for p in rnn.parameters():
        #所有参数-lr*(梯度)
        p.data.add_(-learning_rate, p.grad.data)
    return output, loss.item()

def trainLSTM(lstm,category_tensor,line_tensor,learning_rate=0.005):
    # 构建传统的RNN训练函数
    # 定义损失函数为nn.NLLLoss，因为RNN的最后一层是nn.LogSoftmax, 两者的内部计算逻辑正好能够吻合.
    criterion = nn.NLLLoss()

    #初始化rnn参数
    hidden,c= lstm.initHidden()
    #梯度归0
    lstm.zero_grad()
    #rnn遍历所有的字符,从而得到输出结果
    for i in range(line_tensor.size()[0]): #遍历每一个字母
        output,(hidden,c)=lstm(line_tensor[i],hidden,c)
    #计算损失
    loss = criterion(output.squeeze(0), category_tensor)
    #反向传播
    loss.backward()
    #更新参数
    for p in lstm.parameters():
        #所有参数-lr*(梯度)
        p.data.add_(-learning_rate, p.grad.data)
    return output, loss.item()

def trainGRU(gru, category_tensor, line_tensor, learning_rate=0.005):
    # 构建传统的RNN训练函数
    # 定义损失函数为nn.NLLLoss，因为RNN的最后一层是nn.LogSoftmax, 两者的内部计算逻辑正好能够吻合.
    criterion = nn.NLLLoss()

    #初始化rnn参数
    hidden= gru.initHidden()
    #梯度归0
    gru.zero_grad()
    #rnn遍历所有的字符,从而得到输出结果
    for i in range(line_tensor.size()[0]): #遍历每一个字母
        output,hidden=gru(line_tensor[i], hidden)
    #计算损失
    loss = criterion(output.squeeze(0), category_tensor)
    #反向传播
    loss.backward()
    #更新参数
    for p in gru.parameters():
        #所有参数-lr*(梯度)
        p.data.add_(-learning_rate, p.grad.data)
    return output, loss.item()


    # 构建训练过程的日志打印函数

def train(nn,train_type_fn,all_categories,category_lines,epoch=10000):
    """训练日志打印函数,train_type_fn表示选择用什么模型训练函数,
    如trainRNN,trainLSTM,trainGRU等"""
    # 设置结果的打印间隔
    print_every = 50
    # 设置绘制损失曲线上的制图间隔
    plot_every = 10

    # 每个制图间隔损失保存列表
    all_losses = []
    # 保存制图使用
    all_train_acc = []
    # 获得训练开始时间戳
    start = time.time()
    # 设置初始间隔损失为0
    current_loss = 0
    # 添加======
    current_acc = 0
    # 从1开始进行训练迭代, 共n_iters次

    for iter in range(1, epoch + 1):
        # 通过randomTrainingExample函数随机获取一组训练数据和对应的类别
        category, line, category_tensor, line_tensor = randomTrainingExample(all_categories,category_lines)
        # 将训练数据和对应类别的张量表示传入到train函数中
        output, loss = train_type_fn(nn,category_tensor, line_tensor)
        # 计算制图间隔中的总损失
        current_loss += loss
        # 改造===
        # 取该迭代步上的output通过categoryFromOutput函数获得对应的类别和类别索引
        guess, guess_i = categoryFromOutput(output,all_categories)
        current_acc += 1 if guess == category else 0

        # 如果迭代数能够整除打印间隔
        if iter % print_every == 0:
            # 然后和真实的类别category做比较, 如果相同则打对号, 否则打叉号.
            correct = '✓' if guess == category else '✗ (%s)' % category
            # 打印迭代步, 迭代步百分比, 当前训练耗时, 损失, 该步预测的名字, 以及是否正确
            print('%d %d%% (%s) %.4f %s / %s %s|| acc:%.4f' % (
            iter, iter / epoch * 100, time_since(start), loss, line, guess, correct, current_acc / print_every))
            all_train_acc.append(current_acc / print_every)
            current_acc = 0

        # 如果迭代数能够整除制图间隔
        if iter % plot_every == 0:
            # 将保存该间隔中的平均损失到all_losses列表中
            all_losses.append(current_loss / plot_every)
            # 间隔损失重置为0
            current_loss = 0

    # return current_acc / n_iters
    # 返回对应的总损失列表和训练耗时
    return all_losses, all_train_acc, int(time.time() - start)

def evaluateRNN(rnn,line_tensor):
    """RNN评估函数,line_tensor表示名字对应的张量"""
    hidden = rnn.initHidden() #初始隐藏输入
    for i in range(line_tensor.size()[0]):
        output,hidden=rnn(line_tensor[i],hidden)
        #输出概率
    return output.squeeze(0)

def evaluateLSTM(lstm,line_tensor):
    hidden,c=lstm.initHidden()
    for i in range(line_tensor.size()[0]):
        output,(hidden,c)=lstm(line_tensor[i],hidden,c)
    return output.squeeze(0)

def evaluateGRU(gru,line_tensor):
    hidden=gru.initHidden()
    for i in range(line_tensor.size()[0]):
        output,hidden=gru(line_tensor[i],hidden)
    return output.squeeze(0)



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
    # print(all_category) #打印国家类别数
    # print(category_lines['Irish'][:10]) #打印前十个瑞士名字
    #
    # #onehot编码将名字转换为张量
    # print(lineToTensor('Jones')) #将名字转换为张量
    # print(lineToTensor('Jones').size()) #打印张量大小 torch.Size([len, 1, 57]),len为名字长度

    input_size=len(all_letters) #类别数
    hidden_size=128 #隐藏层大小
    output_size=len(all_category) #输出大小
    num_layer=3 #隐藏层层数

    input=lineToTensor(category_lines['Irish'][:5]) #将名字转换为张量
    #hidden
    hidden=c=torch.randn(num_layer,1,hidden_size) #随机生成隐藏层张量

    rnn=RNN(input_size,hidden_size,output_size,num_layer) #定义RNN
    lstm=LSTM(input_size,hidden_size,output_size,num_layer) #定义LSTM
    gru=GRU(input_size,hidden_size,output_size,num_layer) #定义GRU
    # rnn_outp ut,next_hidden=rnn(input,hidden) #RNN
    # print('rnn',rnn_output,next_hidden,categoryFromOutput(rnn_output[0],all_category))
    #
    # lstm_output,(next_hidden,c)=lstm(input,hidden,c) #LSTM
    # print('lstm',lstm_output,next_hidden,c,categoryFromOutput(rnn_output[0],all_category))
    #
    # gru_output,next_hidden=gru(input,hidden) #GRU
    # print('gru',gru_output,next_hidden,categoryFromOutput(rnn_output[0],all_category))
    #

    #
    # for i in range(10):
    #     category, line, category_tensor, line_tensor=randomTrainingExample(all_category,category_lines)
    #     print('category=',category,'/line=',line,'\n')
    # print(line_tensor)
    # print("训练RNN")
    # for i in range(10):
    #     category, line, category_tensor, line_tensor = randomTrainingExample(all_category, category_lines)
    #     # print(category_tensor.size())
    #     output, loss=trainRNN(rnn, category_tensor, line_tensor, 0.005)
    #     print('i=',i,output,loss)
    # print("训练LSTM")
    # for i in range(10):
    #     category, line, category_tensor, line_tensor = randomTrainingExample(all_category, category_lines)
    #     # print(category_tensor.size())
    #     output, loss = trainLSTM(lstm, category_tensor, line_tensor, 0.005)
    #     print('i=', i, output, loss)
    # print("训练GRU")
    # for i in range(10):
    #     category, line, category_tensor, line_tensor = randomTrainingExample(all_category, category_lines)
    #     # print(category_tensor.size())
    #     output, loss = trainGRU(gru, category_tensor, line_tensor, 0.005)
    #     print('i=', i, output, loss)


    #假设模型在600s前开始训练
    s=time.time() - 600
    print(time_since(s))
    # 模型训练和保存
    # #训练函数
    # loss_RNN,acc_RNN,time_RNN=train(rnn, trainRNN, all_category, category_lines)
    # loss_LSTM,acc_LSTM,time_LSTM=train(lstm,trainLSTM ,all_category, category_lines)
    # loss_GRU,acc_GRU,time_GRU=train(gru,trainGRU ,all_category, category_lines)
    #
    # # 创建画布0
    # plt.figure(0)
    # # 绘制损失对比曲线
    # plt.plot(loss_RNN, label="RNN")
    # plt.plot(loss_LSTM, color="red", label="LSTM")
    # plt.plot(loss_GRU, color="orange", label="GRU")
    # plt.legend(loc='upper left')
    # plt.savefig('./img/RNN_LSTM_GRU_loss.png')
    #
    # # 创建画布1
    # plt.figure(1)
    # x_data = ["RNN", "LSTM", "GRU"]
    # y_data = [time_RNN, time_LSTM, time_GRU]
    # # 绘制训练耗时对比柱状图
    # plt.bar(range(len(x_data)), y_data, tick_label=x_data)
    # plt.savefig('./img/RNN_LSTM_GRU_period.png')
    # plt.show()
    #
    # # 保存模型
    PATHRNN = './model/name_rnn.pth'
    # torch.save(rnn.state_dict(), PATHRNN)
    #
    PATHLSTM = './model/name_lstm.pth'
    # torch.save(lstm.state_dict(), PATHLSTM)
    #
    PATHGRU = './model/name_gru.pth'
    # torch.save(gru.state_dict(), PATHGRU)

    #读取模型
    PATHRNN = './model/name_rnn.pth'
    PATHLSTM = './model/name_lstm.pth'
    PATHGRU = './model/name_gru.pth'

    rnn_1=RNN(input_size,hidden_size,output_size,num_layer) #定义RNN
    lstm_1=LSTM(input_size,hidden_size,output_size,num_layer) #定义LSTM
    gru_1=GRU(input_size,hidden_size,output_size,num_layer) #定义GRU

    rnn_1.load_state_dict(torch.load(PATHRNN))
    # rnn_1=rnn_1.load_state_dict(torch.load(PATHRNN)) #错误做法,返回_IncompatibleKeys这个对象
    lstm_1.load_state_dict(torch.load(PATHLSTM))
    gru_1.load_state_dict(torch.load(PATHGRU))

    #测试
    print("RNN_output", evaluateRNN(rnn_1, lineToTensor("Bob")))
    print("LSTM_output", evaluateLSTM(lstm_1, lineToTensor("Bob")))
    print("GRU_output", evaluateGRU(gru_1, lineToTensor("Bob")))

    def predict(input_line, model,evaluate, all_category,n_predictions=3):
        """

        :param all_category: 国家类别
        :param input_line:预测的名字
        :param n_predictions: 可能性最高的n个项
        :return:
        """
        print(input_line)
        #predict和value不产生梯度
        input_tensor=lineToTensor(input_line)
        with torch.no_grad():
            # 使输入的名字转换为张量表示, 并使用evaluate函数获得预测输出
            output=evaluate(model,input_tensor)

            topv,topi=output.topk(n_predictions,1,True,True) #返回值是两个张量,第一个张量是n_predictions个最大的值,第二个张量是这些值对应的索引
            #结果list
            predictions=[]
            for i in range(n_predictions):
                #从topv取出output
                value=topv[0][i].item()
                category_index=topi[0][i].item()
                #打印output及其对应的类别
                print('(%.2f) %s'% (value,all_category[category_index]))
                predictions.append((value,all_category[category_index]))


    import inspect, re


    def varname(p): #输出变量名
        for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
            m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
            if m:
                return m.group(1)




    for evaluate_fn in [[rnn,evaluateRNN,"RNN"],[lstm,evaluateLSTM,'LSTM'],[gru,evaluateGRU,'GRU']]:
        print("-"*40)#分隔符
        print(evaluate_fn[2])#输出RNN,LSTM,GRU
        predict('Dovesky',evaluate_fn[0],evaluate_fn[1],all_category)
        predict('Jackson',evaluate_fn[0],evaluate_fn[1],all_category)
        predict('Satoshi',evaluate_fn[0],evaluate_fn[1],all_category)
