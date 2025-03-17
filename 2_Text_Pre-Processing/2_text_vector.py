import fileinput
from os import write

import joblib
from tensorflow.keras.preprocessing.text import Tokenizer
import fasttext
from torch.nn.functional import embedding


def train_onehot_encode():
    vocab = {"周杰伦", "陈奕迅", "王⼒宏", "李宗盛", "吴亦凡", "⿅晗"}
    t = Tokenizer(num_words=None, char_level=False)
    #拟合现有⽂本
    t.fit_on_texts(vocab)

    for token in vocab:
        zero_list = [0] * len(vocab)

        print(t.texts_to_sequences([token]))
        # 使⽤映射器转化现有⽂本数据, 每个词汇对应从1开始的⾃然数
        # 返回样式如: [[2]], 取出其中的数字需要使⽤[0][0]
        token_index = t.texts_to_sequences([token])[0][0] - 1
        #��Ӧλ�ø�ֵ
        zero_list[token_index] = 1
        print(token, '的one-hot编码为:', zero_list)
    joblib.dump(t, './data/t.pkl') # 保存映射器

def test_onehot_encode():
    t = joblib.load('./data/t.pkl')
    token="王⼒宏"
    token_index = t.texts_to_sequences([token])[0][0] - 1
    print(token, '的one-hot编码为:', t.word_index[token])
    zero_list=[0]*len(t.word_index)
    zero_list[token_index] = 1
    print(zero_list)

def vector_vocabulary():
    # model = fasttext.train_unsupervised('./data/fil9') #训练模型.i711800h训练需要30min
    # model2 = fasttext.train_unsupervised('./data/fil9', model='cbow', dim=200, epoch=10, lr=0.1)  # 设置更多超参数
    model = fasttext.load_model('./data/fil9_model.bin')
    print(model.get_word_vector('the')) # 查看单词向量
    # 查找最相似的单词,从而检查向量化是否合理
    print(  model.get_nearest_neighbors('sport'))
    print(  model.get_nearest_neighbors('dog'))
    # 模型保存
    model.save_model('./data/fil9_model.bin')
    # 模型读取
    model2 = fasttext.load_model('./data/fil9_model.bin')
    print(model2.get_nearest_neighbors('dog'))
    # 检查模型是否发生变化

def word_embedding(): #可视化
    import torch
    import json
    from torch.utils.tensorboard import SummaryWriter

    # 随机初始化⼀个100x50的矩阵, 认为它是我们已经得到的词嵌⼊矩阵
    # 代表100个词汇, 每个词汇被表示成50维的向量
    writer=SummaryWriter()
    embedded=torch.randn(100,50)

    # 导⼊事先准备好的100个中⽂词汇⽂件, 形成meta列表原始词汇
    meta=list(map(lambda  x: x.strip(), fileinput.FileInput("./data/vocab100.csv")))
    writer.add_embedding(embedded,metadata=meta)
    writer.close()

if __name__== '__main__':
    # train_onehot_encode()
    # test_onehot_encode()
    # vector_vocabulary()
    word_embedding()