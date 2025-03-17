import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from itertools import chain
import jieba
import jieba.posseg as pseg #词性标注
from wordcloud import WordCloud
# 使⽤jieba中的词性标注功能
import jieba.posseg as pseg
def get_a_list(text):
     """⽤于获取形容词列表"""
     # 使⽤jieba的词性标注⽅法切分⽂本,获得具有词性属性flag和词汇属性word的对象,
     # 从⽽判断flag是否为形容词,来返回对应的词汇
     r = []
     for g in pseg.lcut(text): #lcut()返回一个生成器,
         if g.flag == "a": #形容词
            r.append(g.word) #g.word才是string词汇,g是类
     return r
# 导⼊绘制词云的⼯具包
from wordcloud import WordCloud
# 实例化绘制词云的类
def get_word_cloud(keywords_list):
     #fomt_path是字体路径, 为了能够显示中⽂, max_words指词云图像最多显示多少个词, background_color为背景颜⾊
     wordcloud = WordCloud(font_path="./hotel_cn_data/SimHei.ttf", max_words=100,
                           background_color="white")
     # 将传⼊的列表转化成词云⽣成器需要的字符串形式,list组装成一句话
     keywords_string = " ".join(keywords_list)
     # ⽣成词云
     wordcloud.generate(keywords_string)
     # 绘制图像并显示
     plt.figure()
     plt.imshow(wordcloud, interpolation="bilinear")
     plt.axis("off") #无需坐标
     plt.show()




def show_data_lenth(data):
    sns.countplot('sentence_length', data=data)
    plt.xticks([])  # 观察分布纵坐标,因此无需横坐标
    plt.show()
    # 观察长度分布横坐标,因此无需纵坐标
    sns.distplot(data['sentence_length'])
    plt.yticks([])
    plt.show()
if __name__ == '__main__':
    train_data=pd.read_csv('./hotel_cn_data/train.tsv', sep='\t')
    val_data=pd.read_csv('./hotel_cn_data/dev.tsv', sep='\t')

    sns.countplot('label', data=train_data)
    plt.show()
    sns.countplot('label', data=val_data)
    plt.show()
    train_data['sentence_length'] = list(map(lambda x: len(x), train_data['sentence']))
    val_data['sentence_length'] = list(map(lambda x: len(x), val_data['sentence']))
    #句子长度分布图 ,根据图片,长度集中在0-250之间
    # show_data_lenth(train_data)
    # show_data_lenth(test_data)

    # # 长度分布散点图
    # sns.stripplot(y='sentence_length', x='label',data=train_data)
    # plt.show()
    # sns.stripplot(y='sentence_length', x='label', data=val_data)
    # plt.show()
    # '''
    # 通过散点图,可以清晰地找出异常点的存在,从而实现人工审查
    # '''
    #结巴分词对句子进行分解
    train_vocab=set(chain(*map(lambda x: jieba.lcut(x), train_data['sentence'])))
    val_vocab=set(chain(*map(lambda x: jieba.lcut(x), val_data['sentence'])))
    print('训练集包含词汇总数为: ',len(train_vocab))
    print('验证集包含词汇总数为: ',len(val_vocab))

    # 获得训练集上正样本
    p_train_data = train_data[train_data["label"] == 1]["sentence"]
    # 对正样本的每个句⼦的形容词进行统计 ,chain() 的一个常见场景是当你想对不同的集合中所有元素执行某些操作
    # 这里让train_p_a_vocab作为一个包含所有正面含义词语的迭代器
    train_p_a_vocab = chain(*map(lambda x: get_a_list(x), p_train_data))
    # print(train_p_n_vocab)
    # 获得训练集上负样本
    n_train_data = train_data[train_data["label"] == 0]["sentence"]
    # 获取负样本的每个句⼦的形容词进行统计
    train_n_a_vocab = chain(*map(lambda x: get_a_list(x), n_train_data))
    # 调⽤绘制词云函数
    get_word_cloud(train_p_a_vocab) #训练集正样本
    get_word_cloud(train_n_a_vocab) #训练集负样本