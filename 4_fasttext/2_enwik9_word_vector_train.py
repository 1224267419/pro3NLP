import fasttext
if __name__=='__main__':
    data_file='../2_Text_Pre-Processing/data/fil9'

    # 在训练词向量过程中, 我们可以设定很多常⽤超参数来调节我们的模型效果, 如:
    # ⽆监督训练模式: 'skipgram' 或者 'cbow', 默认为'skipgram', 在实践中，skipgram模式在利⽤⼦词⽅⾯⽐cbow更好.
    # 词嵌⼊维度dim: 默认为100, 但随着语料库的增⼤, 词嵌⼊的维度往往也要更⼤.
    # 数据循环次数epoch: 默认为5, 但当你的数据集⾜够⼤, 可能不需要那么多次.
    # 学习率lr: 默认为0.05, 根据经验, 建议选择[0.01，1]范围内.
    # 使⽤的线程数thread: 默认为12个线程, ⼀般建议和你的cpu核数相同.

    # #因为没有词义信息,这里用无监督学习
    model = fasttext.train_unsupervised(data_file, "cbow", dim=300,
                                        epoch=1, lr=0.1, thread=8)
    model.save_model("./model/vect_model.bin")#保存模型


    model=fasttext.load_model("./model/vect_model.bin")#加载模型

    # print(model.get_word_vector('the'))#获取单词向量
    print(model.get_nearest_neighbors('sport'))#获取最相似的单词
    print(model.get_nearest_neighbors('music'))

