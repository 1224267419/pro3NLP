import numpy as np
import os
import time
import tensorflow as tf
#下载数据
# 参数一： 下载的文件名
# 参数二： 下载的地址
# path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

# 读取数据
text = open("./data/shakespeare.txt", 'rb').read().decode(encoding='utf-8')

# 统计字符个数并查看前250个字符
print('Length of text: {} characters'.format(len(text))) # 1115394 characters
# print(text[:250])
# 统计文本中非重复字符数量
vocab = sorted(set(text))
# print('vocab====:', vocab)
# print('vocab====:', type(vocab))
print ('{} unique characters'.format(len(vocab))) #65 unique characters

# 对文本进行数值映射
char2idx = {u:i for i, u in enumerate(vocab)}
# print('char2idx====:', char2idx)
'''
char2idx====: {'\n': 0, ' ': 1, '!': 2, '$': 3, '&': 4, "'": 5, ',': 6, '-': 7, '.': 8, '3': 9, ':': 10, ';': 11, '?': 12, 'A': 13, 'B': 14, 'C': 15, 'D': 16, 'E': 17, 'F': 18, 'G': 19, 'H': 20, 'I': 21, 'J': 22, 'K': 23, 'L': 24, 'M': 25, 'N': 26, 'O': 27, 'P': 28, 'Q': 29, 'R': 30, 'S': 31, 'T': 32, 'U': 33, 'V': 34, 'W': 35, 'X': 36, 'Y': 37, 'Z': 38, 'a': 39, 'b': 40, 'c': 41, 'd': 42, 'e': 43, 'f': 44, 'g': 45, 'h': 46, 'i': 47, 'j': 48, 'k': 49, 'l': 50, 'm': 51, 'n': 52, 'o': 53, 'p': 54, 'q': 55, 'r': 56, 's': 57, 't': 58, 'u': 59, 'v': 60, 'w': 61, 'x': 62, 'y': 63, 'z': 64}
'''
idx2char = np.array(vocab)
# print('idx2char===:', idx2char)
# print('idx2chartype===:', type(idx2char))
# 使用字符到数字的映射表示所有样本
text_as_int = np.array([char2idx[c] for c in text])
# print('text_as_int===:', text_as_int)  # text_as_int===: [18 47 56 ... 45  8  0]

# zip的用法：https://www.runoob.com/python/python-func-zip.html
for char, _ in zip(char2idx, range(20)):
    # print('char===', char)
    # print('*****', _)
    # 扩展==repr:https://www.runoob.com/python/python-func-repr.html
    # 不加repr可以打印， 只是空格， 换行符等显示不出来
    print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
    # print('  {:4s}: {:3d},'.format(char, char2idx[char]))

# 查看原始语料前13个字符映射后的结果
print ('{} ---- characters mapped to int ---- > {}'.format(repr(text[:13]), text_as_int[:13]))

# 生成训练数据
# 设定输入序列长度
seq_length = 100
# 获取样本总数
examples_per_epoch = len(text) // seq_length
#将数值映射后的文本转化为dataset对象， 方便后续处理
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
# print('char_dataset===:', char_dataset) # <TensorSliceDataset shapes: (), types: tf.int64>

# take：表示取出数据的前5个样本数
# for i in char_dataset.take(5):
#     print(idx2char[i.numpy()])
#     print('vocab===', vocab[i]) # 这个和上面是一样的效果

# 使用dataset的batch方法按照字符长度+1划分（要留出一个向后顺移的位置）
# drop_remainder=True表示删除掉最后一批可能小于批次数量的数据
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)
# print("sequences===:", sequences) #<BatchDataset shapes: (101,), types: tf.int64>

# 查看划分后的5条数据对应的文本内容
# for item in sequences.take(5):
#     print(repr(''.join(idx2char[item.numpy()])))

def split_input_target(chunk):
    """划分输入序列和目标序列函数"""
    # 前100个字符为输入序列，第二个字符开始到最后为目标序列
    input_text = chunk[:-1]
    targt_text = chunk[1:]
    return input_text, targt_text

# 使用map方法调用该函数对每条序列进行划分
dataset = sequences.map(split_input_target)

for input_example, target_example in dataset.take(5):
    print ('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
    print ('Target data:', repr(''.join(idx2char[target_example.numpy()])))

    # 查看将要输入模型中的每个时间步的输入和输出(以前五步为例)
    # 循环每个字符，并打印每个时间步对应的输入和输出
    for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
        print("Step {:4d}".format(i))
        print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
        print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))

# 创建批次数据
# 定义批次大小为64
BATCH_SIZE = 64
# 设置缓存区大小
BUFFER_SIZE = 10000
# 打乱数据并分批次
# 理解： dataset.shuffle就是说维持一个buffer size 大小的 shuffle buffer，图中所需的每个样本从shuffle buffer中获取，
# 取得一个样本后，就从源数据集中加入一个样本到shuffle buffer中   drop_remainder删除最后不足BATCH_SIZE的数据
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE,  drop_remainder=True)

print('datasetbuffer===',dataset) #datasetbuffer=== <BatchDataset shapes: ((64, 100), (64, 100)), types: (tf.int64, tf.int64)>


# 第二步: 构建模型并训练模型
# 获取词汇集的大小  vocab=65
vocab_size=len(vocab)
# 定义词嵌入的维度(超参数)
embedding_dim = 256

# 定义GRU的隐层节点数量
rnn_units = 1024

# 模型包括三个层， 输入层embedding层，中间层GRU层，输出层
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    '''
    :param vocab_size: 获取词汇集的大小
    :param embedding_dim: 定义词嵌入的维度
    :param rnn_units: 定义GRU的隐层节点数量
    :param batch_size: 批次样本数量
    :return:
    '''
    # 使用tf.keras.Sequential定义模型
    # GRU层的参数return_sequences为True说明返回结果为每个时间步的输出，而不是最后时间步的输出
    # stateful参数为True，说明将保留每个batch数据的结果状态作为下一个batch的初始化数据
    # recurrent_initializer='glorot_uniform'，说明GRU的循环核采用均匀分布的初始化方法
    # 模型最终通过全连接层返回一个所有可能字符的概率分布.
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

model = build_model(vocab_size=len(vocab), embedding_dim=embedding_dim,rnn_units=rnn_units, batch_size=BATCH_SIZE)

# 试用模型
for input_example_batch, target_example_batch in dataset.take(1):
    # print('input_example_batch===:',input_example_batch)
    # print('input_example_batch_shape===:',input_example_batch.shape)  #(64, 100)
    example_batch_predictions = model(input_example_batch)
    # print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)") # (64, 100, 65) # (batch_size, sequence_length, vocab_size)

    # print(model.summary())

    # 第一个参数：表示在这个里面构建这么大的形状矩阵
    # 第二个参数： 表示在这个矩阵中取值的个数， 如果是num_samples=1, 表示按照这样的概率分布每行取一个， 如果是2表示每行取2个
    sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
    # sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=2)
    # print('example_batch_predictions===:', example_batch_predictions[0].shape)  #shape=(100, 65), dtype=float32
    # print('sampled_indices===:', sampled_indices) #shape=(100, 1)
    # print('sampled_indices===:', sampled_indices.shape) #shape=(100, 1)
    # 按照最后一维度进行压缩 维度不为1的不能压缩
    sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()
    # print('sampled_indicessqueeze===:', sampled_indices)

    # 也将输入映射成文本内容  将中间的100个数字反转为100个字符
    print("Input: \n", repr("".join(idx2char[input_example_batch[0]])))
    print()

    # 映射这些索引查看对应的文本
    # 在没有训练之前，生成的文本没有任何规律
    print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices ])))

# 定义损失函数 labels真实标签  logits预测标签 由于模型返回logits，因此需要设置from_logits标志
# def loss(labels, logits):
#     return tf.keras.losses.sparse_categorical_crossentropy(labels,  logits, from_logits=True)
#
# # 使用损失函数 此处求出来的是64个样本的总体损失
# example_batch_loss = loss(target_example_batch, example_batch_predictions)
#
# # print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
# # print("scalar_loss:      ", example_batch_loss.numpy().mean())
#
# # 添加优化器
# model.compile(optimizer='adam', loss=loss)
#
# 配置检测点
checkpoint_dir = './training_checkpoints'
# 方法二的检查点
# checkpoint_dir = './new_training_checkpoints'
# 检测点的文件 名
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt_{epoch}')

# 创建检测点的保存回调对象
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True)
# '''
# 扩展： 保存之后的各作用文件
# meta file保存了graph结构,包括 GraphDef, SaverDef等,当存在meta file,我们可以不在文件中定义模型,也可以运行,
# 而如果没有meta file,我们需要定义好模型,再加载data file,得到变量值.
#
# index file为一个 string-string table,table的key值为tensor名,value为BundleEntryProto, BundleEntryProto.
#
# data file保存了模型的所有变量的值
# '''
#
#
# # 模型训练并打印日志
# EPOCHS = 2
# history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
#
#
# ===== 方式二 =====
# 构建训练模型
# model = build_model(vocab_size=len(vocab), embedding_dim=embedding_dim, rnn_units=rnn_units, batch_size=BATCH_SIZE)

# 选择优化器
# optimizer = tf.keras.optimizers.Adam()

# # TensorFlow 2 为我们提供了 tf.function 模块，结合 AutoGraph 机制，
# # 使得我们仅需加入一个简单的 @tf.function 修饰符，就能轻松将模型以图执行模式运行
# @tf.function
# def train_step(inp, target):
#     '''
#     :param inp: 输入模型
#     :param target: 输入对应的标签
#     :return:
#     '''
#     # 打开梯度记录管理器
#     with tf.GradientTape() as tape:
#         # 使用模型进行预测
#         predictions = model(inp)
#
#         # 使用sparse_categorical_crossentropy计算平均损失
#         # from_logits	是否预期为logits张量。默认情况下为False，我们假设对概率分布进行编码
#           https://blog.csdn.net/yxpandjay/article/details/109090533
#         loss = tf.reduce_mean(
#             tf.keras.losses.sparse_categorical_crossentropy(target, predictions, from_logits=True)
#         )

#     # 使用梯度记录管理器求解全部的参数梯度
#     grads = tape.gradient(loss, model.trainable_variables)
#
#     # 使用梯度和优化器更新参数
#     optimizer.apply_gradients(zip(grads, model.trainable_variables))
#     # 返回平均损失
#     return loss
#
# # 进行训练
# # 训练轮数
# EPOCHS = 5
#
# # 进行轮数的训练
# for epoch in range(EPOCHS):
#     # 获取开始时间
#     start = time.time()
#     # 初始化隐层状态
#     hidden = model.reset_states()
#
#     # 进行批次循环
#     for (batch_n, (inp, target)) in enumerate(dataset):
#         # 调用train_step进行训练, 获得批次循环的损失
#         loss = train_step(inp, target)
#
#         # 每100个批次打印轮数，批次和对应的损失
#         if batch_n % 100 == 0:
#             template = 'Epoch {} Batch {} Loss {}'
#             print(template.format(epoch + 1, batch_n, loss))
#     # 每5轮保存一次检测点
#     if (epoch + 1) % 1 == 0:
#         model.save_weights(checkpoint_prefix.format(epoch=epoch))
#
#         # 打印轮数，当前损失，和训练耗时
#         print('Epoch {} Loss {:.4f}'.format(epoch + 1, loss))
#         print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
#
#     model.save_weights(checkpoint_prefix.format(epoch=epoch))
#
#
#
# 第三步: 使用模型生成文本内容

# 恢复模型
model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

# 构建生成函数
def generate_text(model, start_string):
    '''
    :param model: 训练后的模型
    :param start_string: 任意起始的字符串
    :return:
    '''
    # 要生成的字符个数
    num_generate = 1000

    # 将起始的字符串转化为数字
    input_eval = [char2idx[s] for s in start_string]

    # 扩展维度满足模型的输入要求
    input_eval = tf.expand_dims(input_eval, 0)
    print('input_eval_shape===', input_eval.shape)

    # 空列表用于存储结果
    text_generated = []

    # 设定“温度参数”，根据tf.random_categorical方法特点，
    # 温度参数能够调节该方法的输入分布中概率的差距，以便控制随机被选中的概率大小
    temperature = 1.0

    # 初始化模型参数
    model.reset_states()

    # 开始循环生成
    for i in range(num_generate):
        #使用模型获得输出
        predictions = model(input_eval)
        print('predictionshape===', predictions.shape)
        '''
        predictionshape=== (1, 6, 65)
        predictionshape=== (1, 1, 65)
        predictionshape=== (1, 1, 65)
        predictionshape=== (1, 1, 65)
        '''

        #压缩批次维度
        predictions = tf.squeeze(predictions, 0)

        # 使用“温度参数”和tf.random.categorical方法生成最终的预测字符索引
        predictions = predictions/temperature
        # [-1, 0]表示最后一行的第0维度 因为是第一次输入， 所以要取最后一个字符， 作为下一次的输入，
        # 这样后面就可以进行逐字符输入然后进行逐字符预测了
        prediction_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        # print('tf.random.categorical(predictions, num_samples=1)===', tf.random.categorical(predictions, num_samples=1))
        # print('tf.random.categorical===', tf.random.categorical(predictions, num_samples=1)[-1, 0])
        # 将预测的输出再扩展维度作为下一次的模型输入
        input_eval = tf.expand_dims([prediction_id], 0)

        # 将该次输出映射成字符存到列表中
        text_generated.append(idx2char[prediction_id])


    # 最后将初始字符串和生成的字符串进行拼接
    return (start_string+''.join(text_generated))

# 调用
print(generate_text(model, start_string='ROMEO:'))
















