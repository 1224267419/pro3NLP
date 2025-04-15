import fasttext
import re



#拆分数据集
def split_file(file_path, split_size,out_path):
    with open(file_path, 'r') as f:
        line_count = 0
        file_number = 1
        current_file = open(out_path+f'output_file_{file_number}.txt', 'w')
        for line in f:
            current_file.write(line)
            line_count += 1
            if line_count == split_size:
                current_file.close()
                file_number += 1
                current_file = open(out_path+f'output_file_{file_number}.txt', 'w')
                line_count = 0
    current_file.close()

#处理文件
def preprocess_text(input_file, output_file):
    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # 在标点符号前后添加空格
    text = re.sub(r'([.,!?\'/()])', r' \1 ', text)

    # 将所有字母转换为小写
    text = text.lower()

    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(text)

# split_file('./data/cooking/process_cooking.stackexchange.txt', 12000,'./data/cooking/')


#预处理原数据（分离特殊符号，转换为小写
# preprocess_text('./data/cooking/cooking.stackexchange.txt','./data/cooking/process_cooking.stackexchange.txt')
#
# preprocess_text('./data/cooking/output_file_1.txt','./data/cooking/process_output_file_1.txt')
# preprocess_text('./data/cooking/output_file_2.txt','./data/cooking/process_output_file_2.txt')
#
#调用函数进行拆分
# split_file('./data/cooking/cooking.stackexchange.txt', 12000,'./data/cooking/')
# split_file('./data/cooking/process_cooking.stackexchange.txt', 12000,'./data/cooking/process_')



path= "./data/cooking/"


#使用12000条数据进行训练

#模型训练

# model=fasttext.train_supervised(input=path+"output_file_1.txt")
# #预测一下一个句子的类型
# print(model.predict("Which baking dish is best to bake a banana bread ?"))
#
# #(3404, 0.12338425381903642, 0.053421521241414395)
# print("model1: 样本数  精度  召回率:")
# print(model.test(path+"output_file_2.txt"))
#
#
# model2=fasttext.train_supervised(input=path+"process_output_file_1.txt")
# #(3404, 0.15070505287896593, 0.06525057237344187)
# print("\n model2: 样本数  精度  召回率:")
# print(model2.test(path+"process_output_file_2.txt"))
# #处理源数据后召回率和精度都有提升
#
#
# # (3404, 0.5179200940070505, 0.22424319511574664)
# model3=fasttext.train_supervised(input=path+"process_output_file_1.txt",epoch=25)#默认epoch为5，增加epoch从而尝试收敛
# print("\n model3: 样本数  精度  召回率:")
# print(model3.test(path+"process_output_file_2.txt"))
# #增加epoch=25后精度高了近4倍,epoch=50时无变化
#
#
# # (3404, 0.5487661574618097, 0.23759857542610022)
# model4=fasttext.train_supervised(input=path+"process_output_file_1.txt",epoch=75,wordNgrams=2)#默认wordNgrams为1，增加wordNgrams从而尝试收敛,参数增加因此使用更多的epoch进行拟合
# print("\n model4: 样本数  精度  召回率:")
# print(model4.test(path+"process_output_file_2.txt"))
# #增加epoch=25后精度高了近4倍,epoch=50时无变化
#
#
# model5=fasttext.train_supervised(input=path+"process_output_file_1.txt",epoch=75,wordNgrams=2,loss="hs")#默认loss为ns，尝试使用hs(层次softmax)修改输出层，从而降低反向传播的复杂度
# #模型精度和recall有降低，训练速度加快许多
# # (3404, 0.5390716803760282, 0.2334011701857034)
# print("\n model5: 样本数  精度  召回率:")
# print(model5.test(path+"process_output_file_2.txt"))
# #增加epoch=25后精度高了近4倍,epoch=50时无变化

#自动超参数调节
model6=fasttext.train_supervised(input=path+"process_output_file_1.txt",autotuneValidationFile=path+"process_output_file_2.txt", autotuneDuration=600)#自动超参数调节,时间最大为600s，默认为300
# (3404, 0.5919506462984724, 0.25629610786059526)
print("\n model6: 样本数  精度  召回率:")
print(model6.test(path+"process_output_file_2.txt"))

# 对于上述代码，我们使用softmax实现标签输出，但实际上我们应该得到的是多个标签，
# 所以我们往往会选择为每个标签使⽤独⽴的⼆分类器作为输出层结构,
# 对应的损失计算⽅式为'ova'表示one vs all.
# 这种输出层的改变意味着我们在统⼀语料下同时训练多个⼆分类模型,
# 对于⼆分类模型来讲, lr不宜过⼤, 这⾥我们设置为0.2
model7=fasttext.train_supervised(input=path+"process_output_file_1.txt",epoch=25,wordNgrams=2,loss="ova",lr=0.2)
print("\n model7: 样本数  精度  召回率:")
print(model7.test(path+"process_output_file_2.txt"))
print(model7.predict("Which baking dish is best to bake a banana bread ?",k=-1,threshold=0.5)) #k为输出的可能标签数(>threshold的),-1时输出所有的可能标签

#模型保存
model6.save_model("./model/model6.bin")
model7.save_model("./model/model7.bin")

model=fasttext.load_model("./model/model7.bin") #加载模型