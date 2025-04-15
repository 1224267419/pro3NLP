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
model=fasttext.train_supervised(input=path+"output_file_1.txt")
model2=fasttext.train_supervised(input=path+"process_output_file_1.txt")

#预测一下一个句子的类型
print(model.predict("Which baking dish is best to bake a banana bread ?"))

print("model1: 样本数  精度  召回率:")
print(model.test(path+"output_file_2.txt"))

print("\n model2: 样本数  精度  召回率:")
print(model2.test(path+"process_output_file_2.txt"))
#处理源数据后召回率和精度都有提升