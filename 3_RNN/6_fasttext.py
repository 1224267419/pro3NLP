import fasttext

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

# 调用函数进行拆分
# split_file('./data/cooking/cooking.stackexchange.txt', 12000,'./data/cooking/')
path="./data/cooking/"

#使用12000条数据进行训练
model=fasttext.train_supervised(input=path+"output_file_1.txt")

#预测一下一个句子的类型
print(model.predict("Which baking dish is best to bake a banana bread ?"))

print("样本数  精度  召回率:\n")
print(model.test(path+"output_file_2.txt"))