import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

def without_head_model(model,):
    with torch.no_grad():
     encoded_layers, _ = model(tokens_tensor)
    print("不带头的模型输出结果:", encoded_layers)
    print("不带头的模型输出结果的尺⼨:", encoded_layers.shape)
if  __name__ == '__main__':
# # 1.加载预训练模型的映射器tokenizer
#     # 预训练模型来源
#     source = 'huggingface/pytorch-transformers'
#     # 选定加载模型的哪⼀部分, 这⾥是模型的映射器
#     part = 'tokenizer'
#     # 加载的预训练模型的名字
#     model_name = 'bert-base-chinese'
#     tokenizer = torch.hub.load(source, part, model_name)
# # 2. 加载带/不带头的预训练模型
#     part = 'model'
#     model = torch.hub.load(source, part, model_name)
#     # 加载带有语⾔模型头的预训练模型
#     part = 'modelWithLMHead'
#     lm_model = torch.hub.load(source, part, model_name)
#     # 加载带有类模型头的预训练模型
#     part = 'modelForSequenceClassification'
#     classification_model = torch.hub.load(source, part, model_name)
#     # 加载带有问答模型头的预训练模型
#     part = 'modelForQuestionAnswering'
#     qa_model = torch.hub.load(source, part, model_name)
# Load model directly
    from transformers import AutoTokenizer, AutoModelForMaskedLM

    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    model = AutoModelForMaskedLM.from_pretrained("bert-base-chinese")
    # 使⽤tokenizer进⾏数值映射
    input_text = "⼈⽣该如何起头"
    indexed_tokens = tokenizer.encode(input_text)
    # 打印映射后的结构
    print("indexed_tokens:", indexed_tokens)
    # 将映射结构转化为张量输送给不带头的预训练模型
    tokens_tensor = torch.tensor([indexed_tokens])

# 3.模型输出结果

    # # 使⽤不带头的预训练模型获得结果
    # with torch.no_grad():
    #     encoded_layers= model(tokens_tensor)
    # print("不带头的模型输出结果:", encoded_layers)
    # print("不带头的模型输出结果的尺⼨:", encoded_layers.shape)
    #
    # 带有语⾔模型头的预训练模型获得结果
    with torch.no_grad():
        lm_output = model(tokens_tensor)
    print("带语⾔模型头的模型输出结果:", lm_output)
    print("带语⾔模型头的模型输出结果的尺⼨:", lm_output[0].shape)