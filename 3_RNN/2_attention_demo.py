import torch
import torch.nn as nn
import torch.nn.functional as F
def bmm_demo(): #bmm乘法
    # 如果参数1形状是(b × n × m), 参数2形状是(b × m × p), 则输出为(b × n × p)
    input = torch.randn(10, 3, 4)
    mat2 = torch.randn(10, 4, 5)
    res = torch.bmm(input, mat2)
    print(res.size())   # torch.Size([10, 3, 5])


class Attn(nn.Module):
    def __init__(self, query_size,key_size, value_size1, value_size2,output_size):
        """
        :param query_size: 代表query的最后一维大小
        :param key_size: 表key的最后一维大小
        :param value_size1: 代表value第二维大小
        :param value_size2: value最后一维大小
        :param output_size: 输出的最后一维大小
        :return:
        """
        super(Attn, self).__init__()
        #传参
        self.query_size = query_size
        self.key_size = key_size
        self.value_size1=value_size1
        self.value_size2= value_size2
        self.output_size= output_size

        #注意力机制实现第一步所需要的线性层
        self.attn = nn.Linear(self.query_size + self.key_size, value_size1)
        #注意力机制实现第三步所需要的线性层
        self.attn_combine = nn.Linear(self.query_size + self.value_size2, self.output_size)
    def forward(self,Q,K,V):
        """
        Q,K,V都是三维tensor
        :param Q:
        :param K:
        :param V:
        :return:
        """
        #注意力机制实现第一步
        #Q,K按纵轴拼接,做一次线性变换,最后用softmax处理得到结果
        attn_weights = F.relu(self.attn(torch.cat((Q[0],K[0]), 1)))

        #第一步后半部分,将weight和V做矩阵乘法
        #如果二者都是三维张量且[0]维为betch时,做bmm运算
        #unsqueeze用于扩充0维度,使其变为3维( (1,32)->(1,1,32) )
        #bmm (1,1,32)*(1,32,64) ->  (1,1,64)
        attn_applied=torch.bmm(attn_weights.unsqueeze(0),V)


        #第二步,取[0]进行降维,按第一步计算
        #因此要先将Q与第一步计算结果再进行拼接
        output=torch.cat((Q[0],attn_applied[0]),1)

        #第三步,对上面的输出做一次线性变换(64+32 -> 64),再扩展为3维
        output = self.attn_combine(output).unsqueeze(0)
        return output,attn_weights

if __name__== '__main__':
    # bmm_demo()


    query_size = 32
    key_size = 32
    value_size1 = 32
    value_size2 = 64
    output_size = 64

    attn = Attn(query_size, key_size, value_size1, value_size2, output_size)
    Q = torch.randn(1, 1, 32)
    K = torch.randn(1, 1, 32)
    #(32,2)展平变64

    V = torch.randn(1, 32, 64)
    out = attn(Q, K, V)
    print(out[0].shape)
    print(out[1].shape)