import torch


def demo0(): #新建tensor操作
    x = torch.tensor([1, 2, 3])  # array转tensor
    print(x)
    x = torch.empty(5, 3)  # 创建一个未初始化的5*3的矩阵
    print(x)
    x = torch.rand(5, 3)  # 创建一个5*3的随机矩阵,值服从[0,1)均匀分布
    print(x)
    x = torch.randn(5, 3)  # 创建一个5*3的随机矩阵,值服从标准正态分布
    print(x)
    print(torch.zeros(5, 3, dtype=torch.float))  # 创建一个5*3的矩阵,值全为0,float浮点类型
    x = torch.ones(5, 3, dtype=torch.float)
    print(torch.randn_like(x))  # 创建一个与x形状相同的随机矩阵
    print(x.size())  # 查看x的形状
    print(x.shape)  # 查看x的形状
def demo1(): #加减乘除操作
    x = torch.rand(5, 3)
    y = torch.rand(5, 3)
    print(torch.add(x, y))  # 加法
    print(x + y) #和上面结果一致
    #上述加法不改变原来的x和y
    result=torch.empty_like(x)#必须用相同大小的tensor承载结果
    torch.add(x, y, out=result)# 注意,如果out=result,则结果写入result,但形状必须一致
    print(result)
    print(y.add_(x)) # y+=x,注意add_是in-place操作

def demo2(): #view和tolist,item
    x=torch.rand(5,3)
    print(x.view(1,-1).shape)
    print(x.view(-1,1).shape)
    print(x.tolist())
    print(x[0,0].tolist())
    print(x[0,0].item())
def demo3():
    x=torch.tensor([1,2,3])
    y=x.numpy #转换为nparray
    y=y+1
    print(torch.from_numpy(y)) #从nparray转换回来,
    print(x)
def demo4(): #数据转移到GPU上的标准流程
    x=torch.ones(5,3)
    if torch.cuda.is_available():
        # 定义一个设备对象, 这里指定成CUDA, 即使用GPU
        device = torch.device('cuda')
        # 直接在GPU上创建一个Tensor
        y = torch.ones_like(x, device=device)
        print("y在GPU上")
        # 将在CPU上面的x张量移动到GPU上面
        x = x.to(device)
        print("x在GPU上")
        # x和y都在GPU上面, 才能支持加法运算
        z = x + y
        # 此处的张量z在GPU上面
        print(z)
        # 也可以将z转移到CPU上面, 并同时指定张量元素的数据类型
        print(z.to('cpu', torch.double))
    print('执行完成====')

def demo5():
    a=1
def demo6():
    a=1


if __name__ == '__main__':
    # demo0()
    # demo1()
    # demo2()
    # demo3()
    demo4()