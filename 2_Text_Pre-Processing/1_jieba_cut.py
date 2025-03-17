import jieba
from jieba import cut_for_search
import jieba.posseg as pseg
# 按装订区域中的绿色按钮以运行脚本。
if __name__ == '__main__':
    content = "工信处女干事每月经过下属科室都要亲口交代24口交换机等技术性器件的安装工作"
    # cut_a11默认为False
    print(jieba.cut(content, cut_all=True))
    #<generator object Tokenizer.cut at 0x000001FC55975190> ,返回了一个生成器对象
    #精确模式(default)
    print(jieba.lcut(content, cut_all=False))
    #全模式,把所有可能的组词都分割出来,但可能不消除歧义
    print(jieba.lcut(content, cut_all=True))
    #搜索引擎模式,在精确模式的基础上,对长词再次切分,提高召回率,适合用于搜索引擎分词
    print(cut_for_search(content))
    print(pseg.lcut(content))#输出带词性的分词元组