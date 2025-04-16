import fasttext
if __name__=='__main__':
    model=fasttext.load_model('./model/cc.zh.300.bin')
    print(model.get_nearest_neighbors('中国'))