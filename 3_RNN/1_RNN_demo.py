import torch
import torch.nn as nn

def rnn_demo():
    # RNN(input_size, hidden_size, num_layers, nonlinerity(default: 'tanh'), bias(default: True), batch_first(default: False), dropout(default: 0), bidirectional(default: False))
    rnn = nn.RNN(5, 6, 1)
    # input: (seq_len, batch_size, input_size)
    input = torch.randn(4, 3, 5)
    # hidden: (num_layers * num_directions, batch_size, hidden_size),
    h0 = torch.randn(1, 3, 6)
    # 输出两个值,第一个是输出,第二个作为下一层的输入
    output, hn = rnn(input, h0)
    # output: (seq_len, batch_size, hidden_size),output:[2, 3, 6]
    print('output.shape',output.shape)
    # hn: (num_layers * num_directions, batch_size, hidden_size) [1, 3, 6]
    print('hn.shape',hn.shape)
def lstm_demo():
    # LSTM(input_size, hidden_size, num_layers, nonlinerity(default: 'tanh'), bias(default: True), batch_first(default: False), dropout(default: 0), bidirectional(default: False))
    lstm = nn.LSTM(5, 6, 2)
    # input: (seq_len, batch_size, input_size)
    input = torch.randn(1, 3, 5)
    # hidden: (num_layers * num_directions, batch_size, hidden_size),
    h0 = torch.randn(2, 3, 6)   #初始化隐藏层
    c0 = torch.randn(2, 3, 6)   #初始化细胞状态
    # 输出两个值,第一个是输出,第二个作为下一层的输入
    output, (hn, cn) = lstm(input, (h0, c0))
    print('output.shape',output.shape)
    print('hn.shape',hn.shape)
    print('cn.shape',cn.shape)
def gru_demo():
    gru = nn.GRU(5, 6, 2)
    input = torch.randn(1, 3, 5)
    h0 = torch.randn(2, 3, 6)
    output, hn = gru(input, h0)
    print('output.shape',output.shape)
    print('hn.shape',hn.shape)



if __name__ == '__main__':
    # rnn_demo()
    # lstm_demo()
    gru_demo()
