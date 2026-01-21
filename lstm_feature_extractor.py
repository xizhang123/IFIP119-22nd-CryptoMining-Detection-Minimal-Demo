import torch
from torch import nn
from torch.nn import Module
#长短期记忆单元
class LSTM_Unit(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(LSTM_Unit, self).__init__()
        #输入门参数
        self.W_i = nn.Parameter(torch.randn(embedding_dim, hidden_dim))
        self.U_i = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.b_i = nn.Parameter(torch.randn(hidden_dim))
        #遗忘门参数
        self.W_f = nn.Parameter(torch.randn(embedding_dim, hidden_dim))
        self.U_f = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.b_f = nn.Parameter(torch.randn(hidden_dim))
        #加工输入的参数
        self.W_c = nn.Parameter(torch.randn(embedding_dim, hidden_dim))
        self.U_c = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.b_c = nn.Parameter(torch.randn(hidden_dim))
        #输出门参数
        self.W_o = nn.Parameter(torch.randn(embedding_dim, hidden_dim))
        self.U_o = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.b_o = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, x, c, h):
        #x:[batch,emb_dim]
        #计算遗忘门、输入门、输出门的值
        f = torch.sigmoid(x @ self.W_f + h @ self.U_f + self.b_f)
        i = torch.sigmoid(x @ self.W_i + h @ self.U_i + self.b_i)
        o = torch.sigmoid(x @ self.W_o + h @ self.U_o + self.b_o)
        #对输入的变换
        g = torch.tanh(x @ self.W_c + h @ self.U_c + self.b_c)
        #使用输入更新记忆
        c = c * f + g * i
        #新的记忆经过输出门产生隐藏状态
        h = o * torch.tanh(c)
        #返回记忆单元，隐藏状态
        return c, h

# LSTM编码器
class LSTM_Encoder(nn.Module):
    def __init__(self, embedding_dim, encode_dim, deep):
        super(LSTM_Encoder, self).__init__()
        self.deep = deep
        self.encode_dim = encode_dim
        self.lstm = nn.ModuleList([LSTM_Unit(embedding_dim, embedding_dim) for _ in range(deep)])
        self.linear = nn.Linear(embedding_dim*2*deep,encode_dim)
        
    def forward(self, x):
        # x:[batch,len,embed_dim]
        # 初始化隐藏状态与记忆单元
        h = [x.data.new(x.size(0),x.size(2)).fill_(0).float() for _ in range(self.deep)]
        c = [x.data.new(x.size(0),x.size(2)).fill_(0).float() for _ in range(self.deep)]
        outputs = x.data.new(x.size(0),x.size(2)*2*self.deep).fill_(0).float()
        # 遍历每一个时间步
        for t in range(x.size(1)):
            x_i = x[:,t,:] # 取出当前时间步
            # 根据(当前时间步,记忆细胞,隐藏状态)生成输出和下一步的记忆细胞、隐藏状态
            c[0],h[0] = self.lstm[0](x_i,c[0],h[0])
            if t == x.size(1)-1:
                # 将记忆细胞添加到输出编码中
                outputs[:,:x.size(2)] = c[0]
                # 将隐藏状态添加到输出编码中
                outputs[:,x.size(2):x.size(2)*2] = h[0]
            for i in range(1,self.deep):
                # 将上一层的产出送入下一层
                c[i],h[i] = self.lstm[i](h[i-1],c[i],h[i])
                if t == x.size(1)-1:
                    # 将记忆细胞添加到输出编码中
                    outputs[:,x.size(2)*i*2:x.size(2)*i*2+x.size(2)] = c[i]
                    # 将隐藏状态添加到输出编码中
                    outputs[:,x.size(2)*i*2+x.size(2):x.size(2)*i*2+x.size(2)*2] = h[i]
        return self.linear(outputs)

# LSTM解码器
class LSTM_Decoder(nn.Module):
    def __init__(self, embedding_dim, encode_dim, deep):
        super(LSTM_Decoder, self).__init__()
        self.deep = deep
        self.embedding_dim = embedding_dim
        self.linear = nn.Linear(encode_dim,embedding_dim)
        self.lstm = nn.ModuleList([LSTM_Unit(embedding_dim, embedding_dim) for _ in range(deep)])
        
    def forward(self, x, step):
        # feature:[batch,encode_dim] -> [batch,embedding_dim]
        x = self.linear(x)
        # 初始化隐藏状态与记忆单元
        h = [x.data.new(x.size(0),self.embedding_dim).fill_(0).float() for _ in range(self.deep)]
        c = [x.data.new(x.size(0),self.embedding_dim).fill_(0).float() for _ in range(self.deep)]
        inputs = x.data.new(x.size(0),step,self.embedding_dim).fill_(0).float()
        # 遍历每一个时间步
        # 编码从前往后解码从后向前
        for t in range(step):
            # 根据(当前时间步,记忆细胞,隐藏状态)生成输出和下一步的记忆细胞、隐藏状态
            c[0],h[0] = self.lstm[0](x,c[0],h[0])
            for i in range(1,self.deep):
                # 将上一层的产出送入下一层
                c[i],h[i] = self.lstm[i](h[i-1],c[i],h[i])
            inputs[:,step-1-t,:] = h[i]
        return inputs

# LSTM自编码器
class LSTM_AE(nn.Module):
    def __init__(self, embedding_dim, encode_dim, deep, step):
        super(LSTM_AE, self).__init__()
        self.step = step
        self.encoder = LSTM_Encoder(embedding_dim,encode_dim,deep)
        self.decoder = LSTM_Decoder(embedding_dim,encode_dim,deep)
        
    def forward(self,x,rec=True):
        feature = self.encoder(x)
        if rec:
            return self.decoder(feature,self.step), feature
        else:
            return feature
        