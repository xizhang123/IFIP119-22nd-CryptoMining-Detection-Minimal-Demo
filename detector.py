import torch
from torch import nn

class DNN_Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, hidden_n, output_dim):
        super(DNN_Classifier, self).__init__()
        # 输入
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        # 隐藏层
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(hidden_n)])
        # 输出层
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        # 激活函数
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 输入层
        x = self.relu(self.fc_in(x))
        # 隐藏层
        for layer in self.hidden_layers:
            x = self.relu(layer(x))
        # 输出
        x = self.fc_out(x)
        # 压缩的sigmoid自动忽略异常值
        return self.sigmoid(x)#*0.9+0.05


class LSTM_Classifier(nn.Module):
    def __init__(self, te, projector, lstm_ae, dnn_classifier):
        super(LSTM_Classifier, self).__init__()
        self.te = te
        self.projector = projector
        self.lstm_ae = lstm_ae
        self.dnn_classifier = dnn_classifier
        self.use_meta = 1024.0
        self.use_ts = 1.0

    def forward(self,ts,meta):
        payload_flag_vector = self.projector(meta)
        timestamp_vector = self.te(ts)
        meta_vector = torch.cat([timestamp_vector*self.use_ts,payload_flag_vector*self.use_meta],dim=-1)
        reconstitution, feature = self.lstm_ae(meta_vector)
        return meta_vector.data, reconstitution, self.dnn_classifier(feature)