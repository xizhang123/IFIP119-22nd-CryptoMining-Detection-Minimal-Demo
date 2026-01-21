import torch
from torch import nn
class TimeEmbedding(nn.Module):
    def __init__(self, embedding_dim, period_start, period_end):
        assert embedding_dim % 2 == 0, "embedding_dim mod 2 must eq 0"
        assert embedding_dim >= 2,     "embedding_dim must le 2"
        assert period_end > period_start,  "period_end must lt period_start"
        super(TimeEmbedding, self).__init__()
        period_start, period_end = float(period_start), float(period_end)
        self.embedding_dim = embedding_dim
        #用于实现时间戳嵌入的频率向量
        gap = (period_end/period_start)**(1/(embedding_dim//2-1))
        period_tensor = torch.tensor([[(2*torch.pi)/(period_start*gap**d) for d in range(embedding_dim//2)]])
        self.register_buffer('period_tensor',period_tensor)
    def forward(self,x):
        return torch.concat([
            torch.sin(torch.matmul(x,self.period_tensor.to(x.device))),
            torch.cos(torch.matmul(x,self.period_tensor.to(x.device)))
        ],axis=-1)