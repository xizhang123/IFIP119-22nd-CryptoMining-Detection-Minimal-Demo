import torch.nn as nn
import torch.nn.functional as F

class Projector(nn.Module):
    def __init__(self,input_dim,hidden_dim,layers_n,output_dim):
        super(Projector, self).__init__()
        self.input = nn.Linear(input_dim,hidden_dim)
        self.hidden = nn.ModuleList([nn.Linear(hidden_dim,hidden_dim) for _ in range(layers_n)])
        self.output = nn.Linear(hidden_dim,output_dim)
    def forward(self, x):
        x = F.relu(self.input(x))
        for hidden in self.hidden:
            x = F.relu(hidden(x))
        return self.output(x)
