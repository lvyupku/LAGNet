import dgl
import numpy as np
import torch.nn as nn
import torch
from dgl.nn import AGNNConv


class AttentionGCN(nn.Module):
    def __init__(self, num_classes, feat, adj_file, layers=2):
        super(AttentionGCN, self).__init__()
        self.gcn = []
        
        self.gcn = nn.ModuleList([AGNNConv() for _ in range(layers)])
        
        self.g = self.gen_G(num_classes, adj_file)
        self.feat = feat
    
    def forward(self, x):        
        y = self.feat
        for gcn in self.gcn:
            y = gcn(self.g, y)
            
        y = y.transpose(0, 1)
        x = torch.matmul(x, y)
            
        return x
            
    def gen_G(self, num_classes, adj_file):
        #行
        row = []
        #列
        col = []
        
        weight = []
        A = self.gen_A(num_classes, adj_file)
        for i in range(num_classes):
            for j in range(num_classes):
                row.append(i)
                col.append(j)
                weight.append(A[i][j])
        edges = torch.tensor(row), torch.tensor(col)
        g = dgl.graph(edges).to(0)
        
        g.edata['w'] = torch.tensor(weight).cuda(non_blocking=True)
        
        return g

        
    def gen_A(self, num_classes, adj_file):
        import pickle
        result = pickle.load(open(adj_file, 'rb'))
        _adj = result['adj']
        _nums = result['nums']
        _nums = _nums[:, np.newaxis]
        _adj = _adj / _nums
        mmin = 1
        mmax = -1

        for i in range(num_classes):
            for j in range(num_classes):
                if i == j:
                    continue
                mmin = min(_adj[i][j], mmin)
                mmax = max(_adj[i][j], mmax)

        _adj = (_adj - mmin)/(mmax-mmin)
        return _adj
            
    