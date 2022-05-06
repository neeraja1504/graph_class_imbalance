import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer, SpGraphAttentionLayer


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        print("Inside GAT model")    

        # self.out_att = GraphAttentionLayer(nhid * nheads, nhid, dropout=dropout, alpha=alpha, concat=False)
        self.out_proj = nn.Linear(nhid * nheads, nhid)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.out_proj.weight,std=0.05)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_proj(x))
        return x



class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat, 
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads, 
                                             nhid, 
                                             dropout=dropout, 
                                             alpha=alpha, 
                                             concat=False)
        self.out_proj = nn.Linear(nhid * nheads, nhid) 
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.out_proj.weight,std=0.05)                                    


    def forward(self, x, adj):
        print("In forward of model")
        print(x.shape)
        print(adj.shape)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        print(x.shape)
        x = F.dropout(x, self.dropout, training=self.training)
        # x = F.elu(self.out_proj(x))
        x = F.elu(self.out_att(x, adj))
        print(x.shape)
        # return F.log_softmax(x, dim=1)
        return x
        
class GAT_Classifier(nn.Module):
    def __init__(self, nembed, nhid, nclass, dropout, alpha, nheads):
        super(GAT_Classifier, self).__init__()

        
        self.attentions = [GraphAttentionLayer(nembed, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        print("Inside graph classifier")    

        self.out_proj = nn.Linear(nhid * nheads, nhid)

        self.dropout = dropout
        self.mlp = nn.Linear(nhid, nclass)
        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.mlp.weight,std=0.05)
        nn.init.normal_(self.out_proj.weight,std=0.05)

    def forward(self, x, adj):
        print(x.shape)
        print(adj.shape)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_proj(x))
        x = self.mlp(x)

        return x  

class SpGAT_Classifier(nn.Module):
    def __init__(self, nembed, nhid, nclass, dropout, alpha, nheads):
        super(SpGAT_Classifier, self).__init__()

        
        self.attentions = [SpGraphAttentionLayer(nembed, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        print("Success!")
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        print("Inside graph classifier") 

        self.out_att = SpGraphAttentionLayer(nhid * nheads, 
                                             nhid, 
                                             dropout=dropout, 
                                             alpha=alpha, 
                                             concat=False)   

        self.out_proj = nn.Linear(nhid * nheads, nhid)

        self.dropout = dropout
        self.mlp = nn.Linear(nhid, nclass)
        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.mlp.weight,std=0.05)
        nn.init.normal_(self.out_proj.weight,std=0.05)

    def forward(self, x, adj):
        print("In forward of classifier")
        print("x is",x.shape)
        print(adj.shape)
        # x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = F.elu(self.out_proj(x))
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        print("X is",x.shape)
        x = F.dropout(x, self.dropout, training=self.training)
        # x = F.elu(self.out_proj(x))
        x = F.elu(self.out_att(x, adj))
        x = self.mlp(x)
        print("x is",x.shape)


        return x          

import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution




class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nembed, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)

        return x

class GCN_Classifier(nn.Module):
    def __init__(self, nembed, nhid, nclass, dropout):
        super(GCN_Classifier, self).__init__()

        self.gc1 = GraphConvolution(nembed, nhid)
        self.mlp = nn.Linear(nhid, nclass)
        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.mlp.weight,std=0.05)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.mlp(x)

        return x        


