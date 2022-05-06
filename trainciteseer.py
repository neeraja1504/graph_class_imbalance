from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import math
# import matplotlib as mpl
# mpl.use('TkAgg')
import matplotlib.pyplot as plt
import argparse
import numpy as np
import torch 
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset

import warnings
warnings.filterwarnings('ignore')




 

from utils import load_data, accuracy, split_arti, print_class_acc , load_data_Blog , split_genuine , print_class_acc_test
from models import GAT, SpGAT, GAT_Classifier, SpGAT_Classifier
tb = SummaryWriter()
mainlist=[]
labellist=[]


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if(torch.cuda.is_available()):
    gpu=0
    device='cuda:{}'.format(gpu)
else:
  device='cpu'    


# adj, features, labels = load_data_Blog()
# im_class_num = 14 #set it to be the number less than 100
# class_sample_num = 20 

# idx_train, idx_val, idx_test, class_num_mat = split_genuine(labels)
# Load data
# l=[]
# adj, features, labels= load_data()
# class_sample_num = 20
# im_class_num = 3
# print(labels[0])
# # for i in labels:
# #   if(i==0):
# #     l.append(i)
# #   if(i==1):
# #     l.append(i)
# # labels=torch.tensor(l) 
# # torch.set_printoptions(profile="full")   
# # print(labels)     
    


# print(adj.shape)

# c_train_num = []
# for i in range(labels.max().item() + 1):
#     if  i > labels.max().item()-im_class_num: #only imbalance the last classes
#         c_train_num.append(int(class_sample_num*0.5))

#     else:
#         c_train_num.append(class_sample_num)
# print(c_train_num)        

# idx_train, idx_val, idx_test, class_num_mat = split_arti(labels, c_train_num)
 
# for i in idx_val:
#   if(labels[i]>3):
#     min_val.append(i)
# print(min_val)       

# # for i in majority:
# #   for j in range(2708):
# #     if(adj[i][j]>0 and labels[j]==0):
# #       print("Majority node",i,labels[i],"Minority node",j,"Label",labels[j])
# #       neigh.append(j)
# # print(neigh)      
      
dataset = CiteseerGraphDataset()



graph = dataset[0]
    
graph = dgl.add_self_loop(graph)
print(graph)
adj=graph.adj()
print(adj)
adj = torch.FloatTensor(np.array(adj.to_dense()))
print(adj)
print(adj.shape)


    # retrieve the number of classes
n_classes = dataset.num_classes
lista=[]
listb=[]
listc=[]
listd=[]
liste=[]
listf=[]
    # retrieve labels of ground truth
labels = graph.ndata.pop('label').to(device).long()
for i in labels:
  if(i==0):
    lista.append(1)
  if(i==1):
    listb.append(1)
  if(i==2):
    listc.append(1)
  if(i==3):
    listd.append(1)
  if(i==4):
    liste.append(1) 
  if(i==5):
    listf.append(1)   
s= len(lista)+len(listb)+len(listc)+len(listd)+len(liste)+len(listf)
print(len(lista),len(listb),len(listc),len(listd),len(liste),len(listf))
print(s/6)    
    # Extract node features
feats = graph.ndata.pop('feat').to(device)
n_features = feats.shape[-1]
features=feats
print(n_features)

# class_sample_num = 20
# im_class_num = 2

# c_train_num = []
# for i in range(labels.max().item() + 1):
#     if  i > labels.max().item()-im_class_num: #only imbalance the last classes
#         c_train_num.append(int(class_sample_num*0.5))

#     else:
#         c_train_num.append(class_sample_num)
# print(c_train_num)        

# idx_train, idx_val, idx_test, class_num_mat = split_arti(labels, c_train_num)
# print(idx_train)

    #retrieve masks for train/validation/test
train_mask = graph.ndata.pop('train_mask')
val_mask = graph.ndata.pop('val_mask')
test_mask = graph.ndata.pop('test_mask')
print(train_mask)

idx_train = torch.nonzero(train_mask, as_tuple=False).squeeze().to(device)
idx_val = torch.nonzero(val_mask, as_tuple=False).squeeze().to(device)
idx_test = torch.nonzero(test_mask, as_tuple=False).squeeze().to(device)   

majority=[] 
minority=[]
min_val=[]
majority_id=[]
neigh=[]
for i in range(len(idx_train)):
  if(labels[idx_train[i]]==0):
    minority.append(idx_train[i])
    
  # if(labels[idx_train[i]]==5):
  #   minority.append(idx_train[i])
  #   print(labels[idx_train[i]])
    # majority_id.append(i)
  else:
    majority.append(idx_train[i]) 
print(idx_train)    
print(majority)
print(majority_id,len(majority_id))
print(minority)

# Model and optimizer
if args.sparse:
    model = SpGAT(nfeat=n_features, 
                nhid=args.hidden, 
                nclass=int(labels.max()) + 1, 
                dropout=args.dropout, 
                nheads=args.nb_heads, 
                alpha=args.alpha)
    classifier= SpGAT_Classifier(nembed=args.hidden, 
                nhid=args.hidden, 
                nclass=labels.max().item() + 1, 
                dropout=args.dropout,
                nheads=args.nb_heads, 
                alpha=args.alpha)             
else:
    model = GAT(nfeat=n_features, 
                nhid=args.hidden, 
                nclass=int(labels.max()) + 1, 
                dropout=args.dropout, 
                nheads=args.nb_heads, 
                alpha=args.alpha)
    classifier= GAT_Classifier(nembed=args.hidden, 
                nhid=args.hidden, 
                nclass=labels.max().item() + 1, 
                dropout=args.dropout,
                nheads=args.nb_heads, 
                alpha=args.alpha)            
optimizer = optim.Adam(model.parameters(), 
                       lr=args.lr, 
                       weight_decay=args.weight_decay)
                       
optimizer_cls = optim.Adam(classifier.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)                       

if args.cuda:
    model.cuda()
    classifier.cuda()
    features = features.cuda()
    adj = torch.tensor(adj).cuda()
    labels = labels.cuda()
    # class_num_mat=class_num_mat.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()
   

    features, adj, labels = Variable(features), Variable(adj), Variable(labels)


def train(epoch):
    t = time.time()
    model.train()
    classifier.train()
    optimizer.zero_grad()
    optimizer_cls.zero_grad()
    embed = model(features, adj)
    print("embed shape is",embed.shape)
    output = classifier(embed, adj)
    weight = features.new((labels.max().item()+1)).fill_(1)
    # weight[-im_class_num:] = 1+0.95
    for i in range(6):
        c_idx = (labels==i).nonzero()[:,-1].tolist()
        weight[i]=1/math.sqrt(len(c_idx))

    # reg=0    

    # for i in minority:
    #   sub=adj[i] - classifier.attentions[1].weight[i]
    #   reg=reg+torch.linalg.norm(sub) 
      
    # print("reg is",reg)  

    # alpha=1
    # gamma=2.5
    # ce_loss= F.cross_entropy(output[idx_train], labels[idx_train], weight=weight,reduction='mean') 
    # pt = torch.exp(-ce_loss)
    # loss_train = ((alpha * (1-pt)**gamma * ce_loss).mean()) 

    #PolyLoss
    ce_loss= F.cross_entropy(output[idx_train], labels[idx_train], weight=weight,reduction='mean')
    pt=torch.sum(torch.matmul(labels[idx_train].type(torch.float),(torch.nn.Softmax()(output[idx_train].type(torch.float)))),dim=-1)
    loss_train= ce_loss + 0*(1-pt)

   
    # loss_train = F.cross_entropy(output[idx_train], labels[idx_train], weight=weight) 
    # print("Train loss is",loss_train) 
    # loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()
    optimizer_cls.step()
    # list1=[]
    # list2=[]
    # att=[]
    # att2=[]
    # list3=[]
    # list4=[]

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        classifier.eval()
        output = classifier(embed, adj)

    loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print_class_acc(output[idx_train], labels[idx_train], 0)
    # for i in idx_train:
    #   if(labels[i]>3):
    #     att.append(classifier.attentions[0].weight[i].cpu().detach().numpy())
    #     att2.append(str(i.cpu().detach().numpy()))
        # print(i.cpu().detach().numpy())
        # print(i)
    # fig, ax = plt.subplots(figsize =(10, 7))
    # for i in att:
    #   ax.hist(i,bins = [0,0.01,0.02,0.03,0.04,0.05,0.1,0.2,0.3,0.4])

    # for i,j in zip(att2,att):
    #   plt.bar(i,j, color ='maroon')
     
    #   plt.savefig("squares.png")

    
        
    # plt.show();    

    # print(classifier.attentions[0].weight[idx_train[0]])
    # classifier.attentions[0].weight[idx_train[0]].cpu().detach().numpy()
#     if(epoch==args.epochs-1):
#       for i in range(2708):
#          if(classifier.attentions[0].weight[idx_train[7]][i])>0:
#            list1.append(i)
#            list2.append(classifier.attentions[0].weight[idx_train[7]][i].cpu().detach().numpy())
#         # print(classifier.attentions[0].weight[idx_train[0]][i].cpu().detach().numpy())
# # #     fig = plt.figure(figsize = (10, 5))
#       print(list1)
#       print(list2)

    # if(epoch==args.epochs-1):
    #   for i in minority:
    #     list1=[]
    #     list2=[]
    #     list3=[]
    #     print("Node id and label",i,labels[i])
    #     for j in range(3327):
    #       if(classifier.attentions[1].weight[i][j]>0):
    #         list1.append(j)
    #         list2.append((classifier.attentions[1].weight[i][j]))
    #         list3.append((labels[j]))  
    #     print(list1)
    #     print(list3)
    #     print(list2) 
        # mainlist.append(list2) 
        # labellist.append(list3)    
    
    # with open("/content/gdrive/MyDrive/Neeraja/noreg.txt", 'w') as output:
    #    for row in mainlist:
    #       output.write(str(row) + '\n')

    # with open("/content/gdrive/MyDrive/Neeraja/label.txt", 'w') as output:
    #    for row in labellist:
    #       output.write(str(row) + '\n')      

    

 
# # creating the bar plot
#     plt.bar(list1, list2, color ='maroon',
#         width = 0.4) 
#     plt.show()       

    # tb.add_histogram('attention for single node', classifier.attentions[0].weight[idx_train[0]], epoch)
    # tb.add_scalar('Loss', loss_val, epoch)
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()),
          'loss_val: {:.4f}'.format(loss_val.data.item()),
          'acc_val: {:.4f}'.format(acc_val.data.item()),
          'time: {:.4f}s'.format(time.time() - t))

    return loss_val.data.item()


def compute_test(epoch=0):
    model.eval()
    classifier.eval()
    embed = model(features, adj)
    output = classifier(embed, adj)
   
    loss_test = F.cross_entropy(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data.item()),
          "accuracy= {:.4f}".format(acc_test.data.item()))
    print_class_acc_test(output[idx_test], labels[idx_test], 0, pre='test')

# Train model
# if args.load is not None:
#     load_model(args.load)

t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)

    # if epoch % 10 == 0:
    #     compute_test(epoch)

    # if epoch % 100 == 0:
    #     save_model(epoch)


print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
compute_test()    
      

# # Train model
# t_total = time.time()
# loss_values = []
# bad_counter = 0
# best = args.epochs + 1
# best_epoch = 0
# for epoch in range(args.epochs):
#     loss_values.append(train(epoch))

#     torch.save(model.state_dict(), '{}.pkl'.format(epoch))
#     if loss_values[-1] < best:
#         best = loss_values[-1]
#         best_epoch = epoch
#         bad_counter = 0
#     else:
#         bad_counter += 1

#     if bad_counter == args.patience:
#         break

#     files = glob.glob('*.pkl')
#     for file in files:
#         epoch_nb = int(file.split('.')[0])
#         if epoch_nb < best_epoch:
#             os.remove(file)

# files = glob.glob('*.pkl')
# for file in files:
#     epoch_nb = int(file.split('.')[0])
#     if epoch_nb > best_epoch:
#         os.remove(file)

# print("Optimization Finished!")
# print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# # Restore best model
# print('Loading {}th epoch,which is the best epoch'.format(best_epoch))
# model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))

# # Testing
# compute_test()
