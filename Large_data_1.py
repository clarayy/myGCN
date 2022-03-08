import os.path as osp
from math import ceil
from scipy.sparse.coo import coo_matrix
import torch
import torch.nn.functional as F
import os
import torch_geometric.transforms as T
from MyData_fromTU import MyData_fromTU
from MyOwnData import MyOwnDataset
from torch_geometric.loader import DenseDataLoader
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool
import numpy as np
import cv2
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
max_nodes = 1000
class MyFilter(object):
    def __call__(self, data):
        return data.num_nodes <= max_nodes
bmname = 'BA1000'
sname = '_p0.1_m10_test'
path = osp.join(osp.dirname(osp.realpath(__file__)), 'data',
                bmname)
dataset = MyData_fromTU(path, name=bmname+ sname, transform=T.ToDense(max_nodes),
                    pre_filter=MyFilter())
# # max_nodes = 67
# class MyFilter(object):
#     def __call__(self, data):
#         return data.num_nodes <= max_nodes
# bmname = 'high67/high67_p0.1_ob0.5_m10/processed'
# sname = '_p0.1_ob0.5_m10'
# path_processed = osp.join(osp.dirname(osp.realpath(__file__)), 'data',
#                 bmname)
# dataset = MyOwnDataset(path_processed, name=bmname + sname)
# dataset = dataset.shuffle()
# for idx in range(1,2):
#     dataset = torch.load(osp.join(path_processed, 'data_{}.pt'.format(idx)))
# dataset_1,slice = torch.load(osp.join(path_processed, 'data_1.pt'))
# test_loader = DenseDataLoader(dataset_1, batch_size=20, shuffle=True)
# dataset_2 = torch.load(osp.join(path_processed, 'data_2.pt'))
# dataset = dataset_1[2]+dataset_2[2]
n = (len(dataset) + 9) // 10
#test_dataset = dataset[:n]
val_dataset = dataset[n:2 * n]
train_dataset = dataset[2 * n:]
train_loader = DataLoader(dataset=train_dataset, batch_size=20, shuffle=True, num_workers=1)
val_loader = DataLoader(dataset=val_dataset, batch_size=20, shuffle=True, num_workers=1)
#test_dataloader = DataLoader(dataset=test_dataset, batch_size=20, shuffle=True, num_workers=1)
print('num_of_trainData:', len(train_dataset))
print('num_of_valData:', len(val_dataset))
for data in enumerate(train_loader):
    print(data)

max_nodes=62
class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 normalize=False, lin=True):
        super().__init__()

        self.conv1 = DenseSAGEConv(in_channels, hidden_channels, normalize)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels, normalize)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = DenseSAGEConv(hidden_channels, out_channels, normalize)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)

        if lin is True:
            self.lin = torch.nn.Linear(2 * hidden_channels + out_channels,
                                       out_channels)
        else:
            self.lin = None

    def bn(self, i, x):
        batch_size, num_nodes, num_channels = x.size()

        x = x.view(-1, num_channels)
        x = getattr(self, f'bn{i}')(x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x, adj, mask=None):
        batch_size, num_nodes, in_channels = x.size()

        x0 = x
        x1 = self.bn(1, F.relu(self.conv1(x0, adj, mask)))
        x2 = self.bn(2, F.relu(self.conv2(x1, adj, mask)))
        x3 = self.bn(3, F.relu(self.conv3(x2, adj, mask)))

        x = torch.cat([x1, x2, x3], dim=-1)

        if self.lin is not None:
            x = F.relu(self.lin(x))

        return x


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

        num_nodes = ceil(0.25 * max_nodes)
        self.gnn1_pool = GNN(2, 64, num_nodes)
        self.gnn1_embed = GNN(2, 64, 64, lin=False)

        num_nodes = ceil(0.25 * num_nodes)
        self.gnn2_pool = GNN(3 * 64, 64, num_nodes)
        self.gnn2_embed = GNN(3 * 64, 64, 64, lin=False)

        self.gnn3_embed = GNN(3 * 64, 64, 64, lin=False)

        self.lin1 = torch.nn.Linear(3 * 64, 64)
        self.lin2 = torch.nn.Linear(64, 50)

    def forward(self, x, adj, mask=None):
        s = self.gnn1_pool(x, adj, mask)
        x = self.gnn1_embed(x, adj, mask)

        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)

        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)

        x, adj, l2, e2 = dense_diff_pool(x, adj, s)

        x = self.gnn3_embed(x, adj)

        x = x.mean(dim=1)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1), l1 + l2, e1 + e2


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train(epoch):
    model.train()
    loss_all = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output, _, _ = model(data.x, data.adj, data.mask)
        loss = F.nll_loss(output, data.y.view(-1))
        loss.backward()
        loss_all += data.y.size(0) * loss.item()
        optimizer.step()
    return loss_all / len(train_dataset)


# @torch.no_grad()
# def test(loader):
#     model.eval()
#     correct = 0
#
#     for data in loader:
#         data = data.to(device)
#         pred = model(data.x, data.adj, data.mask)[0].max(dim=1)[1]
#         correct += pred.eq(data.y.view(-1)).sum().item()
#     return correct / len(loader.dataset)
@torch.no_grad()                 #加上GCN distance
def test(loader):
    model.eval()
    correct = 0
    labels = []
    preds = []
    for data in loader:
        data = data.to(device)
        pred = model(data.x, data.adj, data.mask)[0].max(dim=1)[1]
        correct += pred.eq(data.y.view(-1)).sum().item()

        labels.append(np.squeeze(data['y'].data.numpy()))
        preds.append(pred.cpu().data.numpy())

    labels = np.hstack(labels)
    preds = np.hstack(preds)
    # print("labels:", labels)
    # print("preds:", preds)
    read_dic = np.load(bmname+"_short_path.npy", allow_pickle=True).item()
    # print(read_dic[2][3])
    distance = []
    for i in range(len(labels)):
        a = read_dic[labels[i]][preds[i]]
        distance.append(a)
    #print(distance)
    result_dis = {}
    sum_dis = 0
    for i in set(distance):
        result_dis[i] = distance.count(i)
        sum_dis = i * result_dis[i] + sum_dis
    average_dis = sum_dis / len(labels)

    return correct / len(loader.dataset), average_dis

best_val_acc = test_acc = 0
for epoch in range(1, 51):
    train_loss = train(epoch)
    val_acc, val_GCN = test(val_loader)
    # if val_acc > best_val_acc:
    #     test_acc, test_GCN = test(test_loader)
    #     best_val_acc = val_acc
    print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, '
          f'Val Acc: {val_acc:.4f} '
          f'Val_GCN: {val_GCN:.4f}')
    if epoch % 10 == 0:
        torch.save(model.state_dict(), 'model/' + bmname+ '_diff_pool.pt')
#保存模型
torch.save(model.state_dict(), 'model/' + bmname + '_diff_pool.pt')
