import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Linear
import torch_geometric.transforms as T
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GraphMultisetTransformer
from MyData_fromTU import MyData_fromTU
import numpy as np

# max_nodes = 62
# class MyFilter(object):
#     def __call__(self, data):
#         return data.num_nodes <= max_nodes
bmname = 'dolphins'
# path = osp.join(osp.dirname(osp.realpath(__file__)), 'data',
#                 bmname)
# dataset = MyData_fromTU(path, name=bmname+'_p0.1_m10', transform=T.ToDense(max_nodes),
#                     pre_filter=MyFilter())
# dataset = dataset.shuffle()
path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', bmname)
dataset = MyData_fromTU(path, name=bmname+'_p0.5_m50').shuffle()
# path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'PROTEINS')
# dataset = TUDataset(path, name='PROTEINS')
avg_num_nodes = int(dataset.data.num_nodes / len(dataset))
n = (len(dataset) + 9) // 10
test_dataset = dataset[:n]
val_dataset = dataset[n:2 * n]
train_dataset = dataset[2 * n:]
test_loader = DataLoader(test_dataset, batch_size=20)
val_loader = DataLoader(val_dataset, batch_size=20)
train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True)


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = GCNConv(dataset.num_features, 32)
        self.conv2 = GCNConv(32, 32)
        self.conv3 = GCNConv(32, 32)

        self.pool = GraphMultisetTransformer(
            in_channels=96,
            hidden_channels=64,
            out_channels=32,
            Conv=GCNConv,
            num_nodes=avg_num_nodes,
            pooling_ratio=0.25,
            pool_sequences=['GMPool_G', 'SelfAtt', 'GMPool_I'],
            num_heads=4,
            layer_norm=False,
        )

        self.lin1 = Linear(32, 16)
        self.lin2 = Linear(16, dataset.num_classes)

    def forward(self, x0, edge_index, batch):
        x1 = self.conv1(x0, edge_index).relu()
        x2 = self.conv2(x1, edge_index).relu()
        x3 = self.conv3(x2, edge_index).relu()
        x = torch.cat([x1, x2, x3], dim=-1)

        x = self.pool(x, batch, edge_index)

        x = self.lin1(x).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)

        return x.log_softmax(dim=-1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)


def train():
    model.train()

    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        total_loss += data.num_graphs * float(loss)
        optimizer.step()
    return total_loss / len(train_dataset)


# @torch.no_grad()
# def test(loader):
#     model.eval()
#
#     total_correct = 0
#     for data in loader:
#         data = data.to(device)
#         out = model(data.x, data.edge_index, data.batch)
#         total_correct += int((out.argmax(dim=-1) == data.y).sum())
#     return total_correct / len(loader.dataset)
@torch.no_grad()
def test(loader):
    model.eval()

    total_correct = 0
    labels = []
    preds = []
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        total_correct += int((out.argmax(dim=-1) == data.y).sum())

        labels.append(np.squeeze(data['y'].data.numpy()))
        preds.append(out.argmax(dim=-1).cpu().data.numpy())
    labels = np.hstack(labels)
    preds = np.hstack(preds)
    # print("labels:", labels)
    # print("preds:", preds)
    read_dic = np.load(bmname + "_short_path.npy", allow_pickle=True).item()
    # print(read_dic[2][3])
    distance = []
    for i in range(len(labels)):
        a = read_dic[labels[i]][preds[i]]
        distance.append(a)
    # print(distance)
    result_dis = {}
    sum_dis = 0
    for i in set(distance):
        result_dis[i] = distance.count(i)
        sum_dis = i * result_dis[i] + sum_dis
    average_dis = sum_dis / len(labels)
    return total_correct / len(loader.dataset), average_dis

for epoch in range(1, 201):
    train_loss = train()
    val_acc, val_GCN = test(val_loader)
    test_acc, test_GCN = test(test_loader)
    print(f'Epoch: {epoch:03d}, Loss: {train_loss:.4f}, '
          f'Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f} '
          f'Val_GCN: {val_GCN:.4f}, Test_GCN:{test_GCN:.4f}')
# for epoch in range(1, 201):
#     train_loss = train()
#     val_acc = test(val_loader)
#     test_acc = test(test_loader)
#     print(f'Epoch: {epoch:03d}, Loss: {train_loss:.4f}, '
#           f'Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f} '
#           )