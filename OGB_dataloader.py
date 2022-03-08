from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import DataLoader
import torch
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = PygGraphPropPredDataset(name= "ogbg-molhiv", root= "data/")
print(dataset.num_classes)
split_idx = dataset.get_idx_split()
test_loader =  DataLoader(dataset[split_idx["test"]],batch_size=32,shuffle=False)
print(dataset[0])
print(dataset[0].y)
for data in test_loader:
    #data = data.to(device)
    print(data[0])
    print(len(data[0].y))