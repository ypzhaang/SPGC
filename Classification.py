import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
from torch_geometric.datasets import TUDataset
import numpy as np
import os
import shutil
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class Config:
    dataset = 'MUTAG'
    coarsening_ratio = 0.5
    solver_params = {
        'lambda_param': 0.4,
        'alpha_param': 0.8,
        'gamma_param': 0.05,
        'beta_sim': 0.5,
        'max_iters': 50,
        'epsilon': 1e-8
    }
    gnn_params = {
        'hidden_dim': 64,
        'epochs': 100,
        'lr': 0.01,
        'batch_size': 64
    }
    mlp_params = {
        'hidden1': 256,
        'hidden2': 128,
        'dropout': 0.5
    }
    k_folds = 10

class ThreeLayerMLP(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, Config.mlp_params['hidden1'])
        self.fc2 = torch.nn.Linear(Config.mlp_params['hidden1'], Config.mlp_params['hidden2'])
        self.fc3 = torch.nn.Linear(Config.mlp_params['hidden2'], num_classes)
        self.dropout = torch.nn.Dropout(Config.mlp_params['dropout'])
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

class GCNExtractor(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, Config.gnn_params['hidden_dim']).to(device)
        self.conv2 = GCNConv(Config.gnn_params['hidden_dim'], Config.gnn_params['hidden_dim']).to(device)
    
    def forward(self, data):
        data = data.to(device)
        x = F.relu(self.conv1(data.x, data.edge_index))
        x = F.relu(self.conv2(x, data.edge_index))
        return global_mean_pool(x, torch.zeros(x.size(0), dtype=torch.long, device=device))
