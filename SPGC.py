import torch
import numpy as np
import pandas as pd
import json
import chardet
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data
from torch_geometric.utils import is_undirected, to_dense_adj, degree, dense_to_sparse
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_data():
    data = torch.load(r'/mnt/5468e/daihuan/Q/SpeqNets-pycharm/graph_data_ucd_pre_correct.pt')

    def to_undirected(data):
        edge_index = data.edge_index
        edge_index = torch.tensor(edge_index).t()
        reversed_edges = edge_index[[1, 0], :]
        combined_edges = torch.cat([edge_index, reversed_edges], dim=1)
        unique_edges = torch.unique(combined_edges.T, dim=0).T
        
        undirected_data = Data(x=data.x, edge_index=unique_edges)
        for key in data.keys():
            if key not in ['edge_index', 'x']:
                undirected_data[key] = data[key]
        return undirected_data
    
    undirected_data = to_undirected(data)
    return undirected_data

class NodeSimilarityCalculator:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
    
    def get_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, 
                               padding=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    
    def calculate_similarity_matrix(self, explanation_file, node_names):
        with open(explanation_file, 'rb') as file:
            raw_data = file.read()
            encoding = chardet.detect(raw_data)['encoding']
        with open(explanation_file, 'r', encoding=encoding) as file:
            lines = file.readlines()
        nodes = []
        node_explanations = {}
        for line in lines:
            parts = line.strip().split(':', 1)
            if len(parts) == 2:
                node_name = parts[0].strip()
                explanation = parts[1].strip()
                if node_name in node_names:
                    nodes.append(node_name)
                    node_explanations[node_name] = explanation
        embeddings = {}
        for node in tqdm(nodes, desc=" "):
            embeddings[node] = self.get_embedding(node_explanations[node])
        num_nodes = len(nodes)
        similarity_matrix = np.zeros((num_nodes, num_nodes))
        for i in tqdm(range(num_nodes)):
            for j in range(num_nodes):
                vector_i = embeddings[nodes[i]]
                vector_j = embeddings[nodes[j]]
                cos_sim = np.dot(vector_i, vector_j) / (
                    np.linalg.norm(vector_i) * np.linalg.norm(vector_j))
                similarity_matrix[i][j] = (cos_sim + 1) / 2
        df = pd.DataFrame(similarity_matrix, index=nodes, columns=nodes)
        df.to_csv('similarity_matrix_formula5.csv')
        return similarity_matrix, nodes, embeddings

class GraphMatrixCalculator:
    
    def __init__(self, data, similarity_matrix=None):
        self.data = data
        self.similarity_matrix = similarity_matrix
        self.nodes = data.node_names
        
    def top_b_elements(self, matrix, b):
        flat_matrix = matrix.flatten()
        sorted_indices = np.argsort(flat_matrix)[::-1] 
        result = np.zeros_like(matrix)
        for idx in sorted_indices[:b]:
            i = idx // matrix.shape[1]
            j = idx % matrix.shape[1]
            result[i, j] = matrix[i, j]    
        return result
    
    def calculate_matrices(self, b_ratio=0.1):
        edge_index = self.data.edge_index
        A = to_dense_adj(edge_index)[0].numpy()
        A_binary = (A > 0).astype(float) 
        S = self.similarity_matrix
        term1 = A_binary * S
        term2 = (1 - A_binary) * S
        b = int(b_ratio * term2.size) 
        term2_top_b = self.top_b_elements(term2, b)
        W = term1 + term2_top_b
        degrees = W.sum(axis=1)
        D = np.diag(degrees)
        L = D - W
        return A_binary, W, D, L
    
    def save_matrices(self, A, W, D, L):
        for matrix, name in zip([A, W, D, L], ['A', 'W', 'D', 'L']):
            df = pd.DataFrame(matrix, index=self.nodes, columns=self.nodes)
            df.to_csv(f'{name}.csv')


class SPGC:

    def __init__(self, X, k, lambda_param, beta_param, alpha_param, 
                 gamma_param, beta_sim, S, thresh=1e-10, lr0=1e-5):
        self.X = X
        self.k = k 
        self.S = S 
        self.beta_sim = beta_sim         
        self.p = X.shape[0] 
        self.n = X.shape[1] 
        
        self.thresh = thresh
        self.X_tilde = np.random.normal(0, 1, (k, self.n))
        self.C = np.random.normal(0, 1, (self.p, k))
        self.C[self.C < thresh] = thresh
        
        self.beta_param = beta_param
        self.alpha_param = alpha_param
        self.lambda_param = lambda_param
        self.gamma_param = gamma_param
        self.lr0 = lr0
        self.iters = 0
    
    def get_lr(self):
        return self.lr0
    
    def calc_f(self, L):
        fw = 0
        J = np.outer(np.ones(self.k), np.ones(self.k)) / self.k
        log_det_term = -self.gamma_param * np.linalg.slogdet(
            self.C.T @ L @ self.C + J)[1]
        fw += log_det_term
        trace_term = np.trace(self.X_tilde.T @ self.C.T @ L @ self.C @ self.X_tilde)
        fw += trace_term
        recon_error = (self.alpha_param / 2) * np.linalg.norm(
            self.C @ self.X_tilde - self.X, 'fro') ** 2
        fw += recon_error
        l12_norm = (self.lambda_param / 2) * np.linalg.norm(
            self.C.T, ord=2) ** 2
        fw += l12_norm
        sim_penalty = self.beta_sim * np.sum(
            np.exp(-self.S) * (self.C @ self.C.T))
        fw += sim_penalty
        return fw
    
    def update_X_tilde(self, L):
        L_tilde = self.C.T @ L @ self.C
        A = 2 * L_tilde / self.alpha_param + self.C.T @ self.C
        b = self.C.T @ self.X
        self.X_tilde = np.linalg.pinv(A) @ b
        for i in range(len(self.X_tilde)):
            norm = np.linalg.norm(self.X_tilde[i])
            if norm > 0:
                self.X_tilde[i] = self.X_tilde[i] / norm
    
    def grad_C(self, L):
        J = np.outer(np.ones(self.k), np.ones(self.k)) / self.k
        gradC = np.zeros(self.C.shape)
        M = self.C.T @ L @ self.C + J
        inv_M = np.linalg.pinv(M)
        gradC += -2 * self.gamma_param * L @ self.C @ inv_M
        gradC += 2 * L @ self.C @ self.X_tilde @ self.X_tilde.T
        gradC += self.alpha_param * (
            (self.C @ self.X_tilde - self.X) @ self.X_tilde.T)
        gradC += self.lambda_param * self.C
        gradC += 2 * self.beta_sim * (np.exp(-self.S) @ self.C)
        return gradC
    
    def update_C(self, L):
        lr = self.get_lr()
        self.C = self.C - lr * self.grad_C(L)
        self.C[self.C < self.thresh] = self.thresh
        for i in range(len(self.C)):
            norm = np.linalg.norm(self.C[i])
            if norm > 0:
                self.C[i] = self.C[i] / norm
    
    def fit(self, L, max_iters=10, inner_iters=100):
        loss_history = []
        for i in tqdm(range(max_iters), desc=" "):
            for _ in range(inner_iters):
                self.update_C(L) 
            self.update_X_tilde(L)
            loss = self.calc_f(L)
            loss_history.append(loss)
            self.iters += 1    
        return self.C, self.X_tilde, loss_history
    
class MultilayerSPGC:
    def __init__(self, data, similarity_matrix, params=None):
        self.data = data
        self.original_similarity = similarity_matrix
        self.node_names = data.node_names if hasattr(data, 'node_names') else None
        default_params = {
            'coarsening_ratio': 0.5,      
            'lambda_param': 0.5,         
            'beta_param': 0,             
            'alpha_param': 0.5,           
            'gamma_param': None, 
            'beta_sim': 0.7,
            'max_layers': 10,             
            'iterations_per_layer': 10,   
            'inner_iterations': 100,      
        }
        if params:
            default_params.update(params)
        self.params = default_params
        if self.params['gamma_param'] is None:
            self.params['gamma_param'] = data.x.shape[1] / 2
        self.layers = []
        self.tree_hierarchy = {}
        if data.x.is_sparse:
            self.original_X = data.x.to_dense().numpy()
        else:
            self.original_X = data.x.numpy()
        self.original_L = self._compute_laplacian(data)
        
    def _compute_laplacian(self, data):
        edge_index = data.edge_index
        A = to_dense_adj(edge_index)[0].numpy()
        degrees = A.sum(axis=1)
        D = np.diag(degrees)
        L = D - A
        return L
    
    def _compute_next_layer_similarity(self, X_tilde):
        similarity = cosine_similarity(X_tilde)
        return similarity
    
    def coarsen_single_layer(self, current_X, current_L, current_S, layer_idx):
        k = max(1, int(self.params['coarsening_ratio'] * current_L.shape[0]))
        coarsener = SPGC(
            X=current_X,
            k=k,
            lambda_param=self.params['lambda_param'],
            beta_param=self.params['beta_param'],
            alpha_param=self.params['alpha_param'],
            gamma_param=self.params['gamma_param'],
            beta_sim=self.params['beta_sim'],
            S=current_S,
            L=current_L,
            thresh=1e-10,
            lr0=1e-5
        ) 
        C, X_tilde, loss_history = coarsener.fit(
            max_iters=self.params['iterations_per_layer'],
            inner_iters=self.params['inner_iterations']
        )
        next_L = C.T @ current_L @ C
        next_S = self._compute_next_layer_similarity(X_tilde)
        node_mapping = get_node_mapping(C)
       
        layer_info = {
            'layer_idx': layer_idx,
            'C': C,
            'X_tilde': X_tilde,
            'L': next_L,
            'S': next_S,
            'node_mapping': node_mapping,
            'num_nodes': current_L.shape[0],
            'next_num_nodes': next_L.shape[0],
            'loss_history': loss_history
        }
        self.layers.append(layer_info)
        return next_L, X_tilde, next_S, node_mapping
    
    def build_tree_hierarchy(self):
        self.tree_hierarchy[0] = {
            idx: [name] for idx, name in enumerate(self.node_names)
        }
        for layer_idx, layer_info in enumerate(self.layers, 1):
            node_mapping = layer_info['node_mapping']
            new_mapping = {}
            for super_node in range(layer_info['next_num_nodes']):
                children = [n for n, s in node_mapping.items() if s == super_node]
                all_original_names = []
                for child in children:
                    if child in self.tree_hierarchy[layer_idx-1]:
                        all_original_names.extend(self.tree_hierarchy[layer_idx-1][child])
                new_mapping[super_node] = all_original_names
            self.tree_hierarchy[layer_idx] = new_mapping
    
    def coarsen_multiple_layers(self):
        current_L = self.original_L
        current_X = self.original_X
        current_S = self.original_similarity
        for layer_idx in range(self.params['max_layers']):
            next_L, next_X, next_S, node_mapping = self.coarsen_single_layer(
                current_X, current_L, current_S, layer_idx
            )
            current_L = next_L
            current_X = next_X
            current_S = next_S
        self.build_tree_hierarchy()
       
class NodeMappingAnalyzer:
    
    def __init__(self, data, C_matrix):
        self.data = data
        self.C = C_matrix
    
    def get_node_mapping(self):
        node_to_super_node = {}
        for i in range(self.C.shape[0]):
            super_node_index = np.argmax(self.C[i])
            node_to_super_node[i] = super_node_index
        super_node_to_original_nodes = {}
        for node, super_node in node_to_super_node.items():
            if super_node not in super_node_to_original_nodes:
                super_node_to_original_nodes[super_node] = []
            super_node_to_original_nodes[super_node].append(
                self.data.node_names[node])
        return node_to_super_node, super_node_to_original_nodes
    
    def analyze_similarity(self, explanation_file, super_node_to_original_nodes):
        model = SentenceTransformer('all-MiniLM-L6-v2')
        node_descriptions = {}
        with open(explanation_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(':', 1)
                if len(parts) == 2:
                    node_name = parts[0].strip()
                    node_descriptions[node_name] = parts[1].strip()
       
        descriptions = list(node_descriptions.values())
        embeddings = model.encode(descriptions, convert_to_tensor=True)
        node_embeddings = dict(zip(node_descriptions.keys(), embeddings))
        
        similarity_results = {}
        sorted_super_nodes = sorted(super_node_to_original_nodes.keys())
        
        for super_node in tqdm(sorted_super_nodes, desc=" "):
            original_nodes = super_node_to_original_nodes[super_node]
            
            vectors = []
            missing_nodes = []
            for node in original_nodes:
                if node in node_embeddings:
                    vectors.append(node_embeddings[node].cpu().numpy())
                else:
                    missing_nodes.append(node)
     
            sim_matrix = cosine_similarity(vectors)
            sim_matrix = (sim_matrix + 1) / 2
            
            similarity_results[super_node] = {
                'original_nodes': original_nodes,
                'similarity_matrix': sim_matrix,
                'mean_similarity': np.mean(sim_matrix)
            }
        
        return similarity_results
