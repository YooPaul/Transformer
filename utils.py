import torch
import torch.nn as nn
import torch.nn.functional as F
#from torchvision import transforms, utils
#import numpy as np
import multiprocessing
from math import sin, cos, sqrt


# A lookup table mapping integer index to an embedding
class Embedding(nn.Module):
    def __init__(self, num_unique_tokens, embed_dim):
        super(Embedding, self).__init__()
        self.embed = nn.Embedding(num_unique_tokens, embed_dim)

    def forward(self, x):
        # x -> N x Seq 
        x = self.embed(x)
        # x -> N x Seq x embed_dim
        return x

# A method for encoding positional information using sin and cos waves
class PositionalEncoding(nn.Module):
    def __init__(self, max_sequence_len, embed_dim, device):
        super(PositionalEncoding, self).__init__()

        self.position_matrix = torch.zeros((max_sequence_len, embed_dim), device=device, requires_grad=False)
        for pos in range(max_sequence_len):
            for i in range(0, embed_dim, 2):
                self.position_matrix[pos][i] = sin(pos / 10000**(2*i/embed_dim))
                self.position_matrix[pos][i + 1] = cos(pos / 10000**(2*(i + 1)/embed_dim))

        self.position_matrix = self.position_matrix.unsqueeze(0)

    def forward(self, x):
        sequence_len = x.shape[1]
        x = x + self.position_matrix[:,:sequence_len]
        return x

# Normalization module
class Normalization(nn.Module):
    def __init__(self, embed_dim, method='L'):
        super(Normalization, self).__init__()
        self.method = method

        if self.method not in ['L', 'B']:
            self.method = 'L'
        self.gamma = nn.Parameter(torch.ones(embed_dim))
        self.beta = nn.Parameter(torch.zeros(embed_dim))
        self.eps = 1e-7

    def forward(self, x):
        # x -> N x seq x embed_dim

        # Batch Norm
        if self.method == 'B':
            pass
        # Layer Norm 
        elif self.method == 'L':
            mu = torch.mean(x, dim=-1, keepdim=True)
            var = torch.var(x, dim=-1, keepdim=True)
            x = (x - mu) / torch.sqrt(var + self.eps)
            x = self.gamma * x + self.beta

        return x

# Attention mechanism
class Attention(nn.Module):
    def __init__(self, embed_dim, latent_dim, device):
        super(Attention, self).__init__()

        self.W_Q = nn.Linear(embed_dim, latent_dim, device=device)
        self.W_K = nn.Linear(embed_dim, latent_dim, device=device)
        self.W_V = nn.Linear(embed_dim, latent_dim, device=device)

        self.scale = sqrt(latent_dim)

    def forward(self, x, mask=None):
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        scores = torch.matmul(Q, torch.transpose(K,1,2)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -float('inf'))

        scores = F.softmax(scores, dim=-1)

        return torch.matmul(scores, V)

class MultiHeadedAttention(nn.Module):
    def __init__(self, embed_dim, latent_dim, device, num_heads):
        super(MultiHeadedAttention, self).__init__()

        if num_heads < 1:
            num_heads = 1

        self.attention_heads = nn.ModuleList([Attention(embed_dim, latent_dim, device) for _ in range(num_heads)])
        self.W = nn.Linear(latent_dim * num_heads, embed_dim) # bring back to the original input dimensions

    def forward(self, x, mask=None):
        heads = []
        for i, head in enumerate(self.attention_heads):
            heads.append(head(x, mask))
        z = torch.concat(heads, dim=-1)
        return self.W(z)
