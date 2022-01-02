from utils import *

class Encoder(nn.Module):
    def __init__(self, embed_dim, latent_dim, device, num_heads, dropout=0.2):
        super(Encoder, self).__init__()
        
        self.multi_head_attention = MultiHeadedAttention(embed_dim, latent_dim, device, num_heads)
        self.normalize1 = Normalization(embed_dim)
        self.normalize2 = Normalization(embed_dim)

        self.dp1 = nn.Dropout(dropout)
        self.dp2 = nn.Dropout(dropout)

        self.feed_forward = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        # x -> Batch_size x Seq_len x embed_dim
        
        x = x + self.dp1(self.multi_head_attention(x, mask))
        x = self.normalize1(x)
        x = x + self.dp2(F.relu(self.feed_forward(x)))
        x = self.normalize2(x)
        return x

class EncoderStack(nn.Module):
    def __init__(self, num_unique_tokens, max_sequence_len, embed_dim, latent_dim, device, num_heads, num_encoders, dropout=0.2):
        super(EncoderStack, self).__init__()

        self.embedding = Embedding(num_unique_tokens, embed_dim)
        self.pos_encoding = PositionalEncoding(max_sequence_len, embed_dim, device)
        self.encoders = nn.ModuleList([Encoder(embed_dim, latent_dim, device, num_heads, dropout) for _ in range(num_encoders)])

        self.W_K = nn.Linear(embed_dim, embed_dim)
        self.W_V = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for i, encoder in enumerate(self.encoders):
            x = encoder(x, mask)
        K = self.W_K(x)
        V = self.W_V(x)
        return K, V

