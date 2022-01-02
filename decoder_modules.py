from utils import *

# Decoder
class Decoder(nn.Module):
    def __init__(self, embed_dim, latent_dim, device, num_heads, dropout=0.2):
        super(Decoder, self).__init__()
        
        self.multi_head_attention = MultiHeadedAttention(embed_dim, latent_dim, device, num_heads)
        self.normalize1 = Normalization(embed_dim)
        self.normalize2 = Normalization(embed_dim)
        self.normalize3 = Normalization(embed_dim)

        self.dp1 = nn.Dropout(dropout)
        self.dp2 = nn.Dropout(dropout)
        self.dp3 = nn.Dropout(dropout)

        self.W_Q = nn.Linear(embed_dim, embed_dim)
        self.feed_forward = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, K, V, enc_mask=None, dec_mask=None):
        # x -> Batch_size x Seq_len x embed_dim
        
        # Self-attention uses decoder mask
        x = x + self.dp1(self.multi_head_attention(x, dec_mask))
        x = self.normalize1(x)

        # Cross-attention
        Q = self.W_Q(x)
        scores = torch.matmul(Q, torch.transpose(K,1,2)) / sqrt(x.shape[-1])

        # Use encoder mask
        if enc_mask is not None:
            scores = scores.masked_fill(enc_mask == 0, -float('inf'))

        scores = F.softmax(scores, dim=-1)

        x = x + self.dp2(torch.matmul(scores, V))
        x = self.normalize2(x)

        x = x + self.dp3(F.relu(self.feed_forward(x)))
        x = self.normalize3(x)
        return x

class DecoderStack(nn.Module):
    def __init__(self, num_unique_tokens, max_sequence_len, embed_dim, latent_dim, device, num_heads, num_decoders, dropout=0.2):
        super(DecoderStack, self).__init__()

        self.embedding = Embedding(num_unique_tokens, embed_dim)
        self.pos_encoding = PositionalEncoding(max_sequence_len, embed_dim, device)
        self.decoders = nn.ModuleList([Decoder(embed_dim, latent_dim, device, num_heads, dropout) for _ in range(num_decoders)])

        self.linear = nn.Linear(embed_dim, num_unique_tokens)

    def forward(self, x, K, V, enc_mask=None, dec_mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for i, decoder in enumerate(self.decoders):
            x = decoder(x, K, V, enc_mask, dec_mask)
        return self.linear(x) # softmax activation applied by the cross entropy loss function

