from encoder_modules import *
from decoder_modules import *

class Transformer(nn.Module):
    def __init__(self, num_unique_input_tokens, num_unique_output_tokens, max_sequence_len, embed_dim, latent_dim, device, num_heads, num_stacks, dropout=0.2):
        super(Transformer, self).__init__()
        self.encoder_stack = EncoderStack(num_unique_input_tokens, max_sequence_len, embed_dim, latent_dim, device, num_heads, num_stacks, dropout)
        self.decoder_stack = DecoderStack(num_unique_output_tokens, max_sequence_len, embed_dim, latent_dim, device, num_heads, num_stacks, dropout)
        

    def forward(self, enc_seq, dec_seq, enc_mask, dec_mask):
        K, V = self.encoder_stack(enc_seq, enc_mask)
        out = self.decoder_stack(dec_seq, K, V, enc_mask, dec_mask)
        return out
