import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=768, mlp_ratio = 4, num_heads=128, dropout= 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim,num_heads=num_heads)
        self.norm = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(embed_dim * mlp_ratio, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm(x + self.drop(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.drop(ffn_out))
        return x

class TRM(nn.Module):
    def __init__(self, vocab_size:int, embed_dim:int=256, nhead:int=8, num_layers:int=4, n_sup:int=16, max_len:int=512, dropout:float=0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = embed_dim
        self.n_sup = n_sup

        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_len, d_model)

        self.z0 = nn.Parameter(torch.zeros(embed_dim))
