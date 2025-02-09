import torch
import torch.nn as nn
import torch.nn.functional as F
    

class RMSNorm(torch.nn.Module):
    def __init__(self, dim, norm_eps=1e-5):
        super().__init__()
        self.norm_eps = norm_eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.norm_eps)
    
    def forward(self, x):
        out = self._norm(x.float()).type_as(x)
        return out * self.weight


class SwiGLUFeedForward(nn.Module):
    def __init__(self, dim, dim_ffn):
        super().__init__()

        self.w1 = nn.Linear(dim, dim_ffn, bias=False)
        self.w3 = nn.Linear(dim, dim_ffn, bias=False)
        self.w2 = nn.Linear(dim_ffn, dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
    
    
class TransformerLayer(nn.Module):
    
    def __init__(self, dim, dim_ffn, n_heads):
        super().__init__()
        
        self.rmsnorm_1 = RMSNorm(dim=dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=n_heads,
            bias=False,
            batch_first=True
        )
        self.rmsnorm_2 = RMSNorm(dim=dim)
        self.ffn = SwiGLUFeedForward(dim=dim, dim_ffn=dim_ffn)
        
    def forward(self, x, key_padding_mask):
        residual = x
        x = self.rmsnorm_1(x)
        x, _ = self.attn(query=x, key=x, value=x, key_padding_mask=key_padding_mask)
        x = x + residual
        residual = x
        x = self.rmsnorm_2(x)
        x = self.ffn(x)
        x = x + residual
        return x
    
    
class TransformerModel(nn.Module):
    def __init__(self, dim, dim_ffn, n_heads, n_layers):
        super().__init__()
        self.model = nn.ModuleList([
            TransformerLayer(dim=dim, dim_ffn=dim_ffn, n_heads=n_heads)
            for _ in range(n_layers)
        ])
        
    def forward(self, x, key_padding_mask):
        for layer in self.model:
            x = layer(x, key_padding_mask)
        return x


class ZACTEstimator(nn.Module):
    def __init__(self, cfg):
        super(ZACTEstimator, self).__init__()
        self.cfg = cfg
        self.n_quantizers = cfg['n_quantizers']
        
        self.word_embedding = nn.Embedding(
            cfg['vocab_size'] + 1, 
            cfg['input_dim'], 
            padding_idx=cfg['vocab_size'],
        )
        self.lm_head = nn.Linear(cfg['input_dim'], cfg['vocab_size'] + 1, bias=False)
        with torch.no_grad():
            self.lm_head.weight = self.word_embedding.weight
        
        self.model = TransformerModel(
            dim=cfg['hidden_dim'],
            dim_ffn=cfg['ffn_dim'],
            n_heads=cfg['n_heads'],
            n_layers=cfg['n_layers']
        )
        
        self.proj_out_1 = nn.Sequential(
            RMSNorm(cfg['hidden_dim']),
            nn.Linear(
                in_features=cfg['hidden_dim'],
                out_features=cfg['n_quantizers'] * cfg['input_dim']
            )
        )
        
        self.proj_out_2 = nn.Sequential(
            RMSNorm(cfg['input_dim']),
            nn.Linear(
                in_features=cfg['input_dim'],
                out_features=cfg['input_dim']
            )
        )
        
    def get_word_embedding(self, tokens):
        return self.word_embedding(tokens)

    def get_logits(self, hidden):
        return self.lm_head(hidden)
                
    def forward(self, xt, mask):
        """Velocity estimation

        Args:
            xt (BxQxLxD): x at timestep t
            mask
            t (B): timestep

        Returns:
            (BxQxLxD): velocity at given timestep
        """
        
        vt = self.model(xt, key_padding_mask=mask)
        
        vt = self.proj_out_1(vt)
        bv, lv, dv = vt.shape
        vt = vt.view(bv, lv, self.n_quantizers, dv // self.n_quantizers)
        vt = vt.permute(0, 2, 1, 3).contiguous()
        vt = self.proj_out_2(vt)
        mask = mask.unsqueeze(1).expand(-1, vt.size(1), -1).unsqueeze(3)
        vt = vt * ~mask
        
        return vt, mask