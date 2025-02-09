import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .estimator import ZACTEstimator, RMSNorm
from zact.utils.tools import get_mask_from_lengths


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        assert dim % 2 == 0, "SinusoidalPosEmb requires dim to be even"
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        self.emb = torch.exp(torch.arange(half_dim).float() * -emb)

    def forward(self, x, scale=1000):
        if x.ndim < 1:
            x = x.unsqueeze(0)
        emb = self.emb.to(x.device)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    
    
class AbsPosEmbedding(nn.Module):
    def __init__(self, max_seq_len, dim):
        super(AbsPosEmbedding, self).__init__()

        self.pos_ids = torch.arange(max_seq_len).expand((1, -1))
        self.emb = nn.Embedding(max_seq_len, dim)
        
    def forward(self, x):
        _, x_len, _ = x.shape
        pos_ids = self.pos_ids[:,:x_len]
        pos_emb = self.emb(pos_ids.to(x.device))
        x = x + pos_emb
        return x
    
    
class TimeEmbedding(nn.Module):
    def __init__(
        self,
        hidden_dim,
        time_emb_scale,
        ):
        super(TimeEmbedding, self).__init__()
        
        self.emb = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * time_emb_scale),
            nn.SiLU(),
            nn.Linear(hidden_dim * time_emb_scale, hidden_dim)
        )
        
    def forward(self, t):
        return self.emb(t)


class TimeEstimator(nn.Module):
    def __init__(self, hidden_dim, ffn_dim, n_heads, n_layers):
        super().__init__()
        
        self.time_token = nn.Parameter(torch.rand(1, 1, hidden_dim))
        self.time_estimator = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=n_heads,
                dim_feedforward=ffn_dim,
                batch_first=True,
            ),
            num_layers=n_layers
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
                
    def forward(self, x):
        t = self.time_token.expand(x.size(0), -1, -1)
        x_t = torch.cat([t, x], dim=1)
        x_t = self.time_estimator(x_t)
        t = self.head(x_t[:,0,:]) # (b, 1)
        return t
    

class QuantizerEmbedding(nn.Module):
    def __init__(self, n_quantizers, hidden_dim):
        super(QuantizerEmbedding, self).__init__()
        
        self.quantizer_ids = torch.arange(n_quantizers).expand((1, -1))
        self.quantizer_emb = nn.Embedding(n_quantizers, hidden_dim)
        
    def forward(self, x):
        q_emb = self.quantizer_emb(self.quantizer_ids.to(x.device))
        q_emb = q_emb.unsqueeze(2).expand(-1, -1, x.size(2), -1).contiguous()
        x = x + q_emb
        return x


class ZACTEmbedding(nn.Module):
    def __init__(
        self, 
        input_dim, 
        hidden_dim, 
        time_ffn_dim,
        time_n_heads,
        time_n_layers,
        time_scale, 
        n_quantizers, 
        max_seq_len
        ):
        
        super().__init__()
        self.quantizer_embedding = QuantizerEmbedding(
            n_quantizers=n_quantizers,
            hidden_dim=input_dim,
        )
        
        self.proj_in = nn.Sequential(
            nn.Linear(
                in_features=n_quantizers * input_dim,
                out_features=hidden_dim
            ),
            nn.Tanh(),
            nn.Linear(
                in_features=hidden_dim,
                out_features=hidden_dim
            ),
            RMSNorm(hidden_dim)
        )
        
        self.time_estimator = TimeEstimator(
            hidden_dim=hidden_dim,
            ffn_dim=time_ffn_dim,
            n_heads=time_n_heads,
            n_layers=time_n_layers,
        )
        self.time_embedding = TimeEmbedding(
            hidden_dim=hidden_dim,
            time_emb_scale=time_scale,
        )
        
        self.time_norm = RMSNorm(hidden_dim)
        
        self.pos_embedding = AbsPosEmbedding(
            max_seq_len=max_seq_len,
            dim=hidden_dim,     
        )
        
    def forward(self, prior, prompts):
        
        # Quantizer Embedding
        prior = self.quantizer_embedding(prior)
        prompts = self.quantizer_embedding(prompts)
        
        # Folding
        bx, qx, lx, dx = prior.shape
        prior = prior.permute(0, 2, 1, 3).contiguous()
        prior = prior.view(bx, lx, qx * dx)
        prior = self.proj_in(prior)
        
        _, _, lx, _ = prompts.shape
        prompts = prompts.permute(0, 2, 1, 3).contiguous()
        prompts = prompts.view(bx, lx, qx * dx)
        prompts = self.proj_in(prompts)
        
        # Time estimation
        time = self.time_estimator(prior)
        
        # Concat
        xt_noise = prior + torch.randn_like(prior)
        zt = torch.cat([prompts, xt_noise], dim=1)
        
        # Time Embedding
        t = self.time_embedding(time)
        t = t.expand(-1, zt.size(1), -1).contiguous()
        zt = zt + t
        zt = self.time_norm(zt)
        
        # Position Embedding
        zt = self.pos_embedding(zt)
        
        return zt, time.unsqueeze(2).unsqueeze(3)


class FlowMatching(nn.Module):
    def __init__(self, config):
        super(FlowMatching, self).__init__()
        
        self.config = config
        self.estimator = ZACTEstimator(config['estimator'])
        self.embedding = ZACTEmbedding(
            input_dim=config['estimator']['input_dim'],
            hidden_dim=config['estimator']['hidden_dim'],
            time_ffn_dim=config['estimator']['ffn_dim'],
            time_n_heads=config['estimator']['n_heads'],
            time_n_layers=config['estimator']['time_n_layers'],
            time_scale=config['estimator']['time_scale'],
            n_quantizers=config['estimator']['n_quantizers'],
            max_seq_len=config['estimator']['max_seq_len']
        )
    
    def forward(self, xt, mask):
        return self.estimator(xt, mask)
    
    def compute_loss(
        self, 
        prior, 
        x1_tgt, 
        x_len, 
        x_max_len,
        prompts,
        ):
        """Compute Flow Matching loss

        Args:
            prior (BxQxLxD): prior codes
            x1_tgt (BxQxL): target codes
            x_len (B): sequence length of target codes
            x_max_len (1): max sequence length of target 
            prompts (BxQxL): acoustic prompt
        """

        # Embedding
        x1 = self.estimator.get_word_embedding(x1_tgt)
        prompts = self.estimator.get_word_embedding(prompts)
        prompts_len = prompts.size(2)
        
        # get mask
        mask = get_mask_from_lengths(prompts_len + x_len, x_max_len + prompts_len)
        
        zt, time = self.embedding(prior, prompts)

        # compute flow matching loss
        vt, mask = self.forward(zt, mask)
        vt = vt[:,:,prompts_len:,:]
        mask = mask[:,:,prompts_len:,:]

        dx = 1 / (1 - time) * (x1 - prior) * ~mask
        fm_loss = F.mse_loss(vt, dx)
        
        # compute anchor loss
        x1_est = prior + (1 - time) * vt
        x1_est = x1_est * ~mask
        x1_est = self.estimator.get_logits(x1_est)
        x1_est = x1_est.permute(0, 3, 1, 2).contiguous() # (BxQxLxD) -> (BxDxQxL)

        anchor_loss = F.cross_entropy(
            x1_est, 
            x1_tgt, 
            ignore_index=self.config["estimator"]["vocab_size"]
        )

        return {
            'flow_loss': fm_loss, 
            'anchor_loss': anchor_loss, 
            # 'tau': time.squeeze().detach().clone().cpu().item()
        }
    
    def sampling(
        self, 
        prior, 
        x_len, 
        x_max_len,
        prompts,
        temperature=1.0,
    ):
        """Fixed solver for ODEs.

        Args:
            prior (BxQxLxD): prior codes
            x_len (B): sequence length of target codes
            x_max_len (1): max sequence length of target codes
            prompts (BxQxL): prompt codes
            temperature (float, optional): temperature. Defaults to 1.0.

        Returns:
            (BxLxD): output
        """
        # mask content quantizers & get embedding
        prompts[:,1:3,:] = self.config.estimator.vocab_size
        bs, qs, _ = prompts.shape
        eos = torch.zeros(
            (bs, qs, 1), 
            device=prompts.device, 
            dtype=prompts.dtype,
        ) + self.config.estimator.vocab_size
        prompts = torch.cat([prompts, eos], dim=-1)
        prompts = self.estimator.get_word_embedding(prompts)
        prompts_len = prompts.size(2)
        
        # get mask
        mask = get_mask_from_lengths(prompts_len + x_len, x_max_len + prompts_len)
        
        zt, time = self.embedding(prior, prompts)
        
        zt, time = self.embedding(prior, prompts)

        return self._onestep_sampling(
            zt=zt,
            prior=prior,
            time=time,
            mask=mask,
            prompts_len=prompts_len
        )
    
    def _onestep_sampling(
        self, 
        zt,
        prior,
        time,
        mask,
        prompts_len,
        **kwargs,
    ):
        vt, _ = self.forward(zt, mask)
        vt = vt[:,:,prompts_len:,:]
        x1 = prior + (1 - time) * vt
        
        logits = self.estimator.get_logits(x1)
        logits = logits.permute(0, 3, 1, 2).contiguous() # (b, c, n, l)
        
        return {'logits': logits, 'x1': x1}
    
    def _euler_sampling(
        self, 
        zt,
        mask,
        prompts,
        prompts_len,
        n_timesteps,
        **kwargs,
    ):    
        ts = torch.linspace(self.time.item(), 1, n_timesteps + 1, device=xt.device)
        
        for i in range(1, len(ts)):
            dt = ts[i] - ts[i-1]
            vt, _ = self.forward(zt, mask, self.time)
            zt = zt + dt * vt
            zt[:,:,:prompts_len,:] = prompts
            
        xt = zt[:,:,prompts_len:,:]
        logits = self.estimator.get_logits(xt)
        logits = logits.permute(0, 3, 1, 2).contiguous() # (b, c, n, l)
                
        return {'logits': logits, 'x1': xt}
        