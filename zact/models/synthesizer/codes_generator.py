import torch
import torch.nn as nn
from zact.models.module import (
    Encoder, 
    Decoder,
    DecoderLayer,
)
import torch.nn.functional as F
from .duration_predictor import VarianceAdaptor
from zact.utils.tools import get_mask_from_lengths


class CodesGenerator(nn.Module):
    def __init__(self, config):
        super(CodesGenerator, self).__init__()
        
        self.config = config
        encoder_hidden = config["transformer"]["encoder_hidden"]
        decoder_hidden = config["transformer"]["decoder_hidden"]
        output_dim = config["transformer"]["output_dim"]
        vocab_size = config["codec"]["vocab_size"]
        n_quantizers = config["codec"]["n_quantizers"]

        self.encoder = Encoder(config)
        self.variance_adaptor = VarianceAdaptor(config)
        self.bridge_decoder = nn.Linear(encoder_hidden, decoder_hidden)
        self.shared_decoder = Decoder(config)
        
        self.codes_estimator = nn.ModuleList()
        for _ in range(n_quantizers):
            self.codes_estimator.append(DecoderLayer(config))
                
        self.proj_out = nn.Sequential(
            nn.Linear(
                in_features=decoder_hidden,
                out_features=output_dim
            ),
            nn.Tanh()
        )
        self.head = nn.Linear(
            in_features=decoder_hidden,
            out_features=vocab_size + 1, 
        )
    
    def forward(
        self,
        texts,
        src_lens,
        max_src_len,
        tgt_lens=None,
        max_tgt_len=None,
        d_targets=None,
        d_control=1.0,
        ):
        
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        tgt_masks = (
            get_mask_from_lengths(tgt_lens, max_tgt_len)
            if tgt_lens is not None
            else None
        )
        
        output = self.encoder(texts, src_masks)

        (
            output,
            log_d_predictions,
            d_rounded,
            tgt_lens,
            tgt_masks,
        ) = self.variance_adaptor(
            output,
            src_lens,
            src_masks,
            tgt_masks,
            max_tgt_len,
            d_targets,
            d_control,
        )
        
        output = self.bridge_decoder(output)
        output, tgt_masks = self.shared_decoder(output, tgt_masks)
                
        hiddens = []
        for estimator in self.codes_estimator:            
            output, tgt_masks = estimator(output, tgt_masks)
            hiddens.append(output.unsqueeze(1))
            
        hiddens = torch.cat(hiddens, dim=1) # (b, n, l, d)
        outputs = self.proj_out(hiddens)
    
        logits = self.head(hiddens)
        mask = tgt_masks.unsqueeze(1).expand(-1, logits.size(1), -1).unsqueeze(3)
        logits = logits * ~mask
        logits = logits.permute(0, 3, 1, 2).contiguous() # (b, c, n, l)

        return (
            outputs,
            logits,
            log_d_predictions,
            d_rounded,
            src_masks,
            tgt_masks,
            src_lens,
            tgt_lens,
        )
        
    def compute_loss(
        self, 
        codes_pred, 
        codes, 
        log_durations_pred, 
        durations,
        src_masks,
        ):
        
        prior_loss = F.cross_entropy(codes_pred, codes, ignore_index=self.config["codec"]["vocab_size"])
        
        log_duration_targets = torch.log(durations.float() + 1)
        log_duration_targets.requires_grad = False
        log_duration_targets = log_duration_targets.masked_select(~src_masks)
        log_durations_pred = log_durations_pred.masked_select(~src_masks)
        duration_loss = F.mse_loss(log_durations_pred, log_duration_targets)
        
        return {'prior_loss': prior_loss, 'duration_loss': duration_loss}