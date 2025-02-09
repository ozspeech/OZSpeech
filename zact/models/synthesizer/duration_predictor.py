import math
import torch
import torch.nn as nn
from collections import OrderedDict
from zact.utils.tools import get_mask_from_lengths, pad


class VarianceAdaptor(nn.Module):
    """Variance Adaptor"""

    def __init__(self, model_config):
        super(VarianceAdaptor, self).__init__()
        self.duration_predictor = VariancePredictor(model_config)
        self.length_regulator = LengthRegulator()

    def forward(
        self,
        x,
        src_lens,
        src_mask,
        tgt_mask=None,
        max_len=None,
        duration_target=None,
        d_control=1.0,
    ):

        log_duration_prediction = self.duration_predictor(x, src_mask)
        
        if duration_target is not None:
            x, tgt_len = self.length_regulator(x, duration_target, src_lens, max_len)
            duration_rounded = duration_target
        else:
            duration_rounded = torch.clamp(
                (torch.round(torch.exp(log_duration_prediction) - 1) * d_control),
                min=0,
            )
            x, tgt_len = self.length_regulator(x, duration_rounded, src_lens, max_len)
            tgt_mask = get_mask_from_lengths(tgt_len)

        return (
            x,
            log_duration_prediction,
            duration_rounded,
            tgt_len,
            tgt_mask,
        )


class LengthRegulator(nn.Module):
    """Length Regulator"""

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, src_lens, max_len):
        output = list()
        tgt_len = list()
        for frame, expand_target, frame_len in zip(x, duration, src_lens):
            expanded = self.expand(frame, expand_target, frame_len)
            output.append(expanded)
            tgt_len.append(expanded.shape[0])

        if max_len is not None:
            output = pad(output, max_len)
        else:
            output = pad(output)

        return output, torch.LongTensor(tgt_len).to(x.device)

    def expand(self, frame, predicted, frame_len):
        out = list()
        for i, vec in enumerate(frame):
            if i < frame_len.item():
                expand_size = predicted[i].item()
            else:
                expand_size = 0
            out.append(vec.expand(max(int(expand_size), 0), -1))
        out = torch.cat(out, 0)

        return out

    def forward(self, x, duration, src_lens, max_len):
        output, tgt_len = self.LR(x, duration, src_lens, max_len)
        return output, tgt_len


class DurationPredictor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.input_size = cfg.variance_predictor.input_size
        self.filter_size = cfg.variance_predictor.filter_size
        self.kernel_size = cfg.variance_predictor.kernel_size
        self.conv_layers = cfg.variance_predictor.conv_layers
        self.cross_attn_per_layer = cfg.variance_predictor.cross_attn_per_layer
        self.attn_head = cfg.variance_predictor.attn_head
        self.drop_out = cfg.variance_predictor.drop_out

        self.conv = nn.ModuleList()
        self.cattn = nn.ModuleList()

        for idx in range(self.conv_layers):
            in_dim = self.input_size if idx == 0 else self.filter_size
            self.conv += [
                nn.Sequential(
                    nn.Conv1d(
                        in_dim,
                        self.filter_size,
                        self.kernel_size,
                        padding=self.kernel_size // 2,
                    ),
                    nn.ReLU(),
                    nn.LayerNorm(self.filter_size),
                    nn.Dropout(self.drop_out),
                )
            ]
            if idx % self.cross_attn_per_layer == 0:
                self.cattn.append(
                    torch.nn.Sequential(
                        nn.MultiheadAttention(
                            self.filter_size,
                            self.attn_head,
                            batch_first=True,
                            kdim=self.filter_size,
                            vdim=self.filter_size,
                        ),
                        nn.LayerNorm(self.filter_size),
                        nn.Dropout(0.2),
                    )
                )

        self.linear = nn.Linear(self.filter_size, 1)
        self.linear.weight.data.normal_(0.0, 0.02)

    def forward(self, x, mask, ref_emb, ref_mask):
        """
        input:
        x: (B, N, d)
        mask: (B, N), mask is 1
        ref_emb: (B, T', d)
        ref_mask: (B, T'), mask is 1

        output:
        dur_pred_log: (B, N)
        """
        
        x = x.transpose(1, -1)  # (B, N, d) -> (B, d, N)
        ref_emb = ref_emb.transpose(1, -1)

        for idx, (conv, act, ln, dropout) in enumerate(self.conv):
            res = x
            if idx % self.cross_attn_per_layer == 0:
                attn_idx = idx // self.cross_attn_per_layer
                attn, attn_ln, attn_drop = self.cattn[attn_idx]

                attn_res = y_ = x.transpose(1, 2)  # (B, d, N) -> (B, N, d)

                y_ = attn_ln(y_)
                y_, _ = attn(
                    y_,
                    ref_emb.transpose(1, 2),
                    ref_emb.transpose(1, 2),
                    key_padding_mask=ref_mask,
                )
                y_ = attn_drop(y_)
                y_ = (y_ + attn_res) / math.sqrt(2.0)

                x = y_.transpose(1, 2)

            x = conv(x)
            x = act(x)
            x = ln(x.transpose(1, 2))
            x = x.transpose(1, 2)

            x = dropout(x)

            if idx != 0:
                x += res

            if mask is not None:
                x = x * ~mask.unsqueeze(1)

        x = self.linear(x.transpose(1, 2))
        x = torch.squeeze(x, -1)

        return x
        

class VariancePredictor(nn.Module):
    """Duration, Pitch and Energy Predictor"""

    def __init__(self, model_config):
        super(VariancePredictor, self).__init__()

        self.input_size = model_config["transformer"]["encoder_hidden"]
        self.filter_size = model_config["variance_predictor"]["filter_size"]
        self.kernel = model_config["variance_predictor"]["kernel_size"]
        self.conv_output_size = model_config["variance_predictor"]["filter_size"]
        self.dropout = model_config["variance_predictor"]["drop_out"]

        self.conv_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1d_1",
                        Conv(
                            self.input_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2,
                        ),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        Conv(
                            self.filter_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=1,
                        ),
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output, mask):
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out.squeeze(-1)

        if mask is not None:
            out = out.masked_fill(mask, 0.0)

        return out


class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        w_init="linear",
    ):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x


if __name__=='__main__':
    from omegaconf import OmegaConf
    
    cfg = {
        'variance_predictor': {
            'input_size': 256,
            'filter_size': 256,
            'kernel_size': 3,
            'conv_layers': 30,
            'cross_attn_per_layer': 3,
            'attn_head': 8,
            'drop_out': 0.5
        }
    }
    durpred = DurationPredictor(OmegaConf.create(cfg))
    print(durpred)
    print('total size: ', sum(p.numel() for p in durpred.parameters()))