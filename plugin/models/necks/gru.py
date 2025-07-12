import torch
import torch.nn as nn
from mmdet.registry import MODELS
from mmengine.model import kaiming_init, constant_init


@MODELS.register_module()
class ConvGRU(nn.Module):
    def __init__(self, out_channels):
        super(ConvGRU, self).__init__()
        kernel_size = 1
        padding = kernel_size // 2
        self.convz = nn.Conv2d(2*out_channels, 
            out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.convr = nn.Conv2d(2*out_channels, 
            out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.convq = nn.Conv2d(2*out_channels, 
            out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.ln = nn.LayerNorm(out_channels)
        self.zero_out = nn.Conv2d(out_channels, out_channels, 1, 1, bias=True)
        

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
        nn.init.zeros_(self.zero_out.weight)
        nn.init.zeros_(self.zero_out.bias)
        
    def forward(self, h, x):
        if len(h.shape) == 3:
            h = h.unsqueeze(0)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        
        hx = torch.cat([h, x], dim=1) # [1, 2c, h, w]
        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        new_x = torch.cat([r * h, x], dim=1) # [1, 2c, h, w]
        q = self.convq(new_x)

        out = ((1 - z) * h + z * q) # (1, C, H, W)
        out = self.ln(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        out = self.zero_out(out)
        out = out + x
        out = out.squeeze(0)

        return out
