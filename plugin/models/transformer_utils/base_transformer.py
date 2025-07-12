import torch.nn as nn

from mmengine.registry import MODELS

@MODELS.register_module()
class PlaceHolderEncoder(nn.Module):

    def __init__(self, *args, embed_dims=None, **kwargs):
        super(PlaceHolderEncoder, self).__init__()
        self.embed_dims = embed_dims

    def forward(self, *args, query=None, **kwargs):
        
        return query