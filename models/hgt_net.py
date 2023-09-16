import torch.nn as nn

from .resnet_ import RebuildResNet101
from paper.transformer import Transformer


class HGT(nn.Module):

    def __init__(self):
        super(HGT,self).__init__()
        self.resnet=RebuildResNet101()

        self.transformer=Transformer(**{ "hgt_num":20,"d_model":128, "N":6, "heads":8,"h0":3,"w0":3})

    def forward(self,x):
        x=self.resnet(x)
        output=self.transformer(x)
        return output
