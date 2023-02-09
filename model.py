import torch.nn as nn
import torch
from typing import Dict, Tuple
import dgl.nn.pytorch as dglnn
from heterographconv import HeteroGraphConv

class HeteroGraphSAGE(nn.Module):
    def __init__(self,
                 input_dropout: float,
                 dropout: float,
                 hidden_dim: int,
                 feat_dict: Dict[Tuple[str, str, str],
                                 Tuple[int, int, int]]):
        super().__init__()
        self.feat_dict = feat_dict
        self.hidden_dim = hidden_dim

        self.conv1 = HeteroGraphConv(
            {rel: dglnn.SAGEConv(in_feats=(feats[0], feats[1]),
                                 out_feats=hidden_dim,
                                 aggregator_type='lstm',
                                 feat_drop=input_dropout,
                                 activation=nn.GELU())
             for rel, feats in feat_dict.items()},
            aggregate='sum')

        self.clf = nn.Sequential(
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2)
        )

        self.norm = nn.LayerNorm(hidden_dim)


    def forward(self, blocks, h_dict: dict) -> dict:
        h_dict = self.conv1(blocks[0], h_dict)
        h_dict = {k: self.norm(v) for k, v in h_dict.items()}
        return self.clf(h_dict['review'])

