"""
Date: created on 06/05/2023
References:
    paper: (arXiv'17) Attentional factorization machines: Learning the weight of feature interactions via attention networks
    url: https://arxiv.org/abs/1708.04617
    code: https://github.com/xue-pai/FuxiCTR/blob/main/model_zoo/AFM/src/AFM.py
Authors: yinpu, email: yinpu.mail@gmail.com
"""
import torch
from torch import nn
from ...basic.layers import LR, EmbeddingLayer

class AFMLayer(nn.Module):
    def __init__(self, num_fields):
        super().__init__()
        self.triu_index = nn.Parameter(torch.triu_indices(num_fields, num_fields, offset=1), requires_grad=False)
    
    def forward(self, x):
        emb1 = torch.index_select(x, 1, self.triu_index[0])
        emb2 = torch.index_select(x, 1, self.triu_index[1])
        return emb1 * emb2

class AFM(nn.Module):
    """Attentional factorization machines
    Args:
        features (list): the list of `Feature Class`. training by MLP. It means the user profile features and context features in origin paper, exclude history and target features.
        history_features (list): the list of `Feature Class`,training by ActivationUnit. It means the user behaviour sequence features, eg.item id sequence, shop id sequence.
        target_features (list): the list of `Feature Class`, training by ActivationUnit. It means the target feature which will execute target-attention with history feature.
        mlp_params (dict): the params of the last MLP module, keys include:`{"dims":list, "activation":str, "dropout":float, "output_layer":bool`}
        attention_mlp_params (dict): the params of the ActivationUnit module, keys include:`{"dims":list, "activation":str, "dropout":float, "use_softmax":bool`}
    """
    def __init__(self, 
                 linear_features, 
                 afm_features,
                 attention_dropout=(0, 0),
                 attention_dim=10,
                 use_attention=True,):
        super().__init__()
        self.use_attention = use_attention
        self.linear_features = linear_features
        self.afm_features = afm_features
        linear_dim =  sum([fea.embed_dim for fea in linear_features])
        embedding_dim = afm_features[0].embed_dim
        num_fm_fields = len(afm_features)
        self.embedding = EmbeddingLayer(linear_features + afm_features)
        self.linear_layer = LR(linear_dim)
        self.afm_layer = AFMLayer(num_fm_fields)
        self.attention_layer = nn.Sequential(nn.Linear(embedding_dim, attention_dim),
                                       nn.ReLU(),
                                       nn.Linear(attention_dim, 1, bias=False),
                                       nn.Softmax(dim=1))
        self.weight_p = nn.Linear(embedding_dim, 1, bias=False)
        self.dropout1 = nn.Dropout(attention_dropout[0])
        self.dropout2 = nn.Dropout(attention_dropout[1])
    
    def forward(self, x):
        input_linear = self.embedding(x, self.linear_features, squeeze_dim=True)
        input_afm = self.embedding(x, self.afm_features, squeeze_dim=False)
        
        y_linear = self.linear_layer(input_linear)
        
        afm_product = self.afm_layer(input_afm)
        if self.use_attention:
            attention_weight = self.attention_layer(afm_product)
            attention_weight = self.dropout1(attention_weight)
            attention_sum = torch.sum(attention_weight * afm_product, dim=1)
            attention_sum = self.dropout2(attention_sum)
            y_afm = self.weight_p(attention_sum)
        else:
            y_afm = torch.flatten(afm_product, start_dim=1).sum(dim=-1).unsqueeze(-1)
            
        y = y_linear + y_afm
        return torch.sigmoid(y.squeeze(1))
        
        
        
        
        