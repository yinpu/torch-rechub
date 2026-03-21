"""
Date: create on 04/05/2022
References:
    paper: (SIGIR'2018) Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate
    url: https://arxiv.org/abs/1804.07931
Authors: Mincai Lai, laimincai@shanghaitech.edu.cn
"""

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from ...basic.layers import MLP, EmbeddingLayer


class ESMM(nn.Module):
    """Entire Space Multi-Task Model

    Args:
        user_features (list): the list of `Feature Class`, training by shared bottom and tower module. It means the user features.
        item_features (list): the list of `Feature Class`, training by shared bottom and tower module. It means the item features.
        cvr_params (dict): the params of the CVR Tower module, keys include:`{"dims":list, "activation":str, "dropout":float`}
        ctr_params (dict): the params of the CTR Tower module, keys include:`{"dims":list, "activation":str, "dropout":float`}
    """

    def __init__(self, user_features, item_features, cvr_params, ctr_params):
        super().__init__()
        self.user_features = user_features
        self.item_features = item_features
        self.embedding = EmbeddingLayer(user_features + item_features)
        self.tower_dims = len(user_features) * user_features[0].embed_dim + len(item_features) * item_features[0].embed_dim
        self.tower_cvr = MLP(self.tower_dims, **cvr_params)
        self.tower_ctr = MLP(self.tower_dims, **ctr_params)

    def forward(self, x):
        # # Field-wise Pooling Layer for user and item
        # embed_user_features = self.embedding(x, self.user_features, squeeze_dim=False).sum(dim=1)  #[batch_size, embed_dim]
        # embed_item_features = self.embedding(x, self.item_features,
        # squeeze_dim=False).sum(dim=1)  #[batch_size, embed_dim]

        # Here we concat all the features instead of field-wise pooling them
        # [batch_size, num_features, embed_dim] --> [batch_size, num_features * embed_dim]
        _batch_size = self.embedding(x, self.user_features, squeeze_dim=False).shape[0]
        embed_user_features = self.embedding(x, self.user_features, squeeze_dim=False).reshape(_batch_size, -1)
        embed_item_features = self.embedding(x, self.item_features, squeeze_dim=False).reshape(_batch_size, -1)

        # print('embed_user_features', embed_user_features.shape)

        input_tower = torch.cat((embed_user_features, embed_item_features), dim=1)
        cvr_logit = self.tower_cvr(input_tower)
        ctr_logit = self.tower_ctr(input_tower)
        cvr_pred = torch.sigmoid(cvr_logit)
        ctr_pred = torch.sigmoid(ctr_logit)
        ctcvr_pred = torch.mul(ctr_pred, cvr_pred)

        ys = [cvr_pred, ctr_pred, ctcvr_pred]
        return torch.cat(ys, dim=1)

    def compute_task_losses(self, predicts, targets, default_loss_fns):
        """Compute ESMM losses with CVR defined only on clicked samples."""
        if len(default_loss_fns) != 3:
            raise ValueError(f"ESMM expects 3 loss functions, got {len(default_loss_fns)}")

        click_mask = targets[:, 1] == 1
        if click_mask.any():
            cvr_loss = default_loss_fns[0](predicts[click_mask, 0], targets[click_mask, 0])
        else:
            cvr_loss = predicts[:, 0].new_tensor(0.0)
        ctr_loss = default_loss_fns[1](predicts[:, 1], targets[:, 1])
        ctcvr_loss = default_loss_fns[2](predicts[:, 2], targets[:, 2])
        return [cvr_loss, ctr_loss, ctcvr_loss]

    def aggregate_task_losses(self, loss_list):
        """Optimize ESMM with CTR and CTCVR losses, keeping CVR loss for reporting."""
        if len(loss_list) != 3:
            raise ValueError(f"ESMM expects 3 task losses, got {len(loss_list)}")
        return sum(loss_list[1:])

    def compute_task_metrics(self, targets, predicts, default_metric_fns=None):
        """Compute ESMM metrics with CVR evaluated only on clicked samples."""
        if default_metric_fns is None:
            default_metric_fns = [roc_auc_score, roc_auc_score, roc_auc_score]
        if len(default_metric_fns) != 3:
            raise ValueError(f"ESMM expects 3 metric functions, got {len(default_metric_fns)}")

        click_mask = targets[:, 1] == 1
        cvr_targets = targets[click_mask, 0]
        cvr_predicts = predicts[click_mask, 0]
        if len(cvr_targets) == 0 or len(set(cvr_targets.tolist())) < 2:
            cvr_metric = float("nan")
        else:
            cvr_metric = default_metric_fns[0](cvr_targets, cvr_predicts)

        ctr_metric = default_metric_fns[1](targets[:, 1], predicts[:, 1])
        ctcvr_metric = default_metric_fns[2](targets[:, 2], predicts[:, 2])
        return [cvr_metric, ctr_metric, ctcvr_metric]
