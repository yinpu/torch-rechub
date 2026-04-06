import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from torch_rechub.basic.features import SparseFeature
from torch_rechub.models.multi_task import ESMM


class DictDataset(Dataset):
    """Minimal dataset returning ``(x_dict, y)`` pairs."""

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.y.size(0)

    def __getitem__(self, idx):
        return {name: value[idx] for name, value in self.x.items()}, self.y[idx]


class BinaryModel(nn.Module):
    """Simple binary classifier used by trainer hook tests."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x_dict):
        return torch.sigmoid(self.linear(x_dict["x"]))


class PairwiseModel(nn.Module):
    """Simple pair-wise matching model used by loss hook tests."""

    def __init__(self):
        super().__init__()
        self.pos_linear = nn.Linear(1, 1)
        self.neg_linear = nn.Linear(1, 1)

    def forward(self, x_dict):
        pos_score = self.pos_linear(x_dict["pos"]).view(-1)
        neg_score = self.neg_linear(x_dict["neg"]).view(-1)
        return pos_score, neg_score


class TwoTowerMatchModel(nn.Module):
    """Minimal two-tower model used for DataParallel loss-hook tests."""

    def __init__(self):
        super().__init__()
        self.user_linear = nn.Linear(1, 1)
        self.item_linear = nn.Linear(1, 1)

    def user_tower(self, x_dict):
        return self.user_linear(x_dict["user"])

    def item_tower(self, x_dict):
        return self.item_linear(x_dict["item"])

    def forward(self, x_dict):
        return torch.sigmoid(self.user_tower(x_dict) + self.item_tower(x_dict))


class MultiTaskBinaryModel(nn.Module):
    """Small two-task model used by MTL custom hook tests."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 2)

    def forward(self, x_dict):
        return torch.sigmoid(self.linear(x_dict["x"]))


class CountingBinaryLoss(object):
    """Callable loss wrapper that records how often it is used."""

    def __init__(self):
        self.calls = 0
        self.last_dtype = None

    def __call__(self, model, x_dict, y):
        self.calls += 1
        self.last_dtype = y.dtype
        y_pred = model(x_dict)
        return F.binary_cross_entropy(y_pred, y)


class CountingPairwiseLoss(object):
    """Callable pair-wise loss wrapper that records how often it is used."""

    def __init__(self):
        self.calls = 0

    def __call__(self, model, x_dict, y):
        self.calls += 1
        pos_score, neg_score = model(x_dict)
        return -(pos_score - neg_score).sigmoid().log().mean()


class TowerAccessLoss(object):
    """Custom match loss that requires the underlying two-tower model."""

    def __init__(self):
        self.calls = 0
        self.model_type = None

    def __call__(self, model, x_dict, y):
        del y
        self.calls += 1
        self.model_type = type(model)
        user_embedding = model.user_tower(x_dict)
        item_embedding = model.item_tower(x_dict)
        return (user_embedding * item_embedding).mean()


class LinearAccessBinaryLoss(object):
    """Custom CTR loss that requires the underlying binary model."""

    def __init__(self):
        self.calls = 0
        self.model_type = None

    def __call__(self, model, x_dict, y):
        self.calls += 1
        self.model_type = type(model)
        logits = model.linear(x_dict["x"])
        return F.binary_cross_entropy(torch.sigmoid(logits), y)


class CountingTaskLosses(object):
    """Batch-level loss hook for MTL custom loss tests."""

    def __init__(self):
        self.calls = 0

    def __call__(self, model, x_dict, ys, y_preds):
        self.calls += 1
        del model, x_dict
        return [
            F.binary_cross_entropy(y_preds[:, 0], ys[:, 0].float()),
            F.binary_cross_entropy(y_preds[:, 1], ys[:, 1].float()),
        ]


class HeadAccessTaskLosses(object):
    """Custom MTL loss that requires access to the underlying task heads."""

    def __init__(self):
        self.calls = 0
        self.model_type = None

    def __call__(self, model, x_dict, ys, y_preds):
        del x_dict
        self.calls += 1
        self.model_type = type(model)
        logits = model.linear.weight.sum()
        return [
            F.binary_cross_entropy(y_preds[:, 0], ys[:, 0].float()) + 0.0 * logits,
            F.binary_cross_entropy(y_preds[:, 1], ys[:, 1].float()) + 0.0 * logits,
        ]


class ShortTaskLosses(object):
    """Invalid MTL loss hook used to verify return-shape validation."""

    def __call__(self, model, x_dict, ys, y_preds):
        del model, x_dict, ys
        return [F.binary_cross_entropy(y_preds[:, 0], y_preds[:, 0].detach())]


class ESMMTaskLosses(object):
    """Minimal three-task loss hook used to mark ESMM custom loss mode."""

    def __call__(self, model, x_dict, ys, y_preds):
        del model, x_dict, ys, y_preds
        return [torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)]


class RecordingLogger(object):
    """Capture trainer metric payloads for assertions."""

    def __init__(self):
        self.history = []

    def log_hyperparams(self, params):
        self.history.append(("hyperparams", params))

    def log_metrics(self, metrics, step=None):
        self.history.append(("metrics", step, metrics))

    def finish(self):
        return None


def build_ctr_dataloader():
    """Create a deterministic single-task dataloader."""
    x = torch.linspace(-1.0, 1.0, steps=16).view(-1, 1)
    y = (x > 0).float()
    return DataLoader(DictDataset({"x": x}, y), batch_size=4, shuffle=False)


def build_ctr_int_label_dataloader():
    """Create a deterministic point-wise dataloader with integer labels."""
    x = torch.linspace(-1.0, 1.0, steps=16).view(-1, 1)
    y = (x > 0).long()
    return DataLoader(DictDataset({"x": x}, y), batch_size=4, shuffle=False)


def build_pairwise_dataloader():
    """Create a deterministic pair-wise dataloader."""
    pos = torch.linspace(0.2, 1.6, steps=16).view(-1, 1)
    neg = torch.linspace(-1.6, -0.2, steps=16).view(-1, 1)
    y = torch.ones(16)
    return DataLoader(DictDataset({"pos": pos, "neg": neg}, y), batch_size=4, shuffle=False)


def build_mtl_dataloader():
    """Create a deterministic two-task dataloader."""
    x = torch.linspace(-1.0, 1.0, steps=16).view(-1, 1)
    task_0 = (x > -0.1).float()
    task_1 = (x > 0.3).float()
    y = torch.cat([task_0, task_1], dim=1)
    return DataLoader(DictDataset({"x": x}, y), batch_size=4, shuffle=False)


def build_esmm_model():
    """Create a minimal ESMM instance for aggregation tests."""
    user_features = [SparseFeature("user", vocab_size=8, embed_dim=4)]
    item_features = [SparseFeature("item", vocab_size=8, embed_dim=4)]
    return ESMM(
        user_features,
        item_features,
        cvr_params={"dims": [4]},
        ctr_params={"dims": [4]},
    )


def binary_logloss_metrics(y_true, y_pred):
    """Return a scalar metric dict for binary classification."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    y_pred = np.clip(y_pred, 1e-6, 1 - 1e-6)
    logloss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)).mean()
    return {"logloss": float(logloss)}


def binary_multi_metrics(y_true, y_pred):
    """Return multiple binary metrics so callers can inspect monitor keys."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    y_pred = np.clip(y_pred, 1e-6, 1 - 1e-6)
    logloss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)).mean()
    mae = np.abs(y_true - y_pred).mean()
    return {"logloss": float(logloss), "mae": float(mae)}


def multitask_mae_metrics(targets, predicts):
    """Return one MAE metric per task."""
    targets = np.asarray(targets)
    predicts = np.asarray(predicts)
    mae = np.abs(targets - predicts).mean(axis=0)
    return {
        "task_0_mae": float(mae[0]),
        "task_1_mae": float(mae[1]),
    }


def multitask_partial_metrics(targets, predicts):
    """Return a metric dict for only one task."""
    targets = np.asarray(targets)
    predicts = np.asarray(predicts)
    mae = np.abs(targets[:, 1] - predicts[:, 1]).mean()
    return {"task_1_mae": float(mae)}


def multitask_regression_metrics(targets, predicts):
    """Return custom metrics including a maximize-style regression score."""
    targets = np.asarray(targets)
    predicts = np.asarray(predicts)
    residual = np.square(targets[:, 0] - predicts[:, 0]).mean()
    r2 = 1.0 - float(residual)
    mae = np.abs(targets[:, 1] - predicts[:, 1]).mean()
    return {
        "task_0_r2": r2,
        "task_1_mae": float(mae),
    }


def binary_logloss(y_true, y_pred):
    """Return scalar logloss for custom evaluator tests."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    y_pred = np.clip(y_pred, 1e-6, 1 - 1e-6)
    return float(-(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)).mean())


def multitask_scalar_mae(targets, predicts):
    """Return a single scalar MAE across all tasks."""
    targets = np.asarray(targets)
    predicts = np.asarray(predicts)
    return float(np.abs(targets - predicts).mean())
