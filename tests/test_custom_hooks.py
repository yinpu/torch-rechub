import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from torch_rechub.basic.callback import EarlyStopper
from torch_rechub.trainers import CTRTrainer, MTLTrainer, MatchTrainer


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


class ShortTaskLosses(object):
    """Invalid MTL loss hook used to verify return-shape validation."""

    def __call__(self, model, x_dict, ys, y_preds):
        del model, x_dict, ys
        return [F.binary_cross_entropy(y_preds[:, 0], y_preds[:, 0].detach())]


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


def binary_logloss_metrics(y_true, y_pred):
    """Return a scalar metric dict for binary classification."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    y_pred = np.clip(y_pred, 1e-6, 1 - 1e-6)
    logloss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)).mean()
    return {"logloss": float(logloss)}


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


def test_early_stopper_supports_min_mode():
    """EarlyStopper should support metrics where smaller values are better."""
    stopper = EarlyStopper(patience=2, mode="min")
    weights = {"w": torch.tensor([1.0])}

    assert stopper.stop_training(0.5, weights) is False
    assert stopper.best_score == 0.5
    assert stopper.stop_training(0.6, weights) is False
    assert stopper.stop_training(0.7, weights) is True


def test_ctr_trainer_supports_custom_loss_and_metrics():
    """CTRTrainer should call custom hooks and expose dict metrics."""
    dataloader = build_ctr_dataloader()
    loss_hook = CountingBinaryLoss()

    with tempfile.TemporaryDirectory() as temp_dir:
        trainer = CTRTrainer(
            model=BinaryModel(),
            optimizer_params={"lr": 0.05},
            n_epoch=1,
            device="cpu",
            model_path=temp_dir,
            compute_loss_func=loss_hook,
            compute_metrics=binary_logloss_metrics,
            metric_for_best_model="logloss",
            greater_is_better=False,
        )

        trainer.train_one_epoch(dataloader)
        metrics = trainer.evaluate(trainer.model, dataloader, return_dict=True)
        scalar_score = trainer.evaluate(trainer.model, dataloader, return_dict=False)

        assert loss_hook.calls > 0
        assert "logloss" in metrics
        assert isinstance(metrics["logloss"], float)
        assert isinstance(scalar_score, float)
        assert trainer.early_stopper.mode == "min"


def test_ctr_trainer_default_evaluate_remains_scalar():
    """Default CTR evaluation should remain backward-compatible."""
    dataloader = build_ctr_dataloader()

    with tempfile.TemporaryDirectory() as temp_dir:
        trainer = CTRTrainer(
            model=BinaryModel(),
            optimizer_params={"lr": 0.05},
            n_epoch=1,
            device="cpu",
            model_path=temp_dir,
        )

        score = trainer.evaluate(trainer.model, dataloader)

        assert isinstance(score, float)


def test_ctr_trainer_scalar_custom_metric_defaults_to_min_monitor():
    """Scalar custom metrics should no longer inherit the built-in AUC monitor."""
    dataloader = build_ctr_dataloader()

    with tempfile.TemporaryDirectory() as temp_dir:
        trainer = CTRTrainer(
            model=BinaryModel(),
            optimizer_params={"lr": 0.05},
            n_epoch=1,
            device="cpu",
            model_path=temp_dir,
            compute_metrics=binary_logloss,
        )

        metrics = trainer.evaluate(trainer.model, dataloader, return_dict=True)

        assert set(metrics) == {"metric"}
        assert trainer.metric_for_best_model == "metric"
        assert trainer.greater_is_better is False
        assert trainer.early_stopper.mode == "min"


def test_ctr_trainer_rejects_custom_monitor_without_custom_metrics():
    """Default CTR evaluator should not relabel AUC as another metric."""
    with tempfile.TemporaryDirectory() as temp_dir:
        with pytest.raises(ValueError, match="only returns 'auc'"):
            CTRTrainer(
                model=BinaryModel(),
                optimizer_params={"lr": 0.05},
                n_epoch=1,
                device="cpu",
                model_path=temp_dir,
                metric_for_best_model="logloss",
                greater_is_better=False,
            )


def test_match_trainer_supports_custom_pairwise_loss():
    """MatchTrainer should honor custom pair-wise loss hooks."""
    dataloader = build_pairwise_dataloader()
    loss_hook = CountingPairwiseLoss()

    with tempfile.TemporaryDirectory() as temp_dir:
        trainer = MatchTrainer(
            model=PairwiseModel(),
            mode=1,
            optimizer_params={"lr": 0.05},
            n_epoch=1,
            device="cpu",
            model_path=temp_dir,
            compute_loss_func=loss_hook,
        )

        trainer.train_one_epoch(dataloader)

        assert loss_hook.calls > 0


def test_match_trainer_prepares_targets_for_custom_pointwise_loss():
    """Custom point-wise losses should receive float labels like the built-in BCE path."""
    dataloader = build_ctr_int_label_dataloader()
    loss_hook = CountingBinaryLoss()

    with tempfile.TemporaryDirectory() as temp_dir:
        trainer = MatchTrainer(
            model=BinaryModel(),
            mode=0,
            optimizer_params={"lr": 0.05},
            n_epoch=1,
            device="cpu",
            model_path=temp_dir,
            compute_loss_func=loss_hook,
        )

        trainer.train_one_epoch(dataloader)

        assert loss_hook.calls > 0
        assert loss_hook.last_dtype == torch.float32


def test_match_trainer_rejects_custom_monitor_without_custom_metrics():
    """Default matching evaluator should not relabel AUC as another metric."""
    with tempfile.TemporaryDirectory() as temp_dir:
        with pytest.raises(ValueError, match="only returns 'auc'"):
            MatchTrainer(
                model=BinaryModel(),
                optimizer_params={"lr": 0.05},
                n_epoch=1,
                device="cpu",
                model_path=temp_dir,
                metric_for_best_model="logloss",
                greater_is_better=False,
            )


def test_match_trainer_scalar_custom_metric_defaults_to_min_monitor():
    """Scalar custom metrics should not inherit the built-in AUC monitor."""
    dataloader = build_ctr_dataloader()

    with tempfile.TemporaryDirectory() as temp_dir:
        trainer = MatchTrainer(
            model=BinaryModel(),
            mode=0,
            optimizer_params={"lr": 0.05},
            n_epoch=1,
            device="cpu",
            model_path=temp_dir,
            compute_metrics=binary_logloss,
        )

        metrics = trainer.evaluate(trainer.model, dataloader, return_dict=True)

        assert set(metrics) == {"metric"}
        assert trainer.metric_for_best_model == "metric"
        assert trainer.greater_is_better is False
        assert trainer.early_stopper.mode == "min"


def test_mtl_trainer_supports_custom_losses_and_metrics():
    """MTLTrainer should accept custom per-task losses and metric dicts."""
    dataloader = build_mtl_dataloader()
    loss_hook = CountingTaskLosses()

    with tempfile.TemporaryDirectory() as temp_dir:
        trainer = MTLTrainer(
            model=MultiTaskBinaryModel(),
            task_types=["classification", "classification"],
            optimizer_params={"lr": 0.05},
            n_epoch=1,
            device="cpu",
            model_path=temp_dir,
            compute_loss_func=loss_hook,
            compute_metrics=multitask_mae_metrics,
            metric_for_best_model="task_1_mae",
            greater_is_better=False,
        )

        trainer.fit(dataloader, dataloader)
        default_metrics = trainer.evaluate(trainer.model, dataloader)
        metrics = trainer.evaluate(trainer.model, dataloader, return_dict=True)
        scalar_score = trainer.evaluate(trainer.model, dataloader, return_dict=False)

        assert loss_hook.calls > 0
        assert default_metrics == metrics
        assert "task_0_mae" in metrics
        assert "task_1_mae" in metrics
        assert isinstance(metrics["task_1_mae"], float)
        assert isinstance(scalar_score, float)
        assert trainer.early_stopper.mode == "min"


def test_mtl_trainer_rejects_custom_monitor_without_custom_metrics():
    """Default MTL evaluator should not pretend to expose custom metric keys."""
    with tempfile.TemporaryDirectory() as temp_dir:
        with pytest.raises(ValueError, match="default evaluation only returns"):
            MTLTrainer(
                model=MultiTaskBinaryModel(),
                task_types=["classification", "classification"],
                optimizer_params={"lr": 0.05},
                n_epoch=1,
                device="cpu",
                model_path=temp_dir,
                metric_for_best_model="task_0_logloss",
            )


def test_mtl_trainer_validates_custom_task_loss_count():
    """Custom MTL loss hooks must return one loss per configured task."""
    dataloader = build_mtl_dataloader()

    with tempfile.TemporaryDirectory() as temp_dir:
        trainer = MTLTrainer(
            model=MultiTaskBinaryModel(),
            task_types=["classification", "classification"],
            optimizer_params={"lr": 0.05},
            n_epoch=1,
            device="cpu",
            model_path=temp_dir,
            compute_loss_func=ShortTaskLosses(),
        )

        with pytest.raises(ValueError, match="must return 2 losses"):
            trainer.train_one_epoch(dataloader)


def test_mtl_trainer_default_evaluate_returns_legacy_task_scores():
    """Built-in MTL evaluation should keep returning per-task scores by default."""
    dataloader = build_mtl_dataloader()

    with tempfile.TemporaryDirectory() as temp_dir:
        trainer = MTLTrainer(
            model=MultiTaskBinaryModel(),
            task_types=["classification", "classification"],
            optimizer_params={"lr": 0.05},
            n_epoch=1,
            device="cpu",
            model_path=temp_dir,
        )

        default_scores = trainer.evaluate(trainer.model, dataloader)
        metrics = trainer.evaluate(trainer.model, dataloader, return_dict=True)
        scalar_score = trainer.evaluate(trainer.model, dataloader, return_dict=False)

        assert trainer.metric_for_best_model == "task_0_auc"
        assert trainer.early_stopper.mode == "max"
        assert isinstance(default_scores, list)
        assert len(default_scores) == 2
        assert default_scores == [metrics["task_0_auc"], metrics["task_1_auc"]]
        assert isinstance(scalar_score, float)
        assert scalar_score == metrics["task_0_auc"]


def test_mtl_trainer_scalar_custom_metric_defaults_to_min_monitor():
    """Scalar MTL metrics should not inherit task-default AUC-style monitoring."""
    dataloader = build_mtl_dataloader()

    with tempfile.TemporaryDirectory() as temp_dir:
        trainer = MTLTrainer(
            model=MultiTaskBinaryModel(),
            task_types=["classification", "classification"],
            optimizer_params={"lr": 0.05},
            n_epoch=1,
            device="cpu",
            model_path=temp_dir,
            compute_metrics=multitask_scalar_mae,
        )

        metrics = trainer.evaluate(trainer.model, dataloader, return_dict=True)

        assert set(metrics) == {"metric"}
        assert trainer.metric_for_best_model == "metric"
        assert trainer.greater_is_better is False
        assert trainer.early_stopper.mode == "min"


def test_mtl_trainer_infers_monitor_direction_from_monitored_task():
    """Monitor direction should follow the task named in metric_for_best_model."""
    with tempfile.TemporaryDirectory() as temp_dir:
        trainer = MTLTrainer(
            model=MultiTaskBinaryModel(),
            task_types=["regression", "classification"],
            optimizer_params={"lr": 0.05},
            n_epoch=1,
            earlystop_taskid=0,
            device="cpu",
            model_path=temp_dir,
            metric_for_best_model="task_1_auc",
        )

        assert trainer.greater_is_better is True
        assert trainer.early_stopper.mode == "max"


def test_mtl_trainer_fit_returns_structured_metric_logs():
    """MTLTrainer.fit should return per-epoch log dicts without task-slot padding."""
    dataloader = build_mtl_dataloader()
    logger = RecordingLogger()

    with tempfile.TemporaryDirectory() as temp_dir:
        trainer = MTLTrainer(
            model=MultiTaskBinaryModel(),
            task_types=["classification", "classification"],
            optimizer_params={"lr": 0.05},
            n_epoch=1,
            device="cpu",
            model_path=temp_dir,
            compute_metrics=multitask_partial_metrics,
            metric_for_best_model="task_1_mae",
            greater_is_better=False,
            model_logger=logger,
        )

        total_log = trainer.fit(dataloader, dataloader)
        metric_logs = [payload for kind, _, payload in logger.history if kind == "metrics"]

        assert isinstance(total_log[0], dict)
        assert "train/task_0_loss" in total_log[0]
        assert "train/task_1_loss" in total_log[0]
        assert "val/task_1_mae" in total_log[0]
        assert "val/task_0_score" not in total_log[0]
        assert "val/task_1_score" not in total_log[0]
        assert total_log[0]["val/task_1_mae"] == metric_logs[-1]["val/task_1_mae"]
