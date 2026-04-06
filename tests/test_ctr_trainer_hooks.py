import tempfile

import pytest
import torch

from torch_rechub.basic.callback import EarlyStopper
from torch_rechub.trainers import CTRTrainer

from tests.trainer_hook_cases import (
    BinaryModel,
    CountingBinaryLoss,
    LinearAccessBinaryLoss,
    binary_logloss,
    binary_logloss_metrics,
    binary_multi_metrics,
    build_ctr_dataloader,
)


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


def test_ctr_trainer_allows_multi_metric_inspection_before_monitor_selection():
    """CTR multi-metric hooks should be inspectable before choosing a monitor."""
    dataloader = build_ctr_dataloader()

    with tempfile.TemporaryDirectory() as temp_dir:
        trainer = CTRTrainer(
            model=BinaryModel(),
            optimizer_params={"lr": 0.05},
            n_epoch=1,
            device="cpu",
            model_path=temp_dir,
            compute_metrics=binary_multi_metrics,
        )

        metrics = trainer.evaluate(trainer.model, dataloader, return_dict=True)

        assert set(metrics) == {"logloss", "mae"}
        assert trainer.metric_for_best_model is None
        with pytest.raises(ValueError, match="Set metric_for_best_model to one of"):
            trainer.evaluate(trainer.model, dataloader, return_dict=False)


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


def test_ctr_trainer_unwraps_dataparallel_for_custom_loss(monkeypatch):
    """Custom CTR losses should receive the base model under DataParallel."""

    class FakeDataParallel(object):
        def __init__(self, module):
            self.module = module

    x_dict = {"x": torch.ones(4, 1)}
    y = torch.ones(4, 1)
    loss_hook = LinearAccessBinaryLoss()

    with tempfile.TemporaryDirectory() as temp_dir:
        trainer = CTRTrainer(
            model=BinaryModel(),
            optimizer_params={"lr": 0.05},
            n_epoch=1,
            device="cpu",
            model_path=temp_dir,
            compute_loss_func=loss_hook,
        )

        monkeypatch.setattr(torch.nn, "DataParallel", FakeDataParallel)
        trainer.model = torch.nn.DataParallel(trainer.model)

        loss = trainer._compute_batch_loss(x_dict, y)

        assert isinstance(loss, torch.Tensor)
        assert loss_hook.calls == 1
        assert loss_hook.model_type is BinaryModel
