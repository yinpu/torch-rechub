import tempfile

import pytest
import torch

from torch_rechub.trainers import MatchTrainer

from tests.trainer_hook_cases import (
    BinaryModel,
    CountingBinaryLoss,
    CountingPairwiseLoss,
    PairwiseModel,
    TowerAccessLoss,
    TwoTowerMatchModel,
    binary_logloss,
    binary_multi_metrics,
    build_ctr_dataloader,
    build_ctr_int_label_dataloader,
    build_pairwise_dataloader,
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


def test_match_trainer_unwraps_dataparallel_for_custom_loss(monkeypatch):
    """Custom match losses should receive the base model under DataParallel."""

    class FakeDataParallel(object):
        def __init__(self, module):
            self.module = module

    x_dict = {
        "user": torch.ones(4, 1),
        "item": torch.ones(4, 1),
    }
    loss_hook = TowerAccessLoss()

    with tempfile.TemporaryDirectory() as temp_dir:
        trainer = MatchTrainer(
            model=TwoTowerMatchModel(),
            mode=0,
            optimizer_params={"lr": 0.05},
            n_epoch=1,
            device="cpu",
            model_path=temp_dir,
            compute_loss_func=loss_hook,
        )

        monkeypatch.setattr(torch.nn, "DataParallel", FakeDataParallel)
        trainer.model = torch.nn.DataParallel(trainer.model)

        loss = trainer._compute_batch_loss(x_dict, torch.ones(4))

        assert isinstance(loss, torch.Tensor)
        assert loss_hook.calls == 1
        assert loss_hook.model_type is TwoTowerMatchModel


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


def test_match_trainer_allows_multi_metric_inspection_before_monitor_selection():
    """Match multi-metric hooks should be inspectable before choosing a monitor."""
    dataloader = build_ctr_dataloader()

    with tempfile.TemporaryDirectory() as temp_dir:
        trainer = MatchTrainer(
            model=BinaryModel(),
            mode=0,
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
