import tempfile

import pytest
import torch

from torch_rechub.trainers import MTLTrainer

from tests.trainer_hook_cases import (
    CountingTaskLosses,
    ESMMTaskLosses,
    HeadAccessTaskLosses,
    MultiTaskBinaryModel,
    RecordingLogger,
    ShortTaskLosses,
    build_esmm_model,
    build_mtl_dataloader,
    multitask_mae_metrics,
    multitask_partial_metrics,
    multitask_regression_metrics,
    multitask_scalar_mae,
)


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


def test_mtl_trainer_unwraps_dataparallel_for_custom_loss(monkeypatch):
    """Custom MTL losses should receive the base model under DataParallel."""

    class FakeDataParallel(object):
        def __init__(self, module):
            self.module = module

    x_dict = {"x": torch.ones(4, 1)}
    ys = torch.ones(4, 2)
    y_preds = torch.full((4, 2), 0.5)
    loss_hook = HeadAccessTaskLosses()

    with tempfile.TemporaryDirectory() as temp_dir:
        trainer = MTLTrainer(
            model=MultiTaskBinaryModel(),
            task_types=["classification", "classification"],
            optimizer_params={"lr": 0.05},
            n_epoch=1,
            device="cpu",
            model_path=temp_dir,
            compute_loss_func=loss_hook,
        )

        monkeypatch.setattr(torch.nn, "DataParallel", FakeDataParallel)
        trainer.model = torch.nn.DataParallel(trainer.model)

        losses = trainer._compute_task_losses(x_dict, ys, y_preds)

        assert len(losses) == 2
        assert all(isinstance(loss, torch.Tensor) for loss in losses)
        assert loss_hook.calls == 1
        assert loss_hook.model_type is MultiTaskBinaryModel


def test_mtl_trainer_esmm_default_aggregation_keeps_legacy_objective():
    """Built-in ESMM loss aggregation should keep ignoring task 0."""
    with tempfile.TemporaryDirectory() as temp_dir:
        trainer = MTLTrainer(
            model=build_esmm_model(),
            task_types=["classification", "classification", "classification"],
            optimizer_params={"lr": 0.05},
            n_epoch=1,
            device="cpu",
            model_path=temp_dir,
        )

        loss = trainer._aggregate_loss(
            [
                torch.tensor(1.0),
                torch.tensor(2.0),
                torch.tensor(3.0),
            ]
        )

        assert loss.item() == pytest.approx(5.0)


def test_mtl_trainer_esmm_custom_loss_aggregation_uses_all_tasks():
    """Custom ESMM loss hooks should be able to override the full objective."""
    with tempfile.TemporaryDirectory() as temp_dir:
        trainer = MTLTrainer(
            model=build_esmm_model(),
            task_types=["classification", "classification", "classification"],
            optimizer_params={"lr": 0.05},
            n_epoch=1,
            device="cpu",
            model_path=temp_dir,
            compute_loss_func=ESMMTaskLosses(),
        )

        loss = trainer._aggregate_loss(
            [
                torch.tensor(1.0),
                torch.tensor(2.0),
                torch.tensor(3.0),
            ]
        )

        assert loss.item() == pytest.approx(2.0)


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


def test_mtl_trainer_allows_multi_metric_inspection_before_monitor_selection():
    """Multi-metric hooks should be inspectable before choosing a monitor."""
    dataloader = build_mtl_dataloader()

    with tempfile.TemporaryDirectory() as temp_dir:
        trainer = MTLTrainer(
            model=MultiTaskBinaryModel(),
            task_types=["classification", "classification"],
            optimizer_params={"lr": 0.05},
            n_epoch=1,
            device="cpu",
            model_path=temp_dir,
            compute_metrics=multitask_mae_metrics,
        )

        metrics = trainer.evaluate(trainer.model, dataloader, return_dict=True)

        assert set(metrics) == {"task_0_mae", "task_1_mae"}
        assert trainer.metric_for_best_model is None
        with pytest.raises(ValueError, match="Set metric_for_best_model to one of"):
            trainer.evaluate(trainer.model, dataloader, return_dict=False)


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


def test_mtl_trainer_infers_regression_monitor_direction_from_metric_name():
    """Regression custom metrics should still infer maximize-style names correctly."""
    with tempfile.TemporaryDirectory() as temp_dir:
        trainer = MTLTrainer(
            model=MultiTaskBinaryModel(),
            task_types=["regression", "classification"],
            optimizer_params={"lr": 0.05},
            n_epoch=1,
            earlystop_taskid=0,
            device="cpu",
            model_path=temp_dir,
            compute_metrics=multitask_regression_metrics,
            metric_for_best_model="task_0_r2",
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
