import math
import tempfile

import numpy as np
import torch
import torch.nn as nn

from torch_rechub.basic.features import SparseFeature
from torch_rechub.models.multi_task import ESMM
from torch_rechub.trainers import MTLTrainer


def build_esmm():
    user_features = [SparseFeature("user_id", vocab_size=8, embed_dim=4)]
    item_features = [SparseFeature("item_id", vocab_size=8, embed_dim=4)]
    return ESMM(user_features, item_features, cvr_params={"dims": [4]}, ctr_params={"dims": [4]})


def test_esmm_task_hooks_mask_cvr_samples():
    model = build_esmm()
    loss_fns = [torch.nn.BCELoss(), torch.nn.BCELoss(), torch.nn.BCELoss()]

    predicts = torch.tensor([
        [0.90, 0.80, 0.72],
        [0.20, 0.70, 0.14],
        [0.60, 0.10, 0.06],
    ])
    targets = torch.tensor([
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ])

    task_losses = model.compute_task_losses(predicts, targets, default_loss_fns=loss_fns)
    expected_cvr = loss_fns[0](predicts[:2, 0], targets[:2, 0])
    expected_ctr = loss_fns[1](predicts[:, 1], targets[:, 1])
    expected_ctcvr = loss_fns[2](predicts[:, 2], targets[:, 2])

    assert torch.isclose(task_losses[0], expected_cvr)
    assert torch.isclose(task_losses[1], expected_ctr)
    assert torch.isclose(task_losses[2], expected_ctcvr)
    assert torch.isclose(model.aggregate_task_losses(task_losses), expected_ctr + expected_ctcvr)

    scores = model.compute_task_metrics(targets.numpy(), predicts.numpy())
    assert not math.isnan(scores[0])
    assert len(scores) == 3


class DummyHookedMTLModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.features = []
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, x_dict):
        base = x_dict["feat"].float().unsqueeze(1) * self.weight
        return torch.cat([base, base + 0.1], dim=1)

    def compute_task_losses(self, predicts, targets, default_loss_fns):
        return [
            ((predicts[:, 0] - targets[:, 0]) ** 2).mean(),
            ((predicts[:, 1] - targets[:, 1]) ** 2).mean(),
        ]

    def aggregate_task_losses(self, loss_list):
        return loss_list[0] + 2 * loss_list[1]

    def compute_task_metrics(self, targets, predicts, default_metric_fns=None):
        return [
            float(np.mean(targets[:, 0] - predicts[:, 0])),
            float(np.mean(targets[:, 1] - predicts[:, 1])),
        ]


def test_mtl_trainer_prefers_model_hooks_and_allows_overrides():
    model = DummyHookedMTLModel()
    batch = [(
        {"feat": torch.tensor([0.2, 0.8])},
        torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
    )]

    with tempfile.TemporaryDirectory() as temp_dir:
        trainer = MTLTrainer(
            model,
            task_types=["classification", "classification"],
            optimizer_params={"lr": 0.01},
            n_epoch=1,
            device="cpu",
            model_path=temp_dir,
        )
        task_losses = trainer._compute_task_losses(model(batch[0][0]), batch[0][1])
        expected_losses = model.compute_task_losses(model(batch[0][0]), batch[0][1].float(), trainer.loss_fns)
        assert all(torch.isclose(lhs, rhs) for lhs, rhs in zip(task_losses, expected_losses))
        assert torch.isclose(trainer._aggregate_task_losses(task_losses), expected_losses[0] + 2 * expected_losses[1])

        scores = trainer.evaluate(model, batch)
        expected_scores = model.compute_task_metrics(batch[0][1].numpy(), model(batch[0][0]).detach().numpy())
        assert scores == expected_scores

        override_trainer = MTLTrainer(
            model,
            task_types=["classification", "classification"],
            optimizer_params={"lr": 0.01},
            n_epoch=1,
            device="cpu",
            model_path=temp_dir,
            custom_loss_funcs=[lambda predicts, targets: torch.tensor(5.0)],
            custom_evaluate_funcs=[lambda targets, predicts: 7.0],
        )
        override_losses = override_trainer._compute_task_losses(model(batch[0][0]), batch[0][1])
        assert torch.isclose(override_losses[0], torch.tensor(5.0))

        override_scores = override_trainer.evaluate(model, batch)
        assert override_scores[0] == 7.0
