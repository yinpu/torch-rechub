import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_rechub.trainers import MTLTrainer


class DummyMTLModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.features = []
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, x_dict):
        base = x_dict["feat"].float().unsqueeze(1) * self.weight
        return torch.cat([base, base + 0.1], dim=1)


def conditional_cvr_loss(preds, targets, task_id):
    assert task_id == 0
    click_mask = targets[:, 1] == 1
    if not click_mask.any():
        return preds[:, task_id].new_tensor(0.0)
    return F.binary_cross_entropy(preds[click_mask, task_id], targets[click_mask, task_id])


def conditional_cvr_gap(targets, predicts):
    click_mask = targets[:, 1] == 1
    return float(np.mean(targets[click_mask, 0] - predicts[click_mask, 0]))


def aggregate_losses(loss_list, preds, targets):
    assert preds.shape[1] == targets.shape[1]
    return loss_list[0] + 2 * loss_list[1]


def test_mtl_trainer_supports_task_aware_custom_hooks():
    model = DummyMTLModel()
    batch = [(
        {"feat": torch.tensor([0.2, 0.8])},
        torch.tensor([[0.0, 1.0], [1.0, 0.0]]),
    )]
    preds = model(batch[0][0])
    targets = batch[0][1].float()

    with tempfile.TemporaryDirectory() as temp_dir:
        trainer = MTLTrainer(
            model,
            task_types=["classification", "classification"],
            optimizer_params={"lr": 0.01},
            n_epoch=1,
            device="cpu",
            model_path=temp_dir,
            custom_loss_funcs=[conditional_cvr_loss],
            custom_evaluate_funcs=[conditional_cvr_gap],
            loss_aggregate_fn=aggregate_losses,
        )
        task_losses = trainer._compute_task_losses(preds, targets)
        expected_loss0 = conditional_cvr_loss(preds, targets, 0)
        expected_loss1 = trainer.loss_fns[1](preds[:, 1], targets[:, 1])
        assert torch.isclose(task_losses[0], expected_loss0)
        assert torch.isclose(task_losses[1], expected_loss1)

        scores = trainer.evaluate(model, batch)
        expected_score0 = conditional_cvr_gap(targets.numpy(), preds.detach().numpy())
        expected_score1 = trainer.evaluate_fns[1](targets.numpy()[:, 1], preds.detach().numpy()[:, 1])
        assert scores[0] == expected_score0
        assert scores[1] == expected_score1
        assert torch.isclose(trainer._aggregate_task_losses(task_losses, preds, targets), expected_loss0 + 2 * expected_loss1)

        override_trainer = MTLTrainer(
            model,
            task_types=["classification", "classification"],
            optimizer_params={"lr": 0.01},
            n_epoch=1,
            device="cpu",
            model_path=temp_dir,
            custom_loss_funcs=[lambda predicts, targets: torch.tensor(5.0)],
            custom_evaluate_funcs=[lambda targets, predicts, task_id: 7.0 + task_id],
        )
        override_losses = override_trainer._compute_task_losses(preds, targets)
        assert torch.isclose(override_losses[0], torch.tensor(5.0))

        override_scores = override_trainer.evaluate(model, batch)
        assert override_scores[0] == 7.0
        assert override_scores[1] == trainer.evaluate_fns[1](targets.numpy()[:, 1], preds.detach().numpy()[:, 1])
