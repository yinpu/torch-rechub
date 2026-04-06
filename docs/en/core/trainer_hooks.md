---
title: Trainer Hook Customization
description: Customize trainer losses, metrics, and monitor rules in Torch-RecHub
---

# Trainer Hook Customization

Compared with `main`, this branch does not mainly add new models. It makes `CTRTrainer`, `MatchTrainer`, and `MTLTrainer` configurable enough to plug in custom loss functions, custom evaluation metrics, and custom early-stopping monitors.

This is the right page if you want to:

- keep the built-in training loop but swap the task loss
- monitor `logloss`, `mae`, `rmse`, or business metrics instead of the default metric
- choose the best checkpoint with a metric that should be minimized
- customize per-task losses and monitor logic in `MTLTrainer`

## What this branch adds

### Single-task custom loss hooks

`CTRTrainer` and `MatchTrainer` now support:

- `compute_loss_func`

Hook signature:

```python
compute_loss_func(model, x_dict, y) -> torch.Tensor
```

For `CTRTrainer`, this hook is only guaranteed to be safe when
`loss_mode=True`. If `loss_mode=False`, the model returns
`(y_pred, auxiliary_loss)`, so custom hooks must unpack the tuple and add the
auxiliary term themselves.

### Custom metric hooks for all main trainers

All three trainers now support:

- `compute_metrics`
- `metric_for_best_model`
- `greater_is_better`

`compute_metrics` may return:

- a single `float`
- a `dict[str, float]`

### Custom task-loss hooks for MTLTrainer

`MTLTrainer` now supports:

- `compute_loss_func`

Main hook signature:

```python
compute_loss_func(model, x_dict, ys, y_preds) -> list[torch.Tensor]
```

The returned loss list must match the number of tasks.

### EarlyStopper can minimize metrics

`EarlyStopper` now supports:

- `mode="max"` for metrics such as AUC
- `mode="min"` for metrics such as loss, logloss, MSE, MAE, and RMSE
- `delta` for minimum required improvement

## Why this page belongs in `core/`

This feature extends trainer behavior directly. It is not a standalone utility and not a scenario tutorial, so the best place is:

- page: `docs/en/core/trainer_hooks.md`
- navigation: next to `Training & Eval`

That placement keeps it close to the trainer API where users will look first.

## Quick usage

### CTRTrainer

```python
import numpy as np
import torch.nn.functional as F

from torch_rechub.trainers import CTRTrainer


def focal_loss(model, x_dict, y):
    y_pred = model(x_dict)
    bce = F.binary_cross_entropy(y_pred, y, reduction="none")
    pt = y * y_pred + (1 - y) * (1 - y_pred)
    return ((1 - pt) ** 2 * bce).mean()


def binary_logloss(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    y_pred = np.clip(y_pred, 1e-6, 1 - 1e-6)
    return {"logloss": float(-(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)).mean())}


trainer = CTRTrainer(
    model=model,
    compute_loss_func=focal_loss,
    compute_metrics=binary_logloss,
    metric_for_best_model="logloss",
    greater_is_better=False,
)
```

### MatchTrainer

```python
from torch_rechub.trainers import MatchTrainer


def pairwise_logsigmoid_loss(model, x_dict, y):
    del y
    pos_score, neg_score = model(x_dict)
    return -(pos_score - neg_score).sigmoid().log().mean()


trainer = MatchTrainer(
    model=model,
    mode=1,
    compute_loss_func=pairwise_logsigmoid_loss,
)
```

### MTLTrainer

```python
import numpy as np
import torch.nn.functional as F

from torch_rechub.trainers import MTLTrainer


def custom_task_losses(model, x_dict, ys, y_preds):
    del model, x_dict
    return [
        F.binary_cross_entropy(y_preds[:, 0], ys[:, 0].float()),
        F.binary_cross_entropy(y_preds[:, 1], ys[:, 1].float()),
    ]


def multitask_metrics(targets, predicts):
    targets = np.asarray(targets)
    predicts = np.asarray(predicts)
    mae = np.abs(targets - predicts).mean(axis=0)
    return {
        "task_0_mae": float(mae[0]),
        "task_1_mae": float(mae[1]),
    }


trainer = MTLTrainer(
    model=model,
    task_types=["classification", "classification"],
    compute_loss_func=custom_task_losses,
    compute_metrics=multitask_metrics,
    metric_for_best_model="task_1_mae",
    greater_is_better=False,
)
```

## Monitor behavior

### Without `compute_metrics`

`CTRTrainer` and `MatchTrainer` still expose only the built-in `auc`. If you want to monitor `logloss`, `mae`, or any other metric, you must provide `compute_metrics`.

`MTLTrainer` still uses the built-in per-task evaluators and monitors `task_{earlystop_taskid}_{default_metric}` by default.

### When `compute_metrics` returns a single float

Trainers normalize it to:

```python
{"metric": value}
```

and default to:

- `metric_for_best_model="metric"`
- `greater_is_better=False`

### When `compute_metrics` returns multiple metrics

You should explicitly set `metric_for_best_model`, for example:

```python
metric_for_best_model="logloss"
greater_is_better=False
```

For `MTLTrainer`, custom metric dict keys are logged directly. `task_{id}_...` naming is optional.

### Direction inference

If you omit `greater_is_better`, the trainers infer it from the metric name:

- names containing `loss`, `logloss`, `mse`, `mae`, `rmse`, or `error` default to minimize
- other names default to maximize

This is only a heuristic. For business-specific metrics, set the direction explicitly.

## Constraints and common mistakes

- `compute_loss_func` must return one loss per configured task.
- If `metric_for_best_model` does not exist in the returned metric dict, the trainer raises an error.
- For experimental metrics, inspect `trainer.evaluate(..., return_dict=True)` first and then configure the monitor key.

## Summary versus `main`

Relative to `main`, this branch turns the trainers from mostly fixed training wrappers into reusable training skeletons:

- custom loss injection without rewriting the loop
- custom validation metrics and best-model selection
- proper support for minimize-style monitors
- better multi-task customization
