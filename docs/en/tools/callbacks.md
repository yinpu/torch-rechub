---
title: Callbacks
description: Torch-RecHub training callbacks
---

# Callbacks

Callbacks are tools that perform specific operations during training, used to implement early stopping, model saving, learning rate adjustment, and more. Torch-RecHub provides a simple and easy-to-use callback interface.

## EarlyStopper

EarlyStopper is an early stopping utility that stops training when a monitored validation score stops improving.

### Features

- Monitor validation metrics such as AUC, logloss, MSE, or MAE
- Trigger early stopping when metrics don't improve for consecutive epochs
- Automatically save best model weights

### Usage

```python
from torch_rechub.basic.callback import EarlyStopper

# Create early stopper for a metric where larger is better
early_stopper = EarlyStopper(patience=10, mode="max")

# Use in training loop
for epoch in range(n_epoch):
    # Train one epoch
    train_one_epoch(model, train_dataloader)

    # Validate
    val_auc = evaluate(model, val_dataloader)

    # Check if early stopping is needed
    if early_stopper.stop_training(val_auc, model.state_dict()):
        print(f'Early stopping at epoch {epoch}')
        print(f'Best validation score: {early_stopper.best_score}')
        # Restore best weights
        model.load_state_dict(early_stopper.best_weights)
        break
```

### Parameters

| Parameter | Type | Description | Default |
| --- | --- | --- | --- |
| `patience` | int | Early stopping patience, i.e., how many consecutive epochs without improvement before stopping | Required |
| `mode` | str | `"max"` when larger is better, `"min"` when smaller is better | `"max"` |
| `delta` | float | Minimum improvement required to reset the patience counter | `0.0` |

### Attributes

| Attribute | Type | Description |
| --- | --- | --- |
| `best_score` | float | Best recorded validation score |
| `best_auc` | float | Backward-compatible alias of the best score |
| `best_weights` | dict | Deep copy of best model weights |
| `trial_counter` | int | Current count of consecutive epochs without improvement |

### Methods

#### stop_training(score, weights)

Determine whether to stop training.

**Parameters:**
- `score` (float): Current validation score in the monitored direction
- `weights` (dict): Current model weights (`model.state_dict()`)

**Returns:**
- `bool`: Returns `True` if training should stop, otherwise `False`

## Using with Trainer

Torch-RecHub trainers have built-in early stopping functionality, controlled via the `earlystop_patience` parameter:

```python
from torch_rechub.trainers import CTRTrainer

trainer = CTRTrainer(
    model=model,
    optimizer_params={"lr": 0.001},
    n_epoch=50,
    earlystop_patience=10,  # Early stopping patience
    device="cuda:0",
    model_path="saved/model"
)

trainer.fit(train_dataloader, val_dataloader)
```

## Complete Example

```python
import torch
from torch_rechub.models.ranking import DeepFM
from torch_rechub.trainers import CTRTrainer
from torch_rechub.basic.callback import EarlyStopper

# Create model
model = DeepFM(
    deep_features=deep_features,
    fm_features=fm_features,
    mlp_params={"dims": [256, 128], "dropout": 0.2}
)

# Method 1: Use Trainer's built-in early stopping
trainer = CTRTrainer(
    model=model,
    optimizer_params={"lr": 0.001, "weight_decay": 1e-5},
    n_epoch=50,
    earlystop_patience=10,
    device="cuda:0"
)
trainer.fit(train_dl, val_dl)

# Method 2: Manual EarlyStopper usage
early_stopper = EarlyStopper(patience=10, mode="max")
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(50):
    model.train()
    for batch in train_dl:
        # Training step
        optimizer.zero_grad()
        loss = model(batch)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    val_auc = evaluate(model, val_dl)
    print(f"Epoch {epoch}, Val AUC: {val_auc:.4f}")

    # Early stopping check
    if early_stopper.stop_training(val_auc, model.state_dict()):
        print(f"Early stopping! Best score: {early_stopper.best_score:.4f}")
        model.load_state_dict(early_stopper.best_weights)
        break
```

For metrics where smaller is better, switch to `mode="min"`:

```python
early_stopper = EarlyStopper(patience=10, mode="min")
```

## Best Practices

1. **Choose appropriate patience value**:
   - Too small may cause premature stopping, missing better results
   - Too large may waste training time
   - Recommend starting with 5-10

2. **Combine with learning rate scheduling**:
   - Try reducing learning rate before early stopping
   - Use `scheduler_fn` and `scheduler_params` to configure learning rate scheduler

3. **Save checkpoints**:
   - Early stopper automatically saves best weights
   - Also recommend using `model_path` parameter to save model to disk
