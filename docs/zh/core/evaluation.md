---
title: 训练与评估
description: Torch-RecHub 模型训练与评估
---

# 训练与评估

Torch-RecHub提供了多种训练器，用于训练不同类型的推荐模型，包括排序模型、召回模型和多任务模型。所有训练器均提供统一的接口，包括训练、评估、预测、ONNX导出，以及可选的实验跟踪与可视化能力。

## 实验跟踪与可视化

- 支持 **WandB / SwanLab / TensorBoardX** 作为 `model_logger`，可传入单个实例或列表。
- 自动记录训练/验证指标与超参数：`train/loss`、`learning_rate`、`val/auc`（CTR/Match）、`val/task_i_auc` / `val/task_i_mse` / 自定义指标键（MTL）、`val/accuracy`（Seq）。
- 不需要记录时传 `model_logger=None`（默认）即可零开销。

```python
from torch_rechub.basic.tracking import WandbLogger, TensorBoardXLogger
from torch_rechub.trainers import CTRTrainer

wb = WandbLogger(project="rechub-demo", name="deepfm")
tb = TensorBoardXLogger(log_dir="./runs/deepfm")

trainer = CTRTrainer(model, model_logger=[wb, tb])
trainer.fit(train_dataloader, val_dataloader)
```

## 训练器

### CTRTrainer

用于训练排序模型（CTR预测模型），如DeepFM、Wide&Deep、DCN等。

```python
from torch_rechub.trainers import CTRTrainer
from torch_rechub.models.ranking import DeepFM

# 创建模型
model = DeepFM(deep_features=deep_features, fm_features=fm_features, mlp_params={"dims": [256, 128], "dropout": 0.2})

# 创建训练器
trainer = CTRTrainer(
    model=model,
    optimizer_params={"lr": 0.001, "weight_decay": 0.0001},
    n_epoch=50,
    earlystop_patience=10,
    device="cuda:0",
    model_path="saved/deepfm"
)

# 训练模型
trainer.fit(train_dataloader, val_dataloader)

# 评估模型
auc = trainer.evaluate(trainer.model, test_dataloader)

# 导出ONNX模型
trainer.export_onnx("deepfm.onnx")

# 可视化模型
trainer.visualization(save_path="deepfm_architecture.pdf")
```

**参数说明：**
- `model`：排序模型实例
- `optimizer_fn`：优化器函数，默认torch.optim.Adam
- `optimizer_params`：优化器参数
- `regularization_params`：正则化参数
- `scheduler_fn`：学习率调度器函数
- `scheduler_params`：学习率调度器参数
- `n_epoch`：训练轮数
- `earlystop_patience`：早停耐心值
- `device`：训练设备
- `gpus`：多GPU列表
- `loss_mode`：损失模式，布尔值。True表示模型只返回预测值，False表示模型返回预测值和额外损失
- `model_path`：模型保存路径
- `compute_loss_func`：可选的自定义损失函数，签名为 `compute_loss_func(model, x_dict, y)`
- `compute_metrics`：可选的自定义评估函数，签名为 `compute_metrics(y_true, y_pred)`，可返回单个 float 或 `dict[str, float]`
- `metric_for_best_model`：用于早停和最优模型选择的指标名
- `greater_is_better`：`metric_for_best_model` 是否“越大越好”

### MatchTrainer

用于训练召回模型，如DSSM、YoutubeDNN、MIND等。

```python
from torch_rechub.trainers import MatchTrainer
from torch_rechub.models.matching import DSSM

# 创建模型
model = DSSM(user_features=user_features, item_features=item_features, temperature=0.02,
             user_params={"dims": [256, 128, 64]}, item_params={"dims": [256, 128, 64]})

# 创建训练器
trainer = MatchTrainer(
    model=model,
    mode=0,  # 0: point-wise, 1: pair-wise, 2: list-wise
    optimizer_params={"lr": 0.001},
    n_epoch=50,
    device="cuda:0",
    model_path="saved/dssm"
)

# 训练模型
trainer.fit(train_dataloader)

# 导出用户塔ONNX模型
trainer.export_onnx("user_tower.onnx", mode="user")

# 导出物品塔ONNX模型
trainer.export_onnx("item_tower.onnx", mode="item")
```

**参数说明：**
- `model`：召回模型实例
- `mode`：训练模式，可选值：0（point-wise）、1（pair-wise）、2（list-wise）
- `optimizer_fn`：优化器函数，默认torch.optim.Adam
- `optimizer_params`：优化器参数
- `regularization_params`：正则化参数
- `scheduler_fn`：学习率调度器函数
- `scheduler_params`：学习率调度器参数
- `n_epoch`：训练轮数
- `earlystop_patience`：早停耐心值
- `device`：训练设备
- `gpus`：多GPU列表
- `model_path`：模型保存路径
- `compute_loss_func`：可选的自定义损失函数，签名为 `compute_loss_func(model, x_dict, y)`，可覆盖内置 point-wise / pair-wise / list-wise 损失
- `compute_metrics`：可选的自定义评估函数，签名为 `compute_metrics(y_true, y_pred)`，可返回单个 float 或 `dict[str, float]`
- `metric_for_best_model`：用于早停和最优模型选择的指标名
- `greater_is_better`：`metric_for_best_model` 是否“越大越好”

### MTLTrainer

用于训练多任务模型，如MMoE、PLE、ESMM、SharedBottom等。

```python
from torch_rechub.trainers import MTLTrainer
from torch_rechub.models.multi_task import MMOE

# 创建模型
model = MMOE(features=features, task_types=["classification", "classification"], n_expert=8,
             expert_params={"dims": [32,16]}, tower_params_list=[{"dims": [32, 16]}, {"dims": [32, 16]}])

# 创建训练器
trainer = MTLTrainer(
    model=model,
    task_types=["classification", "classification"],
    optimizer_params={"lr": 0.001},
    adaptive_params={"method": "uwl"},  # 自适应损失权重方法
    n_epoch=50,
    earlystop_taskid=0,  # 早停依赖的任务ID
    device="cuda:0",
    model_path="saved/mmoe"
)

# 训练模型
trainer.fit(train_dataloader, val_dataloader)

# 导出ONNX模型
trainer.export_onnx("mmoe.onnx")
```

**参数说明：**
- `model`：多任务模型实例
- `task_types`：任务类型列表，可选值：classification、regression
- `optimizer_fn`：优化器函数，默认torch.optim.Adam
- `optimizer_params`：优化器参数
- `regularization_params`：正则化参数
- `scheduler_fn`：学习率调度器函数
- `scheduler_params`：学习率调度器参数
- `adaptive_params`：自适应损失权重参数
- `n_epoch`：训练轮数
- `earlystop_taskid`：早停依赖的任务ID
- `earlystop_patience`：早停耐心值
- `device`：训练设备
- `gpus`：多GPU列表
- `model_path`：模型保存路径
- `compute_loss_func`：可选的多任务 loss hook，签名为 `compute_loss_func(model, x_dict, ys, y_preds)`
- `compute_metrics`：可选的自定义评估函数，签名为 `compute_metrics(targets, predicts)`，可返回单个 float 或 `dict[str, float]`
- `metric_for_best_model`：用于早停和最优模型选择的指标名
- `greater_is_better`：`metric_for_best_model` 是否“越大越好”

## 自定义损失函数与评估函数

现在各类 trainer 都支持显式的 hook 式扩展，使用方式和 Hugging Face `Trainer` 的自定义接口类似。

如果你想看完整说明，包括 hook 签名、monitor 规则、默认行为和常见踩坑，请直接看 [Trainer Hook 自定义](/zh/core/trainer_hooks) 页面。

### 排序 / CTR

```python
import numpy as np
import torch.nn.functional as F

def focal_loss(model, x_dict, y):
    y_pred = model(x_dict)
    bce = F.binary_cross_entropy(y_pred, y, reduction="none")
    pt = y * y_pred + (1 - y) * (1 - y_pred)
    return ((1 - pt) ** 2 * bce).mean()

def logloss_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.clip(np.asarray(y_pred).reshape(-1), 1e-6, 1 - 1e-6)
    logloss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)).mean()
    return {"logloss": float(logloss)}

trainer = CTRTrainer(
    model=model,
    compute_loss_func=focal_loss,
    compute_metrics=logloss_metrics,
    metric_for_best_model="logloss",
    greater_is_better=False,
)
```

### 召回 / Matching

```python
def pairwise_loss(model, x_dict, y):
    pos_score, neg_score = model(x_dict)
    return -(pos_score - neg_score).sigmoid().log().mean()

trainer = MatchTrainer(
    model=model,
    mode=1,
    compute_loss_func=pairwise_loss,
)
```

### 多任务

```python
import numpy as np
import torch.nn.functional as F

def task_losses(model, x_dict, ys, y_preds):
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
    compute_loss_func=task_losses,
    compute_metrics=multitask_metrics,
    metric_for_best_model="task_1_mae",
    greater_is_better=False,
)
```

注意：
- `compute_metrics` 可以返回单个 float，也可以返回指标字典。
- 当 `compute_metrics` 返回单个 float 且未显式传入 monitor 配置时，trainer 会将其记录为 `metric`，并默认按 `greater_is_better=False` 处理。
- 当返回多个指标时，`metric_for_best_model` 必须对应其中一个 key。
- 对于 `loss`、`logloss`、`mse`、`mae`、`rmse` 这类越小越好的指标，需要设置 `greater_is_better=False`。
- 当未提供 `compute_metrics` 时，`CTRTrainer` 和 `MatchTrainer` 的默认评估只会产出内置的 `auc`，因此不支持自定义 monitor 名称。
- 当未提供 `compute_metrics` 时，`MTLTrainer` 默认只产出内置的逐 task 指标，例如 `task_0_auc` 或 `task_1_mse`。

## 回调函数

### EarlyStopper

用于早停，当验证集性能不再提升时停止训练。

```python
from torch_rechub.basic.callback import EarlyStopper

# 创建早停器
early_stopper = EarlyStopper(patience=10, mode="max")

# 在训练过程中使用
if early_stopper.stop_training(auc, model.state_dict()):
    print(f'validation: best score: {early_stopper.best_score}')
    model.load_state_dict(early_stopper.best_weights)
    break
```

**参数说明：**
- `patience`：早停耐心值，即连续多少轮验证集性能没有提升就停止训练
- `mode`：`"max"` 表示指标越大越好，`"min"` 表示指标越小越好
- `delta`：性能提升阈值，即性能提升超过该值才被认为是有效提升

## 损失函数

### RegularizationLoss

用于正则化，支持L1和L2正则化。

```python
from torch_rechub.basic.loss_func import RegularizationLoss

# 创建正则化损失函数
reg_loss_fn = RegularizationLoss(
    embedding_l1=0.0,  # Embedding层L1正则化系数
    embedding_l2=0.0001,  # Embedding层L2正则化系数
    dense_l1=0.0,  # 密集层L1正则化系数
    dense_l2=0.0001  # 密集层L2正则化系数
)
```

### BPRLoss

用于召回模型的 pairwise 损失。

```python
from torch_rechub.basic.loss_func import BPRLoss

# 创建BPR损失函数
bpr_loss = BPRLoss()

# 计算损失
loss = bpr_loss(pos_score, neg_score)
```
