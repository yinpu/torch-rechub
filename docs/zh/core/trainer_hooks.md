---
title: Trainer Hook 自定义
description: Torch-RecHub 训练器自定义损失、指标与监控策略
---

# Trainer Hook 自定义

这个分支相对 `main` 的核心增强，不是新增模型，而是让 `CTRTrainer`、`MatchTrainer`、`MTLTrainer` 可以像通用训练框架一样，显式接入你自己的 loss、metric 和 early stopping 监控逻辑。

如果你之前遇到下面这些问题，这一页就是对应的使用文档：

- 想保留 Torch-RecHub 的训练循环，但把默认 BCE / BPR / 多任务 loss 换成自己的实现
- 想在验证集上监控 `logloss`、`mae`、`rmse` 等非默认指标
- 多任务场景下想自己决定每个 task 的 loss 怎么算、监控哪个 task 的哪个指标
- 想让 early stopping 支持“越小越好”的指标，而不再默认假设所有指标都像 AUC 一样越大越好

## 这次新增了什么

### 1. 单任务 Trainer 支持自定义 loss hook

`CTRTrainer` 和 `MatchTrainer` 新增了：

- `compute_loss_func`

签名统一为：

```python
compute_loss_func(model, x_dict, y) -> torch.Tensor
```

你可以直接复用 Trainer 的训练循环、优化器、正则化、scheduler 和 logger，只替换任务损失本身。

### 2. 所有主训练器支持自定义 metric hook

`CTRTrainer`、`MatchTrainer`、`MTLTrainer` 新增了：

- `compute_metrics`
- `metric_for_best_model`
- `greater_is_better`

其中：

- `compute_metrics` 可以返回单个 `float`
- 也可以返回 `dict[str, float]`
- `metric_for_best_model` 决定 early stopping 和 best checkpoint 监控哪个指标
- `greater_is_better` 决定该指标是“越大越好”还是“越小越好”

### 3. MTLTrainer 支持自定义逐任务 loss 计算

`MTLTrainer` 新增了：

- `compute_loss_func`

核心接口：

```python
compute_loss_func(model, x_dict, ys, y_preds) -> list[torch.Tensor]
```

返回值必须和任务数一致，即每个 task 返回一个标量 loss。Trainer 会继续负责：

- 多任务 loss 聚合
- `uwl` / `gradnorm` / `metabalance` 等已有自适应逻辑
- 正则化损失叠加
- 日志记录
- early stopping

### 4. EarlyStopper 支持最小化指标

`EarlyStopper` 现在支持：

- `mode="max"`：适合 AUC、Recall、NDCG 这类越大越好的指标
- `mode="min"`：适合 loss、logloss、MSE、MAE、RMSE 这类越小越好的指标
- `delta`：只有超过阈值的改善才算真正提升

这意味着 Trainer 可以和 `logloss`、`mae` 等指标正确联动，而不是默认按 AUC 语义处理。

## 为什么把文档放在 `core/`

这次能力本质上属于“训练器接口扩展”，不是一个独立工具，也不是单独的教程案例，所以最合适的位置是：

- 页面位置：`docs/zh/core/trainer_hooks.md`
- 导航位置：`核心组件 / Training & Eval` 旁边

放在这里有两个好处：

1. 用户在看 `CTRTrainer`、`MatchTrainer`、`MTLTrainer` 时，能直接找到这组扩展接口。
2. 这页和 `core/evaluation` 的关系很清晰：前者讲“训练器有哪些能力”，后者讲“怎么针对 loss / metric / monitor 做高级自定义”。

如果放到 `tools/`，会误导用户把它理解成独立工具，而不是 Trainer 自身的能力。

## 使用方式总览

### CTRTrainer

新增参数：

- `compute_loss_func`
- `compute_metrics`
- `metric_for_best_model`
- `greater_is_better`

最常见的场景是：

- 自定义 focal loss
- 监控 `logloss` 或自定义校准指标
- 保留默认训练流程，但用自己的验证逻辑选最佳模型

示例：

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

新增参数和 `CTRTrainer` 一致，但更适合以下场景：

- 替换 point-wise / pair-wise / list-wise 默认 loss
- 对 in-batch negative sampling 之外的训练目标做实验
- 用 AUC 以外的指标选最佳召回模型

示例：

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

注意：

- point-wise 自定义 loss 会收到已经转换好的 `float` 标签
- pair-wise / list-wise 仍然复用 Trainer 自身的 mode 语义
- 如果不传 `compute_metrics`，默认验证指标仍然只有 `auc`

### MTLTrainer

`MTLTrainer` 这次增强最多，支持两类自定义方式。

#### 方式一：保留默认评估逻辑，只替换每个 task 的 loss

```python
import torch.nn.functional as F

from torch_rechub.trainers import MTLTrainer


def custom_task_losses(model, x_dict, ys, y_preds):
    del model, x_dict
    return [
        F.binary_cross_entropy(y_preds[:, 0], ys[:, 0].float()),
        F.binary_cross_entropy(y_preds[:, 1], ys[:, 1].float()),
    ]


trainer = MTLTrainer(
    model=model,
    task_types=["classification", "classification"],
    compute_loss_func=custom_task_losses,
)
```

#### 方式二：同时自定义验证指标和监控目标

```python
import numpy as np

from torch_rechub.trainers import MTLTrainer


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
    compute_metrics=multitask_metrics,
    metric_for_best_model="task_1_mae",
    greater_is_better=False,
)
```

## 监控规则与默认行为

这是这次功能最容易踩坑的部分。

### 1. 不传 `compute_metrics` 时

`CTRTrainer` 和 `MatchTrainer`：

- 仍然只产出默认 `auc`
- 因此 `metric_for_best_model` 只能是 `auc`
- 如果你写成 `logloss`、`mae` 等名称，会直接报错

`MTLTrainer`：

- 仍然按每个 task 的默认评估函数工作
- 默认监控 `task_{earlystop_taskid}_{default_metric}`
- 分类任务默认指标名通常是 `auc`
- 回归任务默认指标名通常是 `mse`

### 2. `compute_metrics` 返回单个 float 时

Trainer 会把它自动包装成：

```python
{"metric": your_value}
```

并且默认：

- `metric_for_best_model = "metric"`
- `greater_is_better = False`

也就是说，如果你返回的是一个匿名标量，Trainer 会保守地把它当成“越小越好”的目标处理。

### 3. `compute_metrics` 返回多个指标时

例如：

```python
{
    "auc": 0.81,
    "logloss": 0.42,
}
```

这时你应该显式传：

```python
metric_for_best_model="logloss"
greater_is_better=False
```

否则当返回多个 key 时，Trainer 无法可靠判断你到底想监控哪个指标。

对 `MTLTrainer` 来说，自定义 metric 字典的 key 会直接进入日志，`task_{id}_...` 命名只是可选约定。

### 4. `greater_is_better` 的自动推断

如果你传了 `metric_for_best_model`，但没传 `greater_is_better`，Trainer 会根据名字推断方向：

- 包含 `loss`、`logloss`、`mse`、`mae`、`rmse`、`error` 等字样时，按“越小越好”
- 其他名称默认按“越大越好”

但这只是启发式规则。对于业务自定义指标，建议显式写出 `greater_is_better`，不要依赖猜测。

## 日志会发生什么变化

开启自定义 metric 后，Trainer 会把验证指标完整记录到 logger。

例如：

- CTR / Match: `val/logloss`、`val/auc`
- MTL: `val/task_0_mae`、`val/task_1_mae`

## 常见约束与报错含义

### `compute_loss_func` 返回数量不对

如果你配置了 2 个 task，却只返回 1 个 loss，Trainer 会报错。原因很直接：Trainer 需要逐 task 的 loss，才能继续做聚合和多任务权重处理。

### `metric_for_best_model` 在返回指标里不存在

这通常表示：

- 指标名拼错了
- `compute_metrics` 返回结构和预期不一致
- MTL 场景下 task id 对不上

## 推荐实践

- 只想换训练 loss 时，优先改 `compute_loss_func`，不要重写整个 Trainer
- 只想换验证指标时，优先改 `compute_metrics`
- 返回多个指标时，始终显式设置 `metric_for_best_model`
- 监控 `loss` / `mae` / `rmse` 一类指标时，显式写 `greater_is_better=False`
- 如果是实验性指标，先打印 `trainer.evaluate(..., return_dict=True)` 看实际 key，再配置 monitor

## 和 `main` 分支相比的行为变化

可以把这次改动理解成三件事：

1. `main` 分支里的 Trainer 更偏“固定训练范式”，loss 和 metric 选择空间有限。
2. 当前分支把 Trainer 变成了“可插拔训练骨架”，你可以只替换 loss / metric，而不用复制训练循环。
3. early stopping 不再默认绑定 AUC 语义，而是能正确支持最小化型指标。

如果你的目标是做研究实验、快速替换损失函数、对齐公司内部评估口径，当前分支会比 `main` 明显更实用。
