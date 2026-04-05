import os
import re

import numpy as np
import torch
import torch.nn as nn
import tqdm

from ..basic.callback import EarlyStopper
from ..basic.loss_func import RegularizationLoss
from ..models.multi_task import ESMM
from ..utils.data import get_loss_func, get_metric_func
from ..utils.mtl import MetaBalance, gradnorm, shared_task_layers


class MTLTrainer(object):
    """A trainer for multi task learning.

    Args:
        model (nn.Module): any multi task learning model.
        task_types (list): types of tasks, only support ["classfication", "regression"].
        optimizer_fn (torch.optim): optimizer function of pytorch (default = `torch.optim.Adam`).
        optimizer_params (dict): parameters of optimizer_fn.
        scheduler_fn (torch.optim.lr_scheduler) : torch scheduling class, eg. `torch.optim.lr_scheduler.StepLR`.
        scheduler_params (dict): parameters of optimizer scheduler_fn.
        adaptive_params (dict): parameters of adaptive loss weight method. Now only support `{"method" : "uwl"}`.
        n_epoch (int): epoch number of training.
        earlystop_taskid (int): task id of earlystop metrics relies between multi task (default = 0).
        earlystop_patience (int): how long to wait after last time validation auc improved (default = 10).
        device (str): `"cpu"` or `"cuda:0"`
        gpus (list): id of multi gpu (default=[]). If the length >=1, then the model will wrapped by nn.DataParallel.
        model_path (str): the path you want to save the model (default="./"). Note only save the best weight in the validation data.
        compute_task_losses_func (callable, optional): custom multi-task loss
            hook with signature ``compute_task_losses_func(model, x_dict, ys,
            y_preds)`` returning one scalar loss tensor per task.
        evaluate_fns (list, optional): custom per-task metric functions used by
            the default evaluation path.
        metric_names (list[str], optional): names used in logs and metric keys
            such as ``task_0_auc``.
        compute_metrics (callable, optional): custom metric hook with signature
            ``compute_metrics(targets, predicts)`` returning a float or a
            metric dict.
        metric_for_best_model (str, optional): metric key monitored by early
            stopping and best-checkpoint selection.
        greater_is_better (bool, optional): whether larger values of
            ``metric_for_best_model`` indicate a better model.
    """

    def __init__(
        self,
        model,
        task_types,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=None,
        regularization_params=None,
        scheduler_fn=None,
        scheduler_params=None,
        adaptive_params=None,
        n_epoch=10,
        earlystop_taskid=0,
        earlystop_patience=10,
        device="cpu",
        gpus=None,
        model_path="./",
        model_logger=None,
        compute_task_losses_func=None,
        evaluate_fns=None,
        metric_names=None,
        compute_metrics=None,
        metric_for_best_model=None,
        greater_is_better=None,
    ):
        self.model = model
        if gpus is None:
            gpus = []
        if optimizer_params is None:
            optimizer_params = {"lr": 1e-3, "weight_decay": 1e-5}
        if regularization_params is None:
            regularization_params = {"embedding_l1": 0.0, "embedding_l2": 0.0, "dense_l1": 0.0, "dense_l2": 0.0}
        self.task_types = task_types
        self.n_task = len(task_types)
        self.loss_weight = None
        self.adaptive_method = None
        if adaptive_params is not None:
            if adaptive_params["method"] == "uwl":
                self.adaptive_method = "uwl"
                self.loss_weight = nn.ParameterList(nn.Parameter(torch.zeros(1)) for _ in range(self.n_task))
                self.model.add_module("loss weight", self.loss_weight)
            elif adaptive_params["method"] == "metabalance":
                self.adaptive_method = "metabalance"
                share_layers, task_layers = shared_task_layers(self.model)
                self.meta_optimizer = MetaBalance(share_layers)
                self.share_optimizer = optimizer_fn(share_layers, **optimizer_params)
                self.task_optimizer = optimizer_fn(task_layers, **optimizer_params)
            elif adaptive_params["method"] == "gradnorm":
                self.adaptive_method = "gradnorm"
                self.alpha = adaptive_params.get("alpha", 0.16)
                share_layers = shared_task_layers(self.model)[0]
                # gradnorm calculate the gradients of each loss on the last
                # fully connected shared layer weight(dimension is 2)
                for i in range(len(share_layers)):
                    if share_layers[-i].ndim == 2:
                        self.last_share_layer = share_layers[-i]
                        break
                self.initial_task_loss = None
                self.loss_weight = nn.ParameterList(nn.Parameter(torch.ones(1)) for _ in range(self.n_task))
                self.model.add_module("loss weight", self.loss_weight)
        if self.adaptive_method != "metabalance":
            self.optimizer = optimizer_fn(self.model.parameters(), **optimizer_params)  # default Adam optimizer
        self.scheduler = None
        if scheduler_fn is not None:
            self.scheduler = scheduler_fn(self.optimizer, **scheduler_params)
        self.task_loss_fns = [get_loss_func(task_type) for task_type in task_types]
        self.compute_task_losses_func = compute_task_losses_func or self._compute_default_task_losses
        if evaluate_fns is None:
            self.evaluate_fns = [get_metric_func(task_type) for task_type in task_types]
        else:
            if len(evaluate_fns) != self.n_task:
                raise ValueError(f"evaluate_fns must have length {self.n_task}, got {len(evaluate_fns)}")
            self.evaluate_fns = list(evaluate_fns)
        self.compute_metrics = compute_metrics
        self._validate_metric_configuration(metric_names, evaluate_fns)
        self.metric_names = self._init_metric_names(task_types, metric_names)
        self.metric_for_best_model = metric_for_best_model
        self.greater_is_better = greater_is_better
        self._should_infer_monitor_direction = greater_is_better is None
        self._initialize_metric_configuration(earlystop_taskid)
        self.n_epoch = n_epoch
        self.earlystop_taskid = earlystop_taskid
        self.early_stopper = EarlyStopper(
            patience=earlystop_patience,
            mode="max" if self.greater_is_better else "min",
        )

        self.gpus = gpus
        if len(gpus) > 1:
            print('parallel running on these gpus:', gpus)
            self.model = torch.nn.DataParallel(self.model, device_ids=gpus)
        # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model_path = model_path
        # Initialize regularization loss
        self.reg_loss_fn = RegularizationLoss(**regularization_params)
        self.model_logger = model_logger

    def train_one_epoch(self, data_loader):
        self.model.train()
        total_loss = np.zeros(self.n_task)
        tk0 = tqdm.tqdm(data_loader, desc="train", smoothing=0, mininterval=1.0)
        for iter_i, (x_dict, ys) in enumerate(tk0):
            x_dict = {k: v.to(self.device) for k, v in x_dict.items()}  # tensor to GPU
            ys = ys.to(self.device)
            y_preds = self.model(x_dict)
            loss_list = self._compute_task_losses(x_dict, ys, y_preds)
            loss = self._aggregate_loss(loss_list)

            # Add regularization loss
            reg_loss = self.reg_loss_fn(self.model)
            loss = loss + reg_loss
            if self.adaptive_method == 'metabalance':
                self.share_optimizer.zero_grad()
                self.task_optimizer.zero_grad()
                self.meta_optimizer.step(loss_list)
                self.share_optimizer.step()
                self.task_optimizer.step()
            elif self.adaptive_method == "gradnorm":
                self.optimizer.zero_grad()
                if self.initial_task_loss is None:
                    self.initial_task_loss = [l.item() for l in loss_list]
                gradnorm(loss_list, self.loss_weight, self.last_share_layer, self.initial_task_loss, self.alpha)
                self.optimizer.step()
                # renormalize
                loss_weight_sum = sum([w.item() for w in self.loss_weight])
                normalize_coeff = len(self.loss_weight) / loss_weight_sum
                for w in self.loss_weight:
                    w.data = w.data * normalize_coeff
            else:
                self.model.zero_grad()
                loss.backward()
                self.optimizer.step()
            total_loss += np.array([l.item() for l in loss_list])
        log_dict = {"task_%d:" % (i): total_loss[i] / (iter_i + 1) for i in range(self.n_task)}
        loss_list = [total_loss[i] / (iter_i + 1) for i in range(self.n_task)]
        print("train loss: ", log_dict)
        if self.loss_weight:
            print("loss weight: ", [w.item() for w in self.loss_weight])

        return loss_list

    def fit(self, train_dataloader, val_dataloader, mode='base', seed=0):
        total_log = []

        # Log hyperparameters once
        for logger in self._iter_loggers():
            logger.log_hyperparams({'n_epoch': self.n_epoch, 'learning_rate': self._current_lr(), 'adaptive_method': self.adaptive_method})

        for epoch_i in range(self.n_epoch):
            _log_per_epoch = self.train_one_epoch(train_dataloader)

            # Collect metrics
            logs = {f'train/task_{task_id}_loss': loss_val for task_id, loss_val in enumerate(_log_per_epoch)}
            lr_value = self._current_lr()
            if lr_value is not None:
                logs['learning_rate'] = lr_value

            if self.scheduler is not None:
                if epoch_i % self.scheduler.step_size == 0:
                    print("Current lr : {}".format(self.optimizer.state_dict()['param_groups'][0]['lr']))
                self.scheduler.step()  # update lr in epoch level by scheduler

            metrics = self.evaluate(self.model, val_dataloader, return_dict=True)
            monitor_value = self._get_monitor_value(metrics)
            print('epoch:', epoch_i, 'validation metrics:', metrics)

            for task_id, score in self._task_metric_values(metrics):
                if score is None:
                    _log_per_epoch.append(float("nan"))
                    continue
                logs[f'val/task_{task_id}_score'] = score
                logs[f'val/task_{task_id}_{self.metric_names[task_id]}'] = score
                _log_per_epoch.append(score)

            for name, value in metrics.items():
                logs[f'val/{name}'] = value
            if self.metric_for_best_model == "auc" or self.metric_for_best_model.endswith("_auc"):
                logs['auc'] = monitor_value
            else:
                logs['monitor'] = monitor_value

            if self.loss_weight:
                for task_id, weight in enumerate(self.loss_weight):
                    logs[f'loss_weight/task_{task_id}'] = weight.item()

            total_log.append(_log_per_epoch)

            # Log metrics once per epoch
            for logger in self._iter_loggers():
                logger.log_metrics(logs, step=epoch_i)

            if self.early_stopper.stop_training(monitor_value, self.model.state_dict()):
                print(
                    'validation best %s of main task %d: %.6f'
                    % (self.metric_for_best_model, self.earlystop_taskid, self.early_stopper.best_score)
                )
                self.model.load_state_dict(self.early_stopper.best_weights)
                break

        torch.save(self.model.state_dict(), os.path.join(self.model_path, "model_{}_{}.pth".format(mode, seed)))  # save best auc model

        for logger in self._iter_loggers():
            logger.finish()

        return total_log

    def _iter_loggers(self):
        """Return logger instances as a list.

        Returns
        -------
        list
            Active logger instances. Empty when ``model_logger`` is ``None``.
        """
        if self.model_logger is None:
            return []
        if isinstance(self.model_logger, (list, tuple)):
            return list(self.model_logger)
        return [self.model_logger]

    def _current_lr(self):
        """Fetch current learning rate regardless of adaptive method."""
        if self.adaptive_method == "metabalance":
            return self.share_optimizer.param_groups[0]['lr'] if hasattr(self, 'share_optimizer') else None
        if hasattr(self, 'optimizer'):
            return self.optimizer.param_groups[0]['lr']
        return None

    def _init_metric_names(self, task_types, metric_names):
        """Resolve metric names used in logs and early stopping."""
        if metric_names is not None:
            if len(metric_names) != self.n_task:
                raise ValueError(f"metric_names must have length {self.n_task}, got {len(metric_names)}")
            return list(metric_names)
        default_names = []
        for task_type in task_types:
            if task_type == "classification":
                default_names.append("auc")
            elif task_type == "regression":
                default_names.append("mse")
            else:
                default_names.append("score")
        return default_names

    def _validate_metric_configuration(self, metric_names, evaluate_fns):
        """Require explicit metric names when built-in task metrics are overridden."""
        if evaluate_fns is not None and self.compute_metrics is None and metric_names is None:
            raise ValueError(
                "Custom evaluate_fns require matching metric_names when compute_metrics is not provided."
            )

    def _initialize_metric_configuration(self, default_task_id):
        """Resolve monitor defaults before training starts."""
        if self.compute_metrics is None:
            if self.metric_for_best_model is None:
                self.metric_for_best_model = f"task_{default_task_id}_{self.metric_names[default_task_id]}"
            if self.greater_is_better is None:
                monitor_task_type = self._infer_monitor_task_type(self.metric_for_best_model, default_task_id)
                self.greater_is_better = self._infer_greater_is_better(self.metric_for_best_model, monitor_task_type)
            self._should_infer_monitor_direction = False
            return
        if self.greater_is_better is None and self.metric_for_best_model is not None:
            monitor_task_type = self._infer_monitor_task_type(self.metric_for_best_model, default_task_id)
            self.greater_is_better = self._infer_greater_is_better(self.metric_for_best_model, monitor_task_type)
            self._should_infer_monitor_direction = False
        elif self.greater_is_better is None:
            # Unnamed scalar custom metrics default to minimizing until a name is resolved.
            self.greater_is_better = False

    def _infer_greater_is_better(self, metric_name, task_type):
        """Infer whether larger metric values indicate better models."""
        metric_name = metric_name.lower()
        if any(loss_name in metric_name for loss_name in ["loss", "mse", "mae", "rmse", "error", "logloss", "log_loss"]):
            return False
        if task_type == "regression":
            return False
        return True

    def _infer_monitor_task_type(self, metric_name, default_task_id):
        """Resolve the task type associated with the monitored metric."""
        match = re.match(r"task_(\d+)_", metric_name)
        if match:
            task_id = int(match.group(1))
            if 0 <= task_id < self.n_task:
                return self.task_types[task_id]
        return self.task_types[default_task_id]

    def _compute_default_task_losses(self, model, x_dict, ys, y_preds):
        """Compute the built-in per-task losses."""
        del model, x_dict
        return [self.task_loss_fns[i](y_preds[:, i], ys[:, i].float()) for i in range(self.n_task)]

    def _compute_task_losses(self, x_dict, ys, y_preds):
        """Compute one scalar loss tensor per task."""
        loss_list = self.compute_task_losses_func(self.model, x_dict, ys, y_preds)
        if len(loss_list) != self.n_task:
            raise ValueError(
                f"compute_task_losses_func must return {self.n_task} losses, got {len(loss_list)}"
            )
        return list(loss_list)

    def _aggregate_loss(self, loss_list):
        """Aggregate task losses according to the configured MTL strategy."""
        base_model = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        if isinstance(base_model, ESMM):
            # ESMM only computes loss for ctr and ctcvr tasks.
            return sum(loss_list[1:])

        if self.adaptive_method is not None:
            if self.adaptive_method == "uwl":
                loss = 0
                for loss_i, w_i in zip(loss_list, self.loss_weight):
                    w_i = torch.clamp(w_i, min=0)
                    loss += 2 * loss_i * torch.exp(-w_i) + w_i
                return loss
        return sum(loss_list) / self.n_task

    def _default_metrics(self, targets, predicts):
        """Compute the default per-task metrics."""
        metrics = {}
        for task_id in range(self.n_task):
            metric_name = f"task_{task_id}_{self.metric_names[task_id]}"
            metrics[metric_name] = float(self.evaluate_fns[task_id](targets[:, task_id], predicts[:, task_id]))
        return metrics

    def _normalize_metrics(self, metrics, default_name):
        """Normalize custom metric output to ``dict[str, float]``."""
        if isinstance(metrics, dict):
            normalized_metrics = {str(name): float(value) for name, value in metrics.items()}
            self._resolve_custom_metric_configuration(normalized_metrics, scalar_output=False)
            return normalized_metrics
        metric_name = default_name or "metric"
        normalized_metrics = {metric_name: float(metrics)}
        self._resolve_custom_metric_configuration(normalized_metrics, scalar_output=True)
        return normalized_metrics

    def _resolve_custom_metric_configuration(self, metrics, scalar_output):
        """Finalize monitor name and direction for custom metric outputs."""
        if self.compute_metrics is None:
            return
        if self.metric_for_best_model is None:
            if scalar_output:
                self.metric_for_best_model = "metric"
            else:
                if len(metrics) != 1:
                    raise ValueError(
                        "Custom compute_metrics returned multiple metrics. "
                        "Set metric_for_best_model to one of: "
                        f"{sorted(metrics.keys())}"
                    )
                self.metric_for_best_model = next(iter(metrics))
        if self._should_infer_monitor_direction:
            if scalar_output and self.metric_for_best_model == "metric":
                self.greater_is_better = False
            else:
                monitor_task_type = self._infer_monitor_task_type(self.metric_for_best_model, self.earlystop_taskid)
                self.greater_is_better = self._infer_greater_is_better(self.metric_for_best_model, monitor_task_type)
            self._should_infer_monitor_direction = False
            self._sync_early_stopper_mode()

    def _sync_early_stopper_mode(self):
        """Keep the early stopper aligned with the active monitor direction."""
        if hasattr(self, "early_stopper"):
            self.early_stopper.mode = "max" if self.greater_is_better else "min"

    def _get_monitor_value(self, metrics):
        """Extract the score tracked by early stopping and model selection."""
        if self.metric_for_best_model not in metrics:
            raise ValueError(
                f"metric_for_best_model={self.metric_for_best_model!r} was not found in evaluation metrics: {sorted(metrics.keys())}"
            )
        return metrics[self.metric_for_best_model]

    def _task_metric_values(self, metrics):
        """Return per-task metric values in task order with task ids preserved."""
        values = []
        for task_id, metric_name in enumerate(self.metric_names):
            key = f"task_{task_id}_{metric_name}"
            values.append((task_id, metrics.get(key)))
        return values

    def evaluate(self, model, data_loader, return_dict=None):
        """Evaluate the model on a validation/test data loader.

        Args:
            model (nn.Module): model to evaluate.
            data_loader (DataLoader): evaluation data loader.
            return_dict (bool, optional): when ``True``, return a metric dict.
                When ``False``, return the historical list output used by the
                built-in per-task evaluators, or the scalar selected by
                ``metric_for_best_model`` when ``compute_metrics`` is set.

        Returns:
            list[float] or float or dict[str, float]: evaluation result.
        """
        if return_dict is None:
            return_dict = self.compute_metrics is not None
        model.eval()
        targets, predicts = list(), list()
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc="validation", smoothing=0, mininterval=1.0)
            for i, (x_dict, ys) in enumerate(tk0):
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}  # tensor to GPU
                ys = ys.to(self.device)
                y_preds = self.model(x_dict)
                targets.extend(ys.tolist())
                predicts.extend(y_preds.tolist())
        targets, predicts = np.array(targets), np.array(predicts)
        if self.compute_metrics is not None:
            metrics = self._normalize_metrics(
                self.compute_metrics(targets, predicts),
                default_name=self.metric_for_best_model,
            )
            if return_dict:
                return metrics
            return self._get_monitor_value(metrics)

        metrics = self._default_metrics(targets, predicts)
        if return_dict:
            return metrics
        return [metrics[f"task_{task_id}_{self.metric_names[task_id]}"] for task_id in range(self.n_task)]

    def predict(self, model, data_loader):
        model.eval()
        predicts = list()
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc="predict", smoothing=0, mininterval=1.0)
            for i, x_dict in enumerate(tk0):
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
                y_preds = model(x_dict)
                predicts.extend(y_preds.tolist())
        return predicts

    def export_onnx(self, output_path, dummy_input=None, batch_size=2, seq_length=10, opset_version=14, dynamic_batch=True, device=None, verbose=False, onnx_export_kwargs=None):
        """Export the trained multi-task model to ONNX format.

        This method exports multi-task learning models (e.g., MMOE, PLE, ESMM, SharedBottom)
        to ONNX format for deployment. The exported model will have multiple outputs
        corresponding to each task.

        Note:
            The ONNX model will output a tensor of shape [batch_size, n_task] where
            n_task is the number of tasks in the multi-task model.

        Args:
            output_path (str): Path to save the ONNX model file.
            dummy_input (dict, optional): Example input dict {feature_name: tensor}.
                If not provided, dummy inputs will be generated automatically.
            batch_size (int): Batch size for auto-generated dummy input (default: 2).
            seq_length (int): Sequence length for SequenceFeature (default: 10).
            opset_version (int): ONNX opset version (default: 14).
            dynamic_batch (bool): Enable dynamic batch size (default: True).
            device (str, optional): Device for export ('cpu', 'cuda', etc.).
                If None, defaults to 'cpu' for maximum compatibility.
            verbose (bool): Print export details (default: False).
            onnx_export_kwargs (dict, optional): Extra kwargs forwarded to ``torch.onnx.export``.

        Returns:
            bool: True if export succeeded, False otherwise.

        Example:
            >>> trainer = MTLTrainer(mmoe_model, task_types=["classification", "classification"], ...)
            >>> trainer.fit(train_dl, val_dl)
            >>> trainer.export_onnx("mmoe.onnx")

            >>> # Export on specific device
            >>> trainer.export_onnx("mmoe.onnx", device="cpu")
        """
        from ..utils.onnx_export import ONNXExporter

        # Handle DataParallel wrapped model
        model = self.model.module if hasattr(self.model, 'module') else self.model

        # Use provided device or default to 'cpu'
        export_device = device if device is not None else 'cpu'

        exporter = ONNXExporter(model, device=export_device)
        return exporter.export(
            output_path=output_path,
            dummy_input=dummy_input,
            batch_size=batch_size,
            seq_length=seq_length,
            opset_version=opset_version,
            dynamic_batch=dynamic_batch,
            verbose=verbose,
            onnx_export_kwargs=onnx_export_kwargs,
        )

    def visualization(self, input_data=None, batch_size=2, seq_length=10, depth=3, show_shapes=True, expand_nested=True, save_path=None, graph_name="model", device=None, dpi=300, **kwargs):
        """Visualize the model's computation graph.

        This method generates a visual representation of the model architecture,
        showing layer connections, tensor shapes, and nested module structures.
        It automatically extracts feature information from the model.

        Parameters
        ----------
        input_data : dict, optional
            Example input dict {feature_name: tensor}.
            If not provided, dummy inputs will be generated automatically.
        batch_size : int, default=2
            Batch size for auto-generated dummy input.
        seq_length : int, default=10
            Sequence length for SequenceFeature.
        depth : int, default=3
            Visualization depth, higher values show more detail.
            Set to -1 to show all layers.
        show_shapes : bool, default=True
            Whether to display tensor shapes.
        expand_nested : bool, default=True
            Whether to expand nested modules.
        save_path : str, optional
            Path to save the graph image (.pdf, .svg, .png).
            If None, displays in Jupyter or opens system viewer.
        graph_name : str, default="model"
            Name for the graph.
        device : str, optional
            Device for model execution. If None, defaults to 'cpu'.
        dpi : int, default=300
            Resolution in dots per inch for output image.
            Higher values produce sharper images suitable for papers.
        **kwargs : dict
            Additional arguments passed to torchview.draw_graph().

        Returns
        -------
        ComputationGraph
            A torchview ComputationGraph object.

        Raises
        ------
        ImportError
            If torchview or graphviz is not installed.

        Notes
        -----
        Default Display Behavior:
            When `save_path` is None (default):
            - In Jupyter/IPython: automatically displays the graph inline
            - In Python script: opens the graph with system default viewer

        Examples
        --------
        >>> trainer = MTLTrainer(model, task_types=["classification", "classification"])
        >>> trainer.fit(train_dl, val_dl)
        >>>
        >>> # Auto-display in Jupyter (no save_path needed)
        >>> trainer.visualization(depth=4)
        >>>
        >>> # Save to high-DPI PNG for papers
        >>> trainer.visualization(save_path="model.png", dpi=300)
        """
        from ..utils.visualization import TORCHVIEW_AVAILABLE, visualize_model

        if not TORCHVIEW_AVAILABLE:
            raise ImportError(
                "Visualization requires torchview. "
                "Install with: pip install torch-rechub[visualization]\n"
                "Also ensure graphviz is installed on your system:\n"
                "  - Ubuntu/Debian: sudo apt-get install graphviz\n"
                "  - macOS: brew install graphviz\n"
                "  - Windows: choco install graphviz"
            )

        # Handle DataParallel wrapped model
        model = self.model.module if hasattr(self.model, 'module') else self.model

        # Use provided device or default to 'cpu'
        viz_device = device if device is not None else 'cpu'

        return visualize_model(
            model,
            input_data=input_data,
            batch_size=batch_size,
            seq_length=seq_length,
            depth=depth,
            show_shapes=show_shapes,
            expand_nested=expand_nested,
            save_path=save_path,
            graph_name=graph_name,
            device=viz_device,
            dpi=dpi,
            **kwargs
        )
