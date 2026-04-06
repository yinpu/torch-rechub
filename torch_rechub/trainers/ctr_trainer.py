import os

import torch
import tqdm
from sklearn.metrics import roc_auc_score

from ..basic.callback import EarlyStopper
from ..basic.loss_func import RegularizationLoss


class CTRTrainer(object):
    """A general trainer for single task learning.

    Args:
        model (nn.Module): any multi task learning model.
        optimizer_fn (torch.optim): optimizer function of pytorch (default = `torch.optim.Adam`).
        optimizer_params (dict): parameters of optimizer_fn.
        scheduler_fn (torch.optim.lr_scheduler) : torch scheduling class, eg. `torch.optim.lr_scheduler.StepLR`.
        scheduler_params (dict): parameters of optimizer scheduler_fn.
        n_epoch (int): epoch number of training.
        earlystop_patience (int): how long to wait after last time validation auc improved (default=10).
        device (str): `"cpu"` or `"cuda:0"`
        gpus (list): id of multi gpu (default=[]). If the length >=1, then the model will wrapped by nn.DataParallel.
        loss_mode (bool): whether the model returns only prediction or prediction with extra loss
            (`True`: `model(x_dict) -> y_pred`, `False`: `model(x_dict) -> (y_pred, other_loss)`).
        model_path (str): the path you want to save the model (default="./"). Note only save the best weight in the validation data.
        embedding_l1 (float): L1 regularization coefficient for embedding parameters (default=0.0).
        embedding_l2 (float): L2 regularization coefficient for embedding parameters (default=0.0).
        dense_l1 (float): L1 regularization coefficient for dense parameters (default=0.0).
        dense_l2 (float): L2 regularization coefficient for dense parameters (default=0.0).
        compute_loss_func (callable, optional): custom loss hook with signature
            ``compute_loss_func(model, x_dict, y)``.
        compute_metrics (callable, optional): custom metric hook with signature
            ``compute_metrics(y_true, y_pred)`` returning a float or a metric dict.
        metric_for_best_model (str): metric key monitored by early stopping and
            best-checkpoint selection.
        greater_is_better (bool): whether larger values of
            ``metric_for_best_model`` indicate a better model.
    """

    def __init__(
        self,
        model,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=None,
        regularization_params=None,
        scheduler_fn=None,
        scheduler_params=None,
        n_epoch=10,
        earlystop_patience=10,
        device="cpu",
        gpus=None,
        loss_mode=True,
        model_path="./",
        model_logger=None,
        compute_loss_func=None,
        compute_metrics=None,
        metric_for_best_model=None,
        greater_is_better=None,
    ):
        self.model = model  # for uniform weights save method in one gpu or multi gpu
        if gpus is None:
            gpus = []
        self.gpus = gpus
        if len(gpus) > 1:
            print('parallel running on these gpus:', gpus)
            self.model = torch.nn.DataParallel(self.model, device_ids=gpus)
        # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.model.to(self.device)
        if optimizer_params is None:
            optimizer_params = {"lr": 1e-3, "weight_decay": 1e-5}
        self.optimizer = optimizer_fn(self.model.parameters(), **optimizer_params)  # default optimizer
        if regularization_params is None:
            regularization_params = {"embedding_l1": 0.0, "embedding_l2": 0.0, "dense_l1": 0.0, "dense_l2": 0.0}
        self.scheduler = None
        if scheduler_fn is not None:
            self.scheduler = scheduler_fn(self.optimizer, **scheduler_params)
        self.loss_mode = loss_mode
        self.criterion = torch.nn.BCELoss()  # default loss cross_entropy
        self.evaluate_fn = roc_auc_score  # default evaluate function
        self.compute_loss_func = compute_loss_func
        self.compute_metrics = compute_metrics
        self.metric_for_best_model = metric_for_best_model
        self.greater_is_better = greater_is_better
        self._should_infer_monitor_direction = greater_is_better is None
        self._validate_metric_configuration()
        self._initialize_metric_configuration()
        self.n_epoch = n_epoch
        self.early_stopper = EarlyStopper(
            patience=earlystop_patience,
            mode="max" if self.greater_is_better else "min",
        )
        self.model_path = model_path
        # Initialize regularization loss
        self.reg_loss_fn = RegularizationLoss(**regularization_params)
        self.model_logger = model_logger

    def train_one_epoch(self, data_loader, log_interval=10):
        self.model.train()
        total_loss = 0
        epoch_loss = 0
        batch_count = 0
        tk0 = tqdm.tqdm(data_loader, desc="train", smoothing=0, mininterval=1.0)
        for i, (x_dict, y) in enumerate(tk0):
            x_dict = {k: v.to(self.device) for k, v in x_dict.items()}  # tensor to GPU
            y = y.to(self.device).float()
            loss = self._compute_batch_loss(x_dict, y)

            # Add regularization loss
            reg_loss = self.reg_loss_fn(self.model)
            loss = loss + reg_loss

            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            epoch_loss += loss.item()
            batch_count += 1
            if (i + 1) % log_interval == 0:
                tk0.set_postfix(loss=total_loss / log_interval)
                total_loss = 0

        # Return average epoch loss
        return epoch_loss / batch_count if batch_count > 0 else 0

    def fit(self, train_dataloader, val_dataloader=None):
        for logger in self._iter_loggers():
            logger.log_hyperparams({'n_epoch': self.n_epoch, 'learning_rate': self.optimizer.param_groups[0]['lr'], 'loss_mode': self.loss_mode})

        for epoch_i in range(self.n_epoch):
            print('epoch:', epoch_i)
            train_loss = self.train_one_epoch(train_dataloader)

            for logger in self._iter_loggers():
                logger.log_metrics({'train/loss': train_loss, 'learning_rate': self.optimizer.param_groups[0]['lr']}, step=epoch_i)

            if self.scheduler is not None:
                if epoch_i % self.scheduler.step_size == 0:
                    print("Current lr : {}".format(self.optimizer.state_dict()['param_groups'][0]['lr']))
                self.scheduler.step()  # update lr in epoch level by scheduler

            if val_dataloader:
                metrics = self._evaluate_metrics(self.model, val_dataloader)
                monitor_value = self._get_monitor_value(metrics)
                print('epoch:', epoch_i, 'validation metrics:', metrics)

                for logger in self._iter_loggers():
                    logger.log_metrics({f'val/{name}': value for name, value in metrics.items()}, step=epoch_i)

                if self.early_stopper.stop_training(monitor_value, self.model.state_dict()):
                    print(f'validation: best {self.metric_for_best_model}: {self.early_stopper.best_score}')
                    self.model.load_state_dict(self.early_stopper.best_weights)
                    break

        torch.save(self.model.state_dict(), os.path.join(self.model_path, "model.pth"))  # save best auc model

        for logger in self._iter_loggers():
            logger.finish()

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

    def evaluate(self, model, data_loader, return_dict=None):
        """Evaluate the model on a validation/test data loader.

        Args:
            model (nn.Module): model to evaluate.
            data_loader (DataLoader): evaluation data loader.
            return_dict (bool, optional): when ``True``, return a metric dict.
                When ``False``, return the scalar selected by
                ``metric_for_best_model``. Defaults to ``True`` only when a
                custom ``compute_metrics`` hook is configured.

        Returns:
            float or dict[str, float]: scalar monitor metric or a metric dict.
        """
        if return_dict is None:
            return_dict = self.compute_metrics is not None
        metrics = self._evaluate_metrics(model, data_loader)
        if return_dict:
            return metrics
        return self._get_monitor_value(metrics)

    def _compute_batch_loss(self, x_dict, y):
        """Compute task loss before regularization."""
        if self.compute_loss_func is not None:
            return self.compute_loss_func(self.model, x_dict, y)
        if self.loss_mode:
            y_pred = self.model(x_dict)
            return self.criterion(y_pred, y)
        y_pred, other_loss = self.model(x_dict)
        return self.criterion(y_pred, y) + other_loss

    def _predict_batch(self, model, x_dict):
        """Run a forward pass and return predictions only."""
        if self.loss_mode:
            return model(x_dict)
        y_pred, _ = model(x_dict)
        return y_pred

    def _evaluate_metrics(self, model, data_loader):
        """Collect predictions on ``data_loader`` and compute metrics."""
        model.eval()
        targets, predicts = list(), list()
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc="validation", smoothing=0, mininterval=1.0)
            for i, (x_dict, y) in enumerate(tk0):
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
                # 确保y是float类型且维度为[batch_size, 1]
                y = y.to(self.device).float().view(-1, 1)
                y_pred = self._predict_batch(model, x_dict)
                targets.extend(y.tolist())
                predicts.extend(y_pred.tolist())
        if self.compute_metrics is not None:
            raw_metrics = self.compute_metrics(targets, predicts)
            return self._normalize_metrics(raw_metrics, default_name=self.metric_for_best_model)
        return self._normalize_metrics(self.evaluate_fn(targets, predicts), default_name="auc")

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

    def _initialize_metric_configuration(self):
        """Resolve monitor defaults before training starts."""
        if self.compute_metrics is None:
            if self.metric_for_best_model is None:
                self.metric_for_best_model = "auc"
            if self.greater_is_better is None:
                self.greater_is_better = True
            self._should_infer_monitor_direction = False
            return
        if self.greater_is_better is None and self.metric_for_best_model is not None:
            self.greater_is_better = self._infer_greater_is_better(self.metric_for_best_model)
            self._should_infer_monitor_direction = False
        elif self.greater_is_better is None:
            # Unnamed scalar custom metrics default to minimizing until a name is resolved.
            self.greater_is_better = False

    def _resolve_custom_metric_configuration(self, metrics, scalar_output):
        """Finalize monitor name and direction for custom metric outputs."""
        if self.compute_metrics is None:
            return
        if self.metric_for_best_model is None:
            if scalar_output:
                self.metric_for_best_model = "metric"
            else:
                if len(metrics) == 1:
                    self.metric_for_best_model = next(iter(metrics))
                else:
                    return
        if self._should_infer_monitor_direction and self.metric_for_best_model is not None:
            if scalar_output and self.metric_for_best_model == "metric":
                self.greater_is_better = False
            else:
                self.greater_is_better = self._infer_greater_is_better(self.metric_for_best_model)
            self._should_infer_monitor_direction = False
            self._sync_early_stopper_mode()

    def _infer_greater_is_better(self, metric_name):
        """Infer whether larger metric values indicate better models."""
        metric_name = metric_name.lower()
        if any(loss_name in metric_name for loss_name in ["loss", "mse", "mae", "rmse", "error", "logloss", "log_loss"]):
            return False
        return True

    def _sync_early_stopper_mode(self):
        """Keep the early stopper aligned with the active monitor direction."""
        if hasattr(self, "early_stopper"):
            self.early_stopper.mode = "max" if self.greater_is_better else "min"

    def _validate_metric_configuration(self):
        """Reject monitor names unsupported by the built-in evaluator."""
        if self.compute_metrics is None and self.metric_for_best_model not in {None, "auc"}:
            raise ValueError(
                "CTRTrainer default evaluation only returns 'auc'. "
                "Pass compute_metrics to monitor a different metric."
            )

    def _get_monitor_value(self, metrics):
        """Extract the score tracked by early stopping and model selection."""
        if self.metric_for_best_model is None:
            raise ValueError(
                "Custom compute_metrics returned multiple metrics. "
                "Set metric_for_best_model to one of: "
                f"{sorted(metrics.keys())}"
            )
        if self.metric_for_best_model not in metrics:
            raise ValueError(
                f"metric_for_best_model={self.metric_for_best_model!r} was not found in evaluation metrics: {sorted(metrics.keys())}"
            )
        return metrics[self.metric_for_best_model]

    def predict(self, model, data_loader):
        model.eval()
        predicts = list()
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc="predict", smoothing=0, mininterval=1.0)
            for i, (x_dict, y) in enumerate(tk0):
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
                y = y.to(self.device)
                y_pred = self._predict_batch(model, x_dict)
                predicts.extend(y_pred.tolist())
        return predicts

    def export_onnx(self, output_path, dummy_input=None, batch_size=2, seq_length=10, opset_version=14, dynamic_batch=True, device=None, verbose=False, onnx_export_kwargs=None):
        """Export the trained model to ONNX format.

        This method exports the ranking model (e.g., DeepFM, WideDeep, DCN) to ONNX format
        for deployment. The export is non-invasive and does not modify the model code.

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
            >>> trainer = CTRTrainer(model, ...)
            >>> trainer.fit(train_dl, val_dl)
            >>> trainer.export_onnx("deepfm.onnx")

            >>> # With custom dummy input
            >>> dummy = {"user_id": torch.tensor([1, 2]), "item_id": torch.tensor([10, 20])}
            >>> trainer.export_onnx("model.onnx", dummy_input=dummy)

            >>> # Export on specific device
            >>> trainer.export_onnx("model.onnx", device="cpu")
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

        Args:
            input_data (dict, optional): Example input dict {feature_name: tensor}.
                If not provided, dummy inputs will be generated automatically.
            batch_size (int): Batch size for auto-generated dummy input (default: 2).
            seq_length (int): Sequence length for SequenceFeature (default: 10).
            depth (int): Visualization depth, higher values show more detail.
                Set to -1 to show all layers (default: 3).
            show_shapes (bool): Whether to display tensor shapes (default: True).
            expand_nested (bool): Whether to expand nested modules (default: True).
            save_path (str, optional): Path to save the graph image (.pdf, .svg, .png).
                If None, displays in Jupyter or opens system viewer.
            graph_name (str): Name for the graph (default: "model").
            device (str, optional): Device for model execution. If None, defaults to 'cpu'.
            dpi (int): Resolution in dots per inch for output image.
                Higher values produce sharper images suitable for papers (default: 300).
            **kwargs: Additional arguments passed to ``torchview.draw_graph()``.

        Returns:
            ComputationGraph: A torchview ComputationGraph object.

        Raises:
            ImportError: If torchview or graphviz is not installed.

        Note:
            When ``save_path`` is None (default):
            - In Jupyter/IPython: automatically displays the graph inline
            - In Python script: opens the graph with system default viewer

        Example:
            >>> trainer = CTRTrainer(model, ...)
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
