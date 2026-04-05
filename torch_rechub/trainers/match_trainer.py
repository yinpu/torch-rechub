import os

import torch
import tqdm
from sklearn.metrics import roc_auc_score

from ..basic.callback import EarlyStopper
from ..basic.loss_func import BPRLoss, RegularizationLoss
from ..utils.match import gather_inbatch_logits, inbatch_negative_sampling


class MatchTrainer(object):
    """A general trainer for Matching/Retrieval

    Args:
        model (nn.Module): any matching model.
        mode (int, optional): the training mode, `{0:point-wise, 1:pair-wise, 2:list-wise}`. Defaults to 0.
        optimizer_fn (torch.optim): optimizer function of pytorch (default = `torch.optim.Adam`).
        optimizer_params (dict): parameters of optimizer_fn.
        scheduler_fn (torch.optim.lr_scheduler) : torch scheduling class, eg. `torch.optim.lr_scheduler.StepLR`.
        scheduler_params (dict): parameters of optimizer scheduler_fn.
        n_epoch (int): epoch number of training.
        earlystop_patience (int): how long to wait after last time validation auc improved (default=10).
        device (str): `"cpu"` or `"cuda:0"`
        gpus (list): id of multi gpu (default=[]). If the length >=1, then the model will wrapped by nn.DataParallel.
        model_path (str): the path you want to save the model (default="./"). Note only save the best weight in the validation data.
        in_batch_neg (bool): whether to use in-batch negative sampling instead of global negatives.
        in_batch_neg_ratio (int): number of negatives to draw from the batch per positive sample when in_batch_neg is True.
        hard_negative (bool): whether to choose hardest negatives within batch (top-k by score) instead of uniform random.
        sampler_seed (int): optional random seed for in-batch sampler to ease reproducibility/testing.
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
        mode=0,
        in_batch_neg=False,
        in_batch_neg_ratio=None,
        hard_negative=False,
        sampler_seed=None,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=None,
        regularization_params=None,
        scheduler_fn=None,
        scheduler_params=None,
        n_epoch=10,
        earlystop_patience=10,
        device="cpu",
        gpus=None,
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
        self.in_batch_neg = in_batch_neg
        self.in_batch_neg_ratio = in_batch_neg_ratio
        self.hard_negative = hard_negative
        self._sampler_generator = None
        if sampler_seed is not None:
            self._sampler_generator = torch.Generator(device=self.device)
            self._sampler_generator.manual_seed(sampler_seed)
        # Check model compatibility for in-batch negative sampling
        if in_batch_neg:
            base_model = model.module if isinstance(model, torch.nn.DataParallel) else model
            if not hasattr(base_model, 'user_tower') or not hasattr(base_model, 'item_tower'):
                raise ValueError(
                    f"Model {type(base_model).__name__} does not support in-batch negative sampling. "
                    "Only two-tower models with user_tower() and item_tower() methods are supported, "
                    "such as DSSM, YoutubeDNN, MIND, GRU4Rec, SINE, ComiRec, SASRec, NARM, STAMP, etc."
                )
        if optimizer_params is None:
            optimizer_params = {"lr": 1e-3, "weight_decay": 1e-5}
        if regularization_params is None:
            regularization_params = {"embedding_l1": 0.0, "embedding_l2": 0.0, "dense_l1": 0.0, "dense_l2": 0.0}
        self.mode = mode
        if mode == 0:  # point-wise loss, binary cross_entropy
            # With in-batch negatives we treat it as list-wise classification over sampled negatives
            self.criterion = torch.nn.CrossEntropyLoss() if in_batch_neg else torch.nn.BCELoss()
        elif mode == 1:  # pair-wise loss
            self.criterion = BPRLoss()
        elif mode == 2:  # list-wise loss, softmax
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError("mode only contain value in %s, but got %s" % ([0, 1, 2], mode))
        self.optimizer = optimizer_fn(self.model.parameters(), **optimizer_params)  # default optimizer
        self.scheduler = None
        if scheduler_fn is not None:
            self.scheduler = scheduler_fn(self.optimizer, **scheduler_params)
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
            y = y.to(self.device)
            loss = self._compute_batch_loss(x_dict, y)

            # Add regularization loss
            reg_loss = self.reg_loss_fn(self.model)
            loss = loss + reg_loss

            # used for debug
            # if i == 0:
            #     print()
            #     if self.mode == 0:
            #         print('pred: ', [f'{float(each):5.2g}' for each in y_pred.detach().cpu().tolist()])
            #         print('truth:', [f'{float(each):5.2g}' for each in y.detach().cpu().tolist()])
            #     elif self.mode == 2:
            #         pred = y_pred.detach().cpu().mean(0)
            #         pred = torch.softmax(pred, dim=0).tolist()
            #         print('pred: ', [f'{float(each):4.2g}' for each in pred])
            #     elif self.mode == 1:
            #         print('pos:', [f'{float(each):5.2g}' for each in pos_score.detach().cpu().tolist()])
            #         print('neg: ', [f'{float(each):5.2g}' for each in neg_score.detach().cpu().tolist()])

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
            logger.log_hyperparams({'n_epoch': self.n_epoch, 'learning_rate': self.optimizer.param_groups[0]['lr'], 'loss_mode': self.mode})

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

    def _prepare_target(self, y):
        """Cast labels to the type expected by the active training mode."""
        if self.mode == 0:
            return y.float()
        return y.long()

    def _compute_default_loss(self, x_dict, y):
        """Compute task loss before regularization using built-in training modes."""
        if self.in_batch_neg:
            base_model = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
            user_embedding = base_model.user_tower(x_dict)
            item_embedding = base_model.item_tower(x_dict)
            if user_embedding is None or item_embedding is None:
                raise ValueError("Model must return user/item embeddings when in_batch_neg is True.")
            if user_embedding.dim() > 2 and user_embedding.size(1) == 1:
                user_embedding = user_embedding.squeeze(1)
            if item_embedding.dim() > 2 and item_embedding.size(1) == 1:
                item_embedding = item_embedding.squeeze(1)
            if user_embedding.dim() != 2 or item_embedding.dim() != 2:
                raise ValueError(f"In-batch negative sampling requires 2D embeddings, got shapes {user_embedding.shape} and {item_embedding.shape}")

            scores = torch.matmul(user_embedding, item_embedding.t())  # bs x bs
            neg_indices = inbatch_negative_sampling(scores, neg_ratio=self.in_batch_neg_ratio, hard_negative=self.hard_negative, generator=self._sampler_generator)
            logits = gather_inbatch_logits(scores, neg_indices)
            if self.mode == 1:  # pair_wise
                return self.criterion(logits[:, 0], logits[:, 1:], in_batch_neg=True)
            targets = torch.zeros(logits.size(0), dtype=torch.long, device=self.device)
            return self.criterion(logits, targets)

        if self.mode == 1:  # pair_wise
            pos_score, neg_score = self.model(x_dict)
            return self.criterion(pos_score, neg_score)
        y_pred = self.model(x_dict)
        return self.criterion(y_pred, y)

    def _compute_batch_loss(self, x_dict, y):
        """Compute task loss before regularization."""
        y = self._prepare_target(y)
        if self.compute_loss_func is not None:
            return self.compute_loss_func(self.model, x_dict, y)
        return self._compute_default_loss(x_dict, y)

    def _evaluate_metrics(self, model, data_loader):
        """Collect predictions on ``data_loader`` and compute metrics."""
        model.eval()
        targets, predicts = list(), list()
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc="validation", smoothing=0, mininterval=1.0)
            for i, (x_dict, y) in enumerate(tk0):
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
                y = y.to(self.device)
                y_pred = model(x_dict)
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
                "MatchTrainer default evaluation only returns 'auc'. "
                "Pass compute_metrics to monitor a different metric."
            )

    def _get_monitor_value(self, metrics):
        """Extract the score tracked by early stopping and model selection."""
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
                y_pred = model(x_dict)
                predicts.extend(y_pred.tolist())
        return predicts

    def inference_embedding(self, model, mode, data_loader, model_path):
        # inference
        assert mode in ["user", "item"], "Invalid mode={}.".format(mode)
        model.mode = mode
        model.load_state_dict(torch.load(os.path.join(model_path, "model.pth"), map_location=self.device, weights_only=True))
        model = model.to(self.device)
        model.eval()
        predicts = []
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc="%s inference" % (mode), smoothing=0, mininterval=1.0)
            for i, x_dict in enumerate(tk0):
                x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
                y_pred = model(x_dict)
                predicts.append(y_pred.data)
        return torch.cat(predicts, dim=0)

    def export_onnx(self, output_path, mode=None, dummy_input=None, batch_size=2, seq_length=10, opset_version=14, dynamic_batch=True, device=None, verbose=False, onnx_export_kwargs=None):
        """Export the trained matching model to ONNX format.

        This method exports matching/retrieval models (e.g., DSSM, YoutubeDNN, MIND)
        to ONNX format. For dual-tower models, you can export user tower and item
        tower separately for efficient online serving.

        Args:
            output_path (str): Path to save the ONNX model file.
            mode (str, optional): Export mode for dual-tower models:
                - "user": Export only the user tower (for user embedding inference)
                - "item": Export only the item tower (for item embedding inference)
                - None: Export the full model (default)
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
            >>> trainer = MatchTrainer(dssm_model, mode=0, ...)
            >>> trainer.fit(train_dl)

            >>> # Export user tower for user embedding inference
            >>> trainer.export_onnx("user_tower.onnx", mode="user")

            >>> # Export item tower for item embedding inference
            >>> trainer.export_onnx("item_tower.onnx", mode="item")

            >>> # Export full model (for online similarity computation)
            >>> trainer.export_onnx("full_model.onnx")

            >>> # Export on specific device
            >>> trainer.export_onnx("user_tower.onnx", mode="user", device="cpu")
        """
        from ..utils.onnx_export import ONNXExporter

        # Handle DataParallel wrapped model
        model = self.model.module if hasattr(self.model, 'module') else self.model

        # Store original mode
        original_mode = getattr(model, 'mode', None)

        # Use provided device or default to 'cpu'
        export_device = device if device is not None else 'cpu'

        try:
            exporter = ONNXExporter(model, device=export_device)
            return exporter.export(
                output_path=output_path,
                mode=mode,
                dummy_input=dummy_input,
                batch_size=batch_size,
                seq_length=seq_length,
                opset_version=opset_version,
                dynamic_batch=dynamic_batch,
                verbose=verbose,
                onnx_export_kwargs=onnx_export_kwargs,
            )
        finally:
            # Restore original mode
            if hasattr(model, 'mode'):
                model.mode = original_mode

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
        >>> trainer = MatchTrainer(model, ...)
        >>> trainer.fit(train_dl)
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
