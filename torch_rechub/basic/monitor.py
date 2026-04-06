class MetricMonitor(object):
    """Track how trainer metrics map to early stopping and best-model selection."""

    def __init__(self, compute_metrics=None, metric_for_best_model=None, greater_is_better=None):
        # These fields are mutable because custom metric hooks may only reveal
        # the final metric names after the first evaluation pass.
        self.compute_metrics = compute_metrics
        self.metric_for_best_model = metric_for_best_model
        self.greater_is_better = greater_is_better
        self._should_infer_monitor_direction = greater_is_better is None
        self.early_stopper = None

    def attach_early_stopper(self, early_stopper):
        """Keep the early stopper aligned with the active monitor direction."""
        self.early_stopper = early_stopper
        self._sync_early_stopper_mode()

    def validate_builtin_metrics(self, valid_metrics, error_message):
        """Reject monitor names unsupported by the built-in evaluator."""
        # Built-in evaluator outputs are fixed, so unsupported monitor names
        # should fail early instead of surfacing later during training.
        if self.compute_metrics is None and self.metric_for_best_model not in {None, *valid_metrics}:
            raise ValueError(error_message)

    def initialize(self, default_metric):
        """Resolve monitor defaults before training starts."""
        if self.compute_metrics is None:
            if self.metric_for_best_model is None:
                self.metric_for_best_model = default_metric
            if self.greater_is_better is None:
                self.greater_is_better = self.infer_greater_is_better(self.metric_for_best_model)
            self._should_infer_monitor_direction = False
            return
        # For custom hooks, keep the monitor unresolved until we see the actual
        # metric payload unless the caller already pinned a metric name.
        if self.greater_is_better is None and self.metric_for_best_model is not None:
            self.greater_is_better = self.infer_greater_is_better(self.metric_for_best_model)
            self._should_infer_monitor_direction = False
        elif self.greater_is_better is None:
            # Unnamed scalar custom metrics default to minimizing until a name is resolved.
            self.greater_is_better = False

    def normalize(self, metrics, default_name=None):
        """Normalize metric output to ``dict[str, float]``."""
        if isinstance(metrics, dict):
            normalized_metrics = {str(name): float(value) for name, value in metrics.items()}
            self._resolve_custom_metric_configuration(normalized_metrics, scalar_output=False)
            return normalized_metrics
        # Scalar hooks are normalized to a one-key dict so trainers can treat
        # built-in and custom metric outputs through the same path.
        metric_name = default_name or self.metric_for_best_model or "metric"
        normalized_metrics = {metric_name: float(metrics)}
        self._resolve_custom_metric_configuration(normalized_metrics, scalar_output=True)
        return normalized_metrics

    def get_monitor_value(self, metrics):
        """Extract the score tracked by early stopping and model selection."""
        # Multi-metric custom hooks must choose an explicit monitor key before
        # trainers can reduce the metric dict to a single early-stop score.
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

    def _resolve_custom_metric_configuration(self, metrics, scalar_output):
        """Finalize monitor name and direction for custom metric outputs."""
        if self.compute_metrics is None:
            return
        # When a custom hook returns exactly one metric, we can safely promote
        # that key to the default monitor without extra trainer-specific code.
        if self.metric_for_best_model is None:
            if scalar_output:
                self.metric_for_best_model = "metric"
            elif len(metrics) == 1:
                self.metric_for_best_model = next(iter(metrics))
            else:
                return
        if self._should_infer_monitor_direction and self.metric_for_best_model is not None:
            if scalar_output and self.metric_for_best_model == "metric":
                self.greater_is_better = False
            else:
                self.greater_is_better = self.infer_greater_is_better(self.metric_for_best_model)
            self._should_infer_monitor_direction = False
            self._sync_early_stopper_mode()

    def _sync_early_stopper_mode(self):
        """Apply the current monitor direction to the attached early stopper."""
        if self.early_stopper is not None and self.greater_is_better is not None:
            self.early_stopper.mode = "max" if self.greater_is_better else "min"

    @staticmethod
    def infer_greater_is_better(metric_name):
        """Infer whether larger metric values indicate better models."""
        metric_name = metric_name.lower()
        if any(loss_name in metric_name for loss_name in ["loss", "mse", "mae", "rmse", "error", "logloss", "log_loss"]):
            return False
        return True
