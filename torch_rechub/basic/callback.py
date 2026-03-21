import copy


class EarlyStopper(object):
    """Stop training when a monitored score fails to improve.

    Args:
        patience (int): Number of consecutive validation calls to wait.
        mode (str): Whether a larger score is better (``"max"``) or a
            smaller score is better (``"min"``).
        delta (float): Minimum score change required to count as improvement.
    """

    def __init__(self, patience, mode="max", delta=0.0):
        if mode not in {"max", "min"}:
            raise ValueError(f"mode must be 'max' or 'min', got {mode!r}")
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.trial_counter = 0
        self.best_score = None
        # Keep ``best_auc`` for backward compatibility with existing callers.
        self.best_auc = None
        self.best_weights = None

    def stop_training(self, score, weights):
        """Check whether training should stop.

        Args:
            score (float): Validation score in the monitored direction.
            weights (tensor): Model weights snapshot.
        """
        if self._is_improved(score):
            self.best_score = score
            self.best_auc = score
            self.trial_counter = 0
            self.best_weights = copy.deepcopy(weights)
            return False
        elif self.trial_counter + 1 < self.patience:
            self.trial_counter += 1
            return False
        else:
            return True

    def _is_improved(self, score):
        """Return whether ``score`` improves the current best checkpoint."""
        if self.best_score is None:
            return True
        if self.mode == "max":
            return score > self.best_score + self.delta
        return score < self.best_score - self.delta
