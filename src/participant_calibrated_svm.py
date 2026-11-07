"""Participant-Grouped Calibrated SVM Module.

Implements mathematically leak-free probability calibration for Support Vector Machines
operating on multi-epoch participant-grouped biomedical data.

Key Invariant:
Standard `sklearn.svm.SVC(probability=True)` performs row-level random 5-fold CV
for Platt scaling sigmoid fitting. When multiple temporal segments originate from
the same participant, standard Platt scaling leaks segments from the same participant
across calibration folds.

`ParticipantCalibratedSVM` resolves this by:
1. Fitting base SVM on training participants.
2. Generating Platt scaling calibration splits strictly partitioned by participant ID
   using `StratifiedGroupKFold`.
3. Ensuring zero intra-participant leakage between hyperplane fitting and calibration fitting.
4. Guaranteeing that held-out validation or test participants are never exposed to calibration.
"""

import warnings
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import SVC, LinearSVC

warnings.filterwarnings("ignore", category=FutureWarning)


class ParticipantCalibratedSVM(BaseEstimator, ClassifierMixin):
    """
    Support Vector Machine with Participant-Grouped Platt Scaling.
    
    Parameters
    ----------
    kernel : str, default='linear'
        SVM kernel ('linear', 'rbf', 'poly').
    C : float, default=1.0
        Regularization parameter.
    gamma : str or float, default='scale'
        Kernel coefficient for 'rbf' and 'poly'.
    degree : int, default=3
        Degree for 'poly' kernel.
    n_calib_splits : int, default=5
        Number of participant-grouped calibration folds.
    random_state : int, default=42
        Random seed for reproducibility.
    """
    
    def __init__(
        self,
        kernel="linear",
        C=1.0,
        gamma="scale",
        degree=3,
        n_calib_splits=5,
        random_state=42,
    ):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.n_calib_splits = n_calib_splits
        self.random_state = random_state
        self.calibrated_classifier_ = None
        self.classes_ = None

    def fit(self, X, y, groups=None):
        """
        Fit the participant-grouped calibrated SVM.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y : array-like of shape (n_samples,)
            Target labels (0 or 1).
        groups : array-like of shape (n_samples,)
            Participant identifiers. If None, raises ValueError.
        """
        if groups is None:
            raise ValueError(
                "ParticipantCalibratedSVM requires `groups` (participant IDs) "
                "to guarantee leakage-free participant-grouped calibration."
            )
            
        X = np.asarray(X)
        y = np.asarray(y)
        groups = np.asarray(groups)
        
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("ParticipantCalibratedSVM currently supports binary classification.")
            
        # Determine number of calibration splits based on unique participants
        n_unique_groups = len(np.unique(groups))
        n_splits = min(self.n_calib_splits, n_unique_groups)
        if n_splits < 2:
            raise ValueError(f"Need at least 2 participant groups for calibration, got {n_unique_groups}")
            
        # Generate participant-grouped calibration splits
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        cv_splits = list(sgkf.split(X, y, groups=groups))
        
        # Verify zero group leakage across calibration folds
        for fold_idx, (train_idx, calib_idx) in enumerate(cv_splits):
            train_groups = set(groups[train_idx])
            calib_groups = set(groups[calib_idx])
            overlap = train_groups.intersection(calib_groups)
            if overlap:
                raise RuntimeError(
                    f"Fatal Leakage Error in Calibration Fold {fold_idx}: "
                    f"Participants {overlap} present in both train and calibration sets!"
                )

        # Base estimator without internal row-level probability fitting
        base_estimator = SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            degree=self.degree,
            probability=False,
            random_state=self.random_state,
        )
        
        # Fit Platt scaling calibration using the participant-grouped splits
        self.calibrated_classifier_ = CalibratedClassifierCV(
            estimator=base_estimator,
            method="sigmoid",
            cv=cv_splits,
        )
        self.calibrated_classifier_.fit(X, y)
        return self

    def predict_proba(self, X):
        """Return calibrated posterior probabilities."""
        if self.calibrated_classifier_ is None:
            raise RuntimeError("Model is not fitted yet.")
        return self.calibrated_classifier_.predict_proba(X)

    def predict(self, X):
        """Predict class labels for samples in X."""
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)

    def decision_function(self, X):
        """Return logit of positive class posterior as continuous score."""
        probs = self.predict_proba(X)
        eps = 1e-15
        p1 = np.clip(probs[:, 1], eps, 1.0 - eps)
        return np.log(p1 / (1.0 - p1))
