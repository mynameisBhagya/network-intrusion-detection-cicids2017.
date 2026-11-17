"""
Classifiers Suite for Network Intrusion Detection (CIC-IDS2017).
Implements the 5 empirical classifiers evaluated in Deliverable 1:
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. XGBoost
5. Support Vector Machine (Linear SVM with probability calibration for scalability)
"""

import logging
from typing import Dict, Any, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb

logger = logging.getLogger(__name__)


def get_classifier(
    classifier_name: str,
    class_weight: Optional[str] = None,
    random_state: int = 42,
    is_multiclass: bool = False,
    **kwargs
):
    """
    Factory function instantiating standardized ML classifiers under uniform evaluation constraints.

    Args:
        classifier_name: 'logistic_regression', 'decision_tree', 'random_forest', 'xgboost', 'svm'
        class_weight: 'balanced' or None (for handling severe class imbalance)
        random_state: Fixed random seed for deterministic reproducibility
        is_multiclass: True if multiclass classification, False for binary
        **kwargs: Model-specific hyperparameter overrides

    Returns:
        BaseEstimator: Configured scikit-learn or XGBoost classifier
    """
    name = classifier_name.lower().replace(" ", "_").replace("-", "_")

    if name in ["logistic_regression", "lr"]:
        params = {
            "max_iter": 1000,
            "class_weight": class_weight,
            "random_state": random_state,
            "solver": "lbfgs"
        }
        params.update(kwargs)
        return LogisticRegression(**params)

    elif name in ["decision_tree", "dt"]:
        params = {
            "max_depth": 15,
            "min_samples_split": 10,
            "class_weight": class_weight,
            "random_state": random_state
        }
        params.update(kwargs)
        return DecisionTreeClassifier(**params)

    elif name in ["random_forest", "rf"]:
        params = {
            "n_estimators": 100,
            "max_depth": 20,
            "min_samples_split": 5,
            "class_weight": class_weight,
            "random_state": random_state,
            "n_jobs": -1
        }
        params.update(kwargs)
        return RandomForestClassifier(**params)

    elif name in ["xgboost", "xgb"]:
        objective = "multi:softprob" if is_multiclass else "binary:logistic"
        eval_metric = "mlogloss" if is_multiclass else "logloss"
        params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "objective": objective,
            "eval_metric": eval_metric,
            "random_state": random_state,
            "tree_method": "hist",
            "n_jobs": -1
        }
        params.update(kwargs)
        return xgb.XGBClassifier(**params)

    elif name in ["svm", "support_vector_machine", "linear_svm"]:
        # LinearSVC provides linear SVM performance scaling to massive tabular flows.
        # CalibratedClassifierCV equips it with well-calibrated predict_proba() for ROC/PR metrics.
        base_svm = LinearSVC(
            dual=False,
            max_iter=2000,
            class_weight=class_weight,
            random_state=random_state
        )
        return CalibratedClassifierCV(estimator=base_svm, cv=3)

    else:
        raise ValueError(
            f"Unknown classifier '{classifier_name}'. Supported options: "
            f"['logistic_regression', 'decision_tree', 'random_forest', 'xgboost', 'svm']"
        )


def get_all_classifiers(
    class_weight: Optional[str] = None,
    random_state: int = 42,
    is_multiclass: bool = False
) -> Dict[str, Any]:
    """Retrieve instances of all 5 classifiers configured under identical protocols."""
    models = {
        "Logistic Regression": get_classifier("lr", class_weight, random_state, is_multiclass),
        "Decision Tree": get_classifier("dt", class_weight, random_state, is_multiclass),
        "Random Forest": get_classifier("rf", class_weight, random_state, is_multiclass),
        "XGBoost": get_classifier("xgb", class_weight, random_state, is_multiclass),
        "SVM": get_classifier("svm", class_weight, random_state, is_multiclass)
    }
    return models
