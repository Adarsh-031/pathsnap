"""CPU baselines with fixed configurations for the first model comparison."""

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def logistic_regression(seed: int):
    return make_pipeline(
        StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, random_state=seed),
    )


def candidate_models(seed: int, jobs: int) -> dict:
    return {
        "lightgbm": LGBMClassifier(
            n_estimators=200, learning_rate=0.05, num_leaves=31,
            random_state=seed, n_jobs=jobs, verbosity=-1,
        ),
        "catboost": CatBoostClassifier(
            iterations=200, learning_rate=0.05, depth=6, loss_function="Logloss",
            random_seed=seed, thread_count=jobs, verbose=False, allow_writing_files=False,
        ),
        "xgboost": XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=6, tree_method="hist",
            eval_metric="logloss", random_state=seed, n_jobs=jobs,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=150, max_depth=18, min_samples_leaf=5,
            random_state=seed, n_jobs=jobs,
        ),
        "logistic_regression": logistic_regression(seed),
    }
