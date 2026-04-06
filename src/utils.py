import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel


def to_dense(x):
    if hasattr(x, "toarray"):
        return x.toarray()
    return x


def score_split(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
    }


def load_train_val_test(loader):
    return loader("train"), loader("val"), loader("test")


def merge_train_val(train_df: pd.DataFrame, val_df: pd.DataFrame):
    return pd.concat([train_df, val_df], ignore_index=True)


def build_submission(case_id_series, predicted_labels, target_col: str = "Overall_class"):
    return pd.DataFrame(
        {
            "case_id": case_id_series.astype(str),
            target_col: predicted_labels,
        }
    )


def manual_bayes_optimize(objective, bounds, n_trials: int, n_init: int, seed: int, candidate_pool: int = 384):
    keys = list(bounds.keys())
    low = np.array([float(bounds[k][0]) for k in keys], dtype=np.float64)
    high = np.array([float(bounds[k][1]) for k in keys], dtype=np.float64)
    dim = len(keys)
    rng = np.random.default_rng(seed)
    n_trials = int(max(1, n_trials))
    n_init = int(max(1, min(n_init, n_trials)))
    x_hist = []
    y_hist = []

    def _eval_point(x):
        params = {k: float(v) for k, v in zip(keys, x)}
        score = float(objective(**params))
        x_hist.append(np.asarray(x, dtype=np.float64))
        y_hist.append(score)

    for _ in range(n_init):
        x0 = rng.uniform(low, high)
        _eval_point(x0)

    for step in range(n_trials - n_init):
        x_arr = np.vstack(x_hist)
        y_arr = np.asarray(y_hist, dtype=np.float64)
        y_mean = float(np.mean(y_arr))
        y_std = float(np.std(y_arr))
        if y_std < 1e-8:
            y_std = 1.0
        y_scaled = (y_arr - y_mean) / y_std

        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
            length_scale=np.ones(dim, dtype=np.float64),
            length_scale_bounds=(1e-2, 1e2),
            nu=2.5,
        ) + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-8, 1e-1))

        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            normalize_y=False,
            random_state=seed + step,
            n_restarts_optimizer=1,
        )

        use_gp = True
        try:
            gp.fit(x_arr, y_scaled)
        except Exception:
            use_gp = False

        x_cand = rng.uniform(low, high, size=(int(max(32, candidate_pool)), dim))
        if use_gp:
            mu, sigma = gp.predict(x_cand, return_std=True)
            acq = mu + 2.0 * sigma
            pick = int(np.argmax(acq))
            x_next = x_cand[pick]
        else:
            x_next = x_cand[int(rng.integers(0, len(x_cand)))]
        _eval_point(x_next)

    return {
        "x_history": np.vstack(x_hist),
        "y_history": np.asarray(y_hist, dtype=np.float64),
        "best_score": float(np.max(y_hist)),
    }
