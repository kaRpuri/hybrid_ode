import json
import pickle
from pathlib import Path
from typing import Dict, Generator, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from models import HybridODE                     # your model definition

# ----------------------------------------------------------------------------- #
# Utility helpers                                                               #
# ----------------------------------------------------------------------------- #
def load_config(path: str = "config.yaml") -> Dict:
    with open(path, "r") as fp:
        return yaml.safe_load(fp)


def load_test_data(processed_dir: str = "processed_data"
                   ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    proc = Path(processed_dir)
    test_samples = jnp.array(np.load(proc / "test_data.npz")["samples"])
    with open(proc / "normalization_params.json") as fp:
        params = json.load(fp)
    norm = {k: jnp.array(v) for k, v in params.items()}
    return test_samples, norm


def denorm(x: jnp.ndarray, mean: jnp.ndarray, std: jnp.ndarray) -> jnp.ndarray:
    return x * std + mean


def batch_iter(data: jnp.ndarray, batch: int
               ) -> Generator[Tuple[jnp.ndarray, int], None, None]:
    n = data.shape[0]
    for i in range(0, n, batch):
        yield data[i:i + batch], i // batch

# ----------------------------------------------------------------------------- #
# Metric computation                                                            #
# ----------------------------------------------------------------------------- #
STATE_NAMES = ["delta_x", "delta_y", "yaw", "steering",
               "velocity", "side_slip", "yaw_rate"]


def yaw_err(pred, true):
    """Shortest-path angular error."""
    return ((pred - true + jnp.pi) % (2 * jnp.pi)) - jnp.pi


def trajectory_metrics(pred: jnp.ndarray, true: jnp.ndarray) -> Dict:
    mse_tot = float(jnp.mean((pred - true) ** 2))

    mse_state = {}
    for idx, name in enumerate(STATE_NAMES):
        if name == "yaw":
            mse_state[name] = float(jnp.mean(yaw_err(pred[..., idx],
                                                     true[..., idx]) ** 2))
        else:
            mse_state[name] = float(jnp.mean((pred[..., idx] -
                                              true[..., idx]) ** 2))

    pos_err = jnp.sqrt((pred[..., 0] - true[..., 0]) ** 2 +
                       (pred[..., 1] - true[..., 1]) ** 2)
    metrics = {
        "mse_total": mse_tot,
        "mse_per_state": mse_state,
        "position_rmse": float(jnp.sqrt(jnp.mean(pos_err ** 2))),
        "position_mae": float(jnp.mean(pos_err)),
        "final_position_rmse": float(jnp.sqrt(jnp.mean(pos_err[:, -1] ** 2))),
        "final_position_mae": float(jnp.mean(pos_err[:, -1])),
        "velocity_rmse": float(jnp.sqrt(
            jnp.mean((pred[..., 4] - true[..., 4]) ** 2))),
        "velocity_mae": float(jnp.mean(jnp.abs(pred[..., 4] - true[..., 4])))
    }
    return metrics

# ----------------------------------------------------------------------------- #
# Fast CSV writer (vectorised)                                                  #
# ----------------------------------------------------------------------------- #
def save_batch(batch_id: int,
               pred: jnp.ndarray,
               true: jnp.ndarray,
               metrics: Dict,
               t_vec: np.ndarray,
               outdir: Path) -> None:
    """
    Writes a single CSV + JSON for this batch using bulk device→host copy
    and vectorised DataFrame construction.
    """
    outdir.mkdir(exist_ok=True)

    # One device→host copy each (cheap)
    pred_np = np.asarray(pred)        # shape (B, T, 7)
    true_np = np.asarray(true)

    B, T, _ = pred_np.shape
    traj_ids = np.repeat(np.arange(B), T)
    step_ids = np.tile(np.arange(T), B)
    times    = np.tile(t_vec, B)

    flat_pred = pred_np.reshape(-1, 7)
    flat_true = true_np.reshape(-1, 7)
    flat_err  = flat_pred - flat_true
    yaw_idx   = STATE_NAMES.index("yaw")
    flat_err[:, yaw_idx] = ((flat_err[:, yaw_idx] + np.pi) % (2*np.pi)) - np.pi

    df_dict = {
        "batch": batch_id,
        "traj":  traj_ids,
        "step":  step_ids,
        "time":  times,
    }
    for i, name in enumerate(STATE_NAMES):
        df_dict[f"pred_{name}"] = flat_pred[:, i]
        df_dict[f"true_{name}"] = flat_true[:, i]
        df_dict[f"err_{name}"]  = flat_err[:, i]

    pd.DataFrame(df_dict).to_csv(outdir / f"batch_{batch_id}.csv",
                                 index=False)

    # Save batch-level metrics
    with open(outdir / f"batch_{batch_id}_metrics.json", "w") as fp:
        json.dump(metrics, fp, indent=2)

# ----------------------------------------------------------------------------- #
# Main routine                                                                  #
# ----------------------------------------------------------------------------- #
def main(cfg_path: str = "config.yaml") -> None:
    cfg = load_config(cfg_path)
    bs  = cfg["training"]["batch_size"]
    dt  = 0.1                                # adjust if needed

    # --------------------------------------------------------------------- #
    print("Loading test data …")
    test_samples, norm = load_test_data()
    n_steps = test_samples.shape[2]
    t_vec   = np.arange(n_steps) * dt

    # --------------------------------------------------------------------- #
    print("Loading trained parameters …")
    params_path = Path(cfg["data"].get("output_dir", "results")) / \
                  "model_params.pkl"
    if not params_path.exists():
        raise FileNotFoundError(f"model params not found: {params_path}")
    with open(params_path, "rb") as fp:
        params = pickle.load(fp)

    # --------------------------------------------------------------------- #
    model  = HybridODE(cfg)
    outdir = Path("test_results")

    print(f"Running inference on {len(jax.devices())} device(s)…")
    overall = []

    for batch, bid in tqdm(batch_iter(test_samples, bs),
                           total=(test_samples.shape[0] + bs - 1) // bs,
                           desc="Batches"):
        st_dim   = 7
        s0_norm  = batch[:, :st_dim, 0]
        u_norm   = batch[:, st_dim:, :].transpose(0, 2, 1)
        gt_norm  = batch[:, :st_dim, :].transpose(0, 2, 1)

        # Forward pass
        pred_norm = model.predict_batch_trajectories(params, s0_norm,
                                                     u_norm, dt)

        # Denormalise
        pred = denorm(pred_norm, norm["state_mean"], norm["state_std"])
        gt   = denorm(gt_norm,   norm["state_mean"], norm["state_std"])

        # Metrics + save
        m = trajectory_metrics(pred, gt)
        overall.append(m)
        save_batch(bid, pred, gt, m, t_vec, outdir)

        print(f"[Batch {bid}]  total MSE={m['mse_total']:.6f} "
              f"Pos RMSE={m['position_rmse']:.4f}")

    # --------------------------------------------------------------------- #
    # Aggregate statistics
    agg = {}
    for key in overall[0]:
        if key == "mse_per_state":
            agg[key] = {n: float(np.mean([o[key][n] for o in overall]))
                        for n in STATE_NAMES}
        else:
            agg[key] = float(np.mean([o[key] for o in overall]))

    with open(outdir / "overall_statistics.json", "w") as fp:
        json.dump(agg, fp, indent=2)

    print("\nFinished!  Results stored in →", outdir.resolve())

# ----------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
