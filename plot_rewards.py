import os

import matplotlib.pyplot as plt
import pandas as pd


def load_first_valid_csv(log_dir: str) -> pd.DataFrame:
    files = [f for f in os.listdir(log_dir) if f.endswith(".csv")]
    if not files:
        raise FileNotFoundError(f"No CSV files found in {log_dir}")
    files.sort()
    for fname in files:
        path = os.path.join(log_dir, fname)
        df = pd.read_csv(path)
        numeric = df.select_dtypes(include="number")
        if numeric.shape[0] >= 1 and numeric.shape[1] >= 1:
            return df
    raise ValueError(f"No usable CSV with numeric data found in {log_dir}")


def get_timesteps_and_rewards(df: pd.DataFrame):
    if "timesteps" in df.columns and "ep_rew_mean" in df.columns:
        ts = df["timesteps"]
        rew = df["ep_rew_mean"]
    elif {"l", "r"}.issubset(df.columns):
        ts = df["l"].cumsum()
        rew = df["r"]
    else:
        numeric = df.select_dtypes(include="number")
        if numeric.shape[1] < 2:
            raise ValueError(
                f"Could not infer timesteps/reward columns from {df.columns}"
            )
        ts = numeric.iloc[:, 0]
        rew = numeric.iloc[:, 1]
    return ts, rew


def normalize_rewards(rew: pd.Series) -> pd.Series:
    max_abs = rew.abs().max()
    if max_abs == 0:
        return rew
    return rew / max_abs


def main():
    baseline_dir = "logs/baseline"
    shaped_dir = "logs/shaped"

    df_base = load_first_valid_csv(baseline_dir)
    df_shaped = load_first_valid_csv(shaped_dir)

    ts_base, rew_base = get_timesteps_and_rewards(df_base)
    ts_shaped, rew_shaped = get_timesteps_and_rewards(df_shaped)

    # Normalize each curve independently to [-1, 1]
    rew_base_norm = normalize_rewards(rew_base)
    rew_shaped_norm = normalize_rewards(rew_shaped)

    plt.figure(figsize=(8, 5))
    plt.plot(ts_base, rew_base_norm, label="Baseline Reward (normalized)", alpha=0.7)
    plt.plot(ts_shaped, rew_shaped_norm, label="Shaped Reward (normalized)", alpha=0.7)

    plt.xlabel("Timesteps")
    plt.ylabel("Normalized Mean Reward")
    plt.title("Baseline vs Shaped Reward Learning Curves (Normalized)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("reward_comparison.png")


if __name__ == "__main__":
    main()