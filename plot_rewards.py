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
        # Skip header-only monitor files that only contain meta JSON in first row
        numeric = df.select_dtypes(include="number")
        if numeric.shape[0] >= 1 and numeric.shape[1] >= 1:
            return df
    raise ValueError(f"No usable CSV with numeric data found in {log_dir}")


def get_timesteps_and_rewards(df: pd.DataFrame):
    # Try common SB3 / Monitor patterns first
    if "timesteps" in df.columns and "ep_rew_mean" in df.columns:
        ts = df["timesteps"]
        rew = df["ep_rew_mean"]
    elif {"l", "r"}.issubset(df.columns):
        # SB3 Monitor: l = episode length, r = episode reward
        ts = df["l"].cumsum()
        rew = df["r"]
    else:
        # Fallback: first two numeric columns as (timesteps, reward)
        numeric = df.select_dtypes(include="number")
        if numeric.shape[1] < 2:
            raise ValueError(
                f"Could not infer timesteps/reward columns from {df.columns}"
            )
        ts = numeric.iloc[:, 0]
        rew = numeric.iloc[:, 1]
    return ts, rew


def main():
    baseline_dir = "logs/baseline"
    shaped_dir = "logs/shaped"

    df_base = load_first_valid_csv(baseline_dir)
    df_shaped = load_first_valid_csv(shaped_dir)

    ts_base, rew_base = get_timesteps_and_rewards(df_base)
    ts_shaped, rew_shaped = get_timesteps_and_rewards(df_shaped)

    plt.figure(figsize=(8, 5))
    plt.plot(ts_base, rew_base, label="Baseline Reward", alpha=0.7)
    plt.plot(ts_shaped, rew_shaped, label="Shaped Reward", alpha=0.7)

    plt.xlabel("Timesteps")
    plt.ylabel("Mean Reward")
    plt.title("Baseline vs Shaped Reward Learning Curves")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("reward_comparison.png")


if __name__ == "__main__":
    main()