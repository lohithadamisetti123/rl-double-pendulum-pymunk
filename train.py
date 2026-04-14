import argparse
import os

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

from environment import DoublePendulumEnv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward_type", type=str, default="shaped", choices=["baseline", "shaped"])
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--save_path", type=str, default="models/ppo_model.zip")
    parser.add_argument("--log_dir", type=str, default="logs/run")
    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    env = DoublePendulumEnv(reward_type=args.reward_type, render_mode=None)
    env = Monitor(env, args.log_dir)

    model = PPO("MlpPolicy", env, verbose=1)

    new_logger = configure(args.log_dir, ["stdout", "csv"])
    model.set_logger(new_logger)

    model.learn(total_timesteps=args.timesteps)

    model.save(args.save_path)
    env.close()


if __name__ == "__main__":
    main()