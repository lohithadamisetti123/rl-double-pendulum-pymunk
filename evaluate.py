import argparse
import os

import imageio
import numpy as np
import pygame
from stable_baselines3 import PPO

from environment import DoublePendulumEnv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--record_gif", type=str, default="False")
    parser.add_argument("--gif_path", type=str, default="media/agent_run.gif")
    return parser.parse_args()


def str2bool(v: str) -> bool:
    return v.lower() in ("1", "true", "yes", "y")


def main():
    args = parse_args()
    record_gif = str2bool(args.record_gif)

    if record_gif:
        os.makedirs(os.path.dirname(args.gif_path), exist_ok=True)

    env = DoublePendulumEnv(reward_type="shaped", render_mode="human")

    model = PPO.load(args.model_path, env=env)

    frames = []

    for _ in range(args.episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()

            if record_gif:
                surface = pygame.display.get_surface()
                frame = pygame.surfarray.array3d(surface)
                frame = np.rot90(frame, k=3)
                frame = np.fliplr(frame)
                frames.append(frame)

    env.close()

    if record_gif and frames:
        imageio.mimsave(args.gif_path, frames, fps=30)


if __name__ == "__main__":
    main()