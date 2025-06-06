import argparse
from pathlib import Path
import json

import gymnasium as gym
import pygame
import numpy as np

import gym_cellular

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_episodes', type=int, default=5)
    parser.add_argument('--experience_path', type=Path)
    return parser

def run_episode(seed: int, experience_path: Path | None) -> float:
    env = gym.make('HelicopterCellularAutomaton-v0', render_mode='human', seed=seed)

    obs, _ = env.reset()
    running = True
    clock = pygame.time.Clock()

    # First render() will call pygame.init() and open the window.
    env.render()

    total_reward = 0
    history = []
    while running:
        action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action = 0  # North
                elif event.key == pygame.K_RIGHT:
                    action = 1  # East
                elif event.key == pygame.K_DOWN:
                    action = 2  # South
                elif event.key == pygame.K_LEFT:
                    action = 3  # West

        if action is not None:
            obs, reward, terminated, truncated, info = env.step(action)
            history.append({
                "obs": {
                    "grid": obs["grid"].tolist(),
                    "position": obs["position"].tolist(),
                },
                "action": action,
                "reward": float(reward),
                "terminated": terminated,
                "truncated": truncated,
            })
            env.render()
            if terminated or truncated:
                running = False
            total_reward += reward
        clock.tick(10)  # Limit to 10 FPS

    if experience_path is not None:
        with open(experience_path, "a") as f:
            f.write(json.dumps(history) + "\n")

    print(f"Total reward: {total_reward}")
    env.close()

    return total_reward

def main(args: argparse.Namespace):
    rng = np.random.default_rng(args.seed)
    rewards = []
    while True:
        seed = rng.integers(0, 1000000)
        reward = run_episode(seed, args.experience_path)
        rewards.append(reward)

        print(f"Reward: {reward}")
        print(f"Mean reward: {np.mean(rewards)}, std: {np.std(rewards)}")
        print("-")
        if len(rewards) >= args.num_episodes:
            break
    print(f"All rewards: [{",".join([str(r) for r in rewards])}]")

if __name__ == '__main__':
    main(get_parser().parse_args())
