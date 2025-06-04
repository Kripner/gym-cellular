import argparse

import gymnasium as gym
import pygame

import gym_cellular

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    return parser

def main(args: argparse.Namespace):
    env = gym.make('HelicopterCellularAutomaton-v0', render_mode='human', seed=args.seed)
    obs, _ = env.reset()
    running = True
    clock = pygame.time.Clock()

    # First render() will call pygame.init() and open the window.
    env.render()

    total_reward = 0
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
            env.render()
            if terminated or truncated:
                running = False
            total_reward += reward
        clock.tick(10)  # Limit to 10 FPS

    print(f"Total reward: {total_reward}")
    env.close()

if __name__ == '__main__':
    main(get_parser().parse_args())
