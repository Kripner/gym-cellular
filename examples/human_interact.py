import gymnasium as gym
import pygame

import gym_cellular

def main():
    env = gym.make('HelicopterCellularAutomaton-v0', render_mode='human')
    obs, _ = env.reset()
    running = True
    clock = pygame.time.Clock()

    # First render() will call pygame.init() and open the window.
    env.render()

    while running:
        action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action = 0  # North
                elif event.key == pygame.K_RIGHT:
                    action = 2  # East
                elif event.key == pygame.K_DOWN:
                    action = 4  # South
                elif event.key == pygame.K_LEFT:
                    action = 6  # West

        if action is not None:
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            if terminated or truncated:
                running = False

        clock.tick(10)  # Limit to 10 FPS

    env.close()

if __name__ == '__main__':
    main()
