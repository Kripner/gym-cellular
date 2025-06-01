#!/usr/bin/env python3
import argparse
import os
import csv
from pathlib import Path

import numpy as np
import pygame
import gymnasium as gym

from gym_cellular.agent.planner import (
    OracleWorldModel,
    StaticWorldModel,
    PlanningAgent,
)
from gym_cellular.environment.helicopter_env import HelicopterEnv

def get_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate depth‐d planning agent on HelicopterEnv."
    )
    parser.add_argument(
        "--env_id",
        type=str,
        default="HelicopterCellularAutomaton-v0",
        help="Registered Gymnasium ID of the environment to evaluate."
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=3,
        help="Depth of tree search for the PlanningAgent."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["oracle", "static"],
        default="oracle",
        help="Whether to use the OracleWorldModel or StaticWorldModel."
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=5,
        help="Number of independent episodes to average over."
    )
    parser.add_argument(
        "--render_mode",
        type=str,
        choices=["human", "rgb_array", "none"],
        default="none",
        help="gym.make render_mode to pass to the environment."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default="out",
        help="Directory to write per-run metrics (CSV) and summary."
    )

    return parser


def run_one_episode(
        env_id: str,
        depth: int,
        model_type: str,
        render_mode: str,
) -> int:
    """
    Runs a single episode in `env_id` using a depth‐d planning agent
    backed by either the Oracle or Static world model. Returns the final
    count of cells == 1 (“trees”) at episode end.
    """
    # Create env via gym.make → this also registers and returns a HelicopterEnv
    env = gym.make(env_id, render_mode=render_mode)
    obs, _ = env.reset()

    # We need direct access to the underlying HelicopterEnv instance to grab:
    #   - automaton (for oracle model)
    #   - agent_pos (for planning recursion)
    wrapped: HelicopterEnv = env.unwrapped  # type: ignore

    height = wrapped.height
    width = wrapped.width

    # Build the chosen world model:
    if model_type == "oracle":
        # Build a fresh automaton of the same class and dims:
        base_auto = wrapped.automaton
        automaton_cls = base_auto.__class__
        # Instantiate with default args (state will be overwritten inside OracleWorldModel)
        # For both GameOfLife and ForestFire, __init__(width, height, ...) is okay
        oracle_auto = automaton_cls(width, height)  # (init_random=whatever)
        world_model = OracleWorldModel(oracle_auto)
    else:
        world_model = StaticWorldModel()

    # Instantiate the planner:
    planner = PlanningAgent(depth=depth,
                            world_model=world_model,
                            height=height,
                            width=width)

    terminated = False
    truncated = False

    # Force an initial render so that pygame’s video system is initialized
    if render_mode == "human":
        wrapped.render()

    while not (terminated or truncated):
        current_grid = wrapped.automaton.get_state()
        agent_pos = tuple(wrapped.agent_pos)

        # Ask planner for the best action
        action = planner.select_action(current_grid, agent_pos)

        obs, reward, terminated, truncated, _ = env.step(action)

        if render_mode == "human":
            wrapped.render()
            # Essential to call `pygame.event.pump()` so the window stays responsive
            pygame.event.pump()

    # At episode end, count how many “trees” (state==1) remain
    final_grid = wrapped.automaton.get_state()
    final_trees = int(np.sum(final_grid == 1))

    env.close()
    if render_mode == "human":
        pygame.quit()

    return final_trees


def main(args: argparse.Namespace):
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "results.csv")

    # Header: run_index, final_tree_count
    with open(csv_path, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["run_index", "final_trees"])

        totals = []
        for run_idx in range(1, args.n_runs + 1):
            final_count = run_one_episode(
                env_id=args.env_id,
                depth=args.depth,
                model_type=args.model_type,
                render_mode=args.render_mode
            )
            writer.writerow([run_idx, final_count])
            totals.append(final_count)
            print(f"Run {run_idx}/{args.n_runs} → final trees = {final_count}")

        average_trees = sum(totals) / len(totals)
        summary_path = os.path.join(args.output_dir, "summary.txt")
        with open(summary_path, mode="w") as f:
            f.write(f"Depth = {args.depth}\n")
            f.write(f"Model = {args.model_type}\n")
            f.write(f"n_runs = {args.n_runs}\n")
            f.write(f"Average final trees = {average_trees:.2f}\n")

    print(f"\nResults written to:\n  {csv_path}\n  {summary_path}")


if __name__ == "__main__":
    main(get_parser().parse_args())
