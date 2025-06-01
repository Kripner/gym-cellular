#!/usr/bin/env python3
import argparse
import os
import csv
from pathlib import Path
import random
import json
import re
import datetime
import time

import numpy as np
import pygame
import gymnasium as gym

from gym_cellular.cellular.forest_fire import ForestFire
from gym_cellular.agent.planner import (
    OracleWorldModel,
    StaticWorldModel,
    PlanningAgent,
)
from gym_cellular.environment.helicopter_env import HelicopterEnv


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_id", type=str, default="HelicopterCellularAutomaton-v0")
    parser.add_argument("--depth", type=int, default=5, help="Depth of tree search for the PlanningAgent.")
    parser.add_argument("--world_model", type=str, choices=["oracle", "static"], default="oracle")
    parser.add_argument("--n_runs", type=int, default=5, help="Number of independent episodes to average over.")
    parser.add_argument("--render_mode", type=str, choices=["human", "rgb_array", "none"], default="none")
    parser.add_argument("--base_dir", type=Path, default="out")
    parser.add_argument("--seed", type=int, default=0)

    return parser


ARGS_WHITELIST = [
    "depth", "world_model", "n_runs", "seed",
]


def run_one_episode(
        env_id: str,
        depth: int,
        world_model: str,
        render_mode: str,
        seed: int,
) -> int:
    """
    Runs a single episode in `env_id` using a depth‐d planning agent
    backed by either the Oracle or Static world model. Returns the final
    count of cells == 1 (“trees”) at episode end.
    """
    # Create env via gym.make → this also registers and returns a HelicopterEnv
    env = gym.make(env_id, render_mode=render_mode, seed=seed)
    obs, _ = env.reset()

    # We need direct access to the underlying HelicopterEnv instance to grab:
    #   - automaton (for oracle model)
    #   - agent_pos (for planning recursion)
    wrapped: HelicopterEnv = env.unwrapped  # type: ignore

    height = wrapped.height
    width = wrapped.width

    # Build the chosen world model:
    if world_model == "oracle":
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
    planner = PlanningAgent(
        depth=depth,
        world_model=world_model,
        height=height,
        width=width,
    )

    terminated = False
    truncated = False

    # Force an initial render so that pygame’s video system is initialized
    if render_mode == "human":
        wrapped.render()

    agent_time = 0
    agent_interactions = 0

    while not (terminated or truncated):
        # TODO: instead, we should use the observation from the environment
        current_grid = wrapped.automaton.get_state()
        agent_pos = tuple(wrapped.agent_pos)

        start_time = time.time()
        action = planner.select_action(current_grid, agent_pos)
        agent_time += time.time() - start_time
        agent_interactions += 1

        obs, reward, terminated, truncated, _ = env.step(action)

        if render_mode == "human":
            wrapped.render()
            # Essential to call `pygame.event.pump()` so the window stays responsive
            pygame.event.pump()

    # At episode end, count how many “trees” (state==1) remain
    final_grid = wrapped.automaton.get_state()
    final_trees = int(np.sum(final_grid == ForestFire.TREE))

    env.close()
    if render_mode == "human":
        pygame.quit()

    print(f"Agent time: {agent_time:.2f}s (avg {agent_time / agent_interactions:.2f}s per interaction)")

    return final_trees


def get_args_descriptor(
        args_ns: argparse.Namespace,
        param_whitelist: list[str] | None = None,
        include_slurm_id=True,
        include_time=True,
) -> str:
    args = vars(args_ns)
    if include_time:
        descriptor = datetime.datetime.now().strftime("%y-%m-%d_%H%M%S")
    else:
        descriptor = ""

    if include_slurm_id and "SLURM_JOB_ID" in os.environ:
        if len(descriptor) > 0:
            descriptor += "-"
        descriptor += f"id={os.environ['SLURM_JOB_ID']}"

    visible_args = {k: v for k, v in sorted(args.items())}
    if param_whitelist is not None:
        visible_args = {k: v for k, v in visible_args.items() if k in param_whitelist}

    def format_value(v: str) -> str:
        if isinstance(v, Path) or "/" in str(v):
            v = str(v)
            if v.endswith("/"):
                v = v[:-1]
            parts = [p for p in v.split("/") if len(p) != 0]
            return "_".join([v[:50] for v in parts[-2:]])
        if isinstance(v, str):
            return v.replace("<", "").replace(">", "")
        return str(v)

    if len(visible_args) > 0:
        if len(descriptor) > 0:
            descriptor += "-"
        descriptor += ",".join((
            "{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), format_value(v))
            for k, v in visible_args.items()
        ))

    assert len(descriptor) > 0
    return descriptor


def dump_args(args, logdir):
    path = os.path.join(logdir, "args.json")
    with open(path, "w") as f:
        data = {k: str(v) for k, v in args.__dict__.items()}
        json.dump(data, f, indent=4, sort_keys=True)
        f.write("\n")


def setup_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def main(args: argparse.Namespace):
    setup_seeds(args.seed)

    descriptor = get_args_descriptor(args, param_whitelist=set(ARGS_WHITELIST))
    log_dir = args.base_dir / descriptor
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"Logging to {log_dir}")
    dump_args(args, log_dir)

    os.makedirs(log_dir, exist_ok=True)
    csv_path = log_dir / "results.csv"

    # Header: run_index, final_tree_count
    with open(csv_path, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["run_index", "final_trees"])

        totals = []
        rng = np.random.RandomState(args.seed)
        for run_idx in range(1, args.n_runs + 1):
            final_count = run_one_episode(
                env_id=args.env_id,
                depth=args.depth,
                world_model=args.world_model,
                render_mode=args.render_mode,
                seed=rng.randint(0, 2**32),
            )
            writer.writerow([run_idx, final_count])
            totals.append(final_count)
            print(f"Run {run_idx}/{args.n_runs} → final trees = {final_count}")

        average_trees = sum(totals) / len(totals)
        summary_path = log_dir / "summary.txt"
        with open(summary_path, mode="w") as f:
            f.write(f"Depth = {args.depth}\n")
            f.write(f"Model = {args.world_model}\n")
            f.write(f"n_runs = {args.n_runs}\n")
            f.write(f"Average final trees = {average_trees:.2f}\n")

    print(f"\nResults written to:\n  {csv_path}\n  {summary_path}")


if __name__ == "__main__":
    main(get_parser().parse_args())
