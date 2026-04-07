"""
Sequential analytical curriculum runner for Hanno / HannoNet1.

Purpose
-------
This script extends the first analytical runner into a simple curriculum-style
experiment:

1. Train the same HannoNet1 controller sequentially on:
   - isotropic quadratic
   - anisotropic quadratic
   - Rosenbrock

2. Freeze the learned controller and evaluate it on a held-out analytical task:
   - unseen anisotropic quadratic curvature profile

Important conceptual boundary
-----------------------------
During the training phase:
- the optimizee is reset every episode,
- the controller persists across all curriculum stages,
- REINFORCE updates the controller after each episode.

During the evaluation phase:
- the controller is NOT updated,
- the held-out task is used only to test learned behavior.

Usage
-----
From the project root, run:

    python -m hanno.experiments.run_analytical_curriculum
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from hanno.core.seeding import seed_everything
from hanno.core.types import EpisodeStep, EpisodeTrajectory, StepDiagnostics
from hanno.environment.observation import ObservationBuilder, ObservationConfig
from hanno.environment.reward import WindowedLogImprovementWithInstabilityPenalty
from hanno.environment.training_env import TrainingEnv
from hanno.environment.update_engine import UpdateEngine
from hanno.policy.hanno_net1 import HannoNet1
from hanno.policy.reinforce import REINFORCETrainer
from hanno.tasks.analytical import (
    make_anisotropic_quadratic,
    make_isotropic_quadratic,
    make_rosenbrock,
    make_himmelblau,
    make_shifted_quadratic
)


@dataclass
class StageConfig:
    """Configuration bundle for one curriculum stage."""

    name: str
    task_factory: Callable[[], object]
    episodes: int
    optimizer_name: str
    base_lr: float
    min_lr_multiplier: float
    max_lr_multiplier: float


def run_episode(
    env: TrainingEnv,
    policy: HannoNet1,
    seed: int,
) -> tuple[EpisodeTrajectory, dict]:
    """Run one full rollout under the current policy."""

    observation, reset_info = env.reset(seed=seed)
    hidden = policy.init_hidden(batch_size=1, device=observation.device)

    trajectory = EpisodeTrajectory()
    done = False
    final_info = dict(reset_info)

    while not done:
        policy_step = policy.forward_step(observation, hidden)
        hidden = policy_step.hidden

        control = policy.action_mapper.to_control_output(policy_step.raw_action)
        next_observation, reward, done, info = env.step(control)

        last_diag = StepDiagnostics(
            loss=info["loss"],
            grad_norm=info["grad_norm"],
            param_norm=env.trace.param_norms[-1] if env.trace.param_norms else 0.0,
            update_norm=info["update_norm"],
            lr_effective=info["effective_lr"],
            instability_flag=bool(info["instability_flag"]),
            extras={
                "entropy_tensor": policy_step.entropy,
                "mean_tensor": policy_step.mean,
                "std_tensor": policy_step.std,
            },
        )

        step = EpisodeStep(
            observation=observation,
            control=control,
            reward=reward,
            log_prob=policy_step.log_prob,
            diagnostics=last_diag,
        )
        trajectory.append(step)

        observation = next_observation
        final_info = info

    episode_info = {
        "initial_loss": reset_info["initial_loss"],
        "final_loss": final_info.get("loss", None),
        "num_steps": len(trajectory),
        "task_name": final_info.get("task_name", reset_info["task_name"]),
        "mean_effective_lr": (
            sum(env.trace.effective_lrs) / len(env.trace.effective_lrs)
            if env.trace.effective_lrs
            else 0.0
        ),
        "instability_count": sum(1 for flag in env.trace.instability_flags if flag),
    }

    return trajectory, episode_info


#@torch.no_grad()
def evaluate_policy(
    env: TrainingEnv,
    policy: HannoNet1,
    num_episodes: int,
    seed_offset: int = 10000,
) -> None:
    """Evaluate a frozen policy on a held-out task with no REINFORCE updates."""

    policy.eval()
    print("\n=== Held-out evaluation phase ===")

    for episode_idx in range(1, num_episodes + 1):
        seed = seed_offset + episode_idx
        trajectory, episode_info = run_episode(env, policy, seed=seed)
        cumulative_reward = sum(step.reward for step in trajectory.steps)

        print(
            f"[Eval {episode_idx:03d}] "
            f"task={episode_info['task_name']} | "
            f"steps={episode_info['num_steps']} | "
            f"initial_loss={episode_info['initial_loss']:.6f} | "
            f"final_loss={episode_info['final_loss']:.6f} | "
            f"cum_reward={cumulative_reward:.6f} | "
            f"mean_effective_lr={episode_info['mean_effective_lr']:.6f} | "
            f"instability_count={episode_info['instability_count']}"
        )

    policy.train()


def main() -> None:
    """Run the sequential analytical curriculum and held-out evaluation."""

    seed_everything(42, deterministic=True)
    device = "cpu"

    reward_fn = WindowedLogImprovementWithInstabilityPenalty(
        window=3,
        eps=1e-8,
        penalty_weight=0.5,
        instability_threshold=1.5,
    )

    observation_builder = ObservationBuilder(
        config=ObservationConfig(window=3, eps=1e-8)
    )

    observation_dim = 10

    policy = HannoNet1(
        observation_dim=observation_dim,
        hidden_dim=64,
        num_layers=2,
        min_lr_multiplier=0.1,
        max_lr_multiplier=3,
    ).to(device)

    trainer = REINFORCETrainer(
        policy=policy,
        learning_rate=1e-3,
        gamma=1.0,
        entropy_weight=1e-3,
        normalize_returns=True,
        grad_clip_norm=1.0,
    )

    stages = [
        StageConfig(
            name="isotropic_quadratic",
            task_factory=lambda: make_isotropic_quadratic(
                dimension=2,
                curvature=1.0,
                horizon=40,
                init_scale=1.5,
                device=device,
            ),
            episodes=50,
            optimizer_name="sgd",
            base_lr=0.001,
            min_lr_multiplier=0.1,
            max_lr_multiplier=3.0,
        ),
        StageConfig(
            name="anisotropic_quadratic",
            task_factory=lambda: make_anisotropic_quadratic(
                diagonal_values=[1.0, 10.0],
                horizon=40,
                init_scale=1.5,
                device=device,
            ),
            episodes=50,
            optimizer_name="sgd",
            base_lr=0.03,
            min_lr_multiplier=0.1,
            max_lr_multiplier=3.0,
        ),
        StageConfig(
            name="rosenbrock",
            task_factory=lambda: make_rosenbrock(
                horizon=50,
                init_scale=1.2,
                device=device,
            ),
            episodes=200,
            optimizer_name="adam",
            base_lr=0.01,
            min_lr_multiplier=0.1,
            max_lr_multiplier=3,
        ),
    ]

    global_episode_idx = 0

    for stage in stages:
        print(f"\n=== Training stage: {stage.name} ===")

        policy.action_mapper.min_lr_multiplier = stage.min_lr_multiplier
        policy.action_mapper.max_lr_multiplier = stage.max_lr_multiplier

        task = stage.task_factory()
        update_engine = UpdateEngine(
            optimizer_name=stage.optimizer_name,
            base_lr=stage.base_lr,
        )
        env = TrainingEnv(
            task=task,
            update_engine=update_engine,
            reward_fn=reward_fn,
            observation_builder=observation_builder,
            instability_threshold=1.5,
        )

        for _ in range(stage.episodes):
            global_episode_idx += 1
            seed = 1000 + global_episode_idx

            trajectory, episode_info = run_episode(env, policy, seed=seed)
            update_stats = trainer.update(trajectory)
            cumulative_reward = sum(step.reward for step in trajectory.steps)

            print(
                f"[Train {global_episode_idx:03d}] "
                f"stage={stage.name} | "
                f"steps={episode_info['num_steps']} | "
                f"initial_loss={episode_info['initial_loss']:.6f} | "
                f"final_loss={episode_info['final_loss']:.6f} | "
                f"cum_reward={cumulative_reward:.6f} | "
                f"mean_return={update_stats.mean_return:.6f} | "
                f"policy_loss={update_stats.policy_loss:.6f} | "
                f"total_loss={update_stats.total_loss:.6f} | "
                f"mean_effective_lr={episode_info['mean_effective_lr']:.6f} | "
                f"instability_count={episode_info['instability_count']}"
            )

    # ------------------------------------------------------------------
    # Held-out analytical evaluation suite
    # ------------------------------------------------------------------

    # 1. Held-out convex test: unseen anisotropic quadratic
    heldout_task_1 = make_anisotropic_quadratic(
        diagonal_values=[1.0, 25.0],
        horizon=40,
        init_scale=1.5,
        device=device,
    )
    heldout_update_engine_1 = UpdateEngine(
        optimizer_name="sgd",
        base_lr=0.03,
    )
    heldout_env_1 = TrainingEnv(
        task=heldout_task_1,
        update_engine=heldout_update_engine_1,
        reward_fn=reward_fn,
        observation_builder=observation_builder,
        instability_threshold=1.5,
    )

    # Quadratic-style control range
    policy.action_mapper.min_lr_multiplier = 0.1
    policy.action_mapper.max_lr_multiplier = 3.0

    print("\\n=== Held-out test 1: unseen anisotropic quadratic ===")
    evaluate_policy(
        env=heldout_env_1,
        policy=policy,
        num_episodes=50,
        seed_offset=50000,
    )

    # 2. Held-out convex test: shifted quadratic
    heldout_task_2 = make_shifted_quadratic(
        center=[2.0, -1.0],
        diagonal_values=[1.0, 8.0],
        horizon=40,
        init_scale=1.5,
        device=device,
    )
    heldout_update_engine_2 = UpdateEngine(
        optimizer_name="sgd",
        base_lr=0.03,
    )
    heldout_env_2 = TrainingEnv(
        task=heldout_task_2,
        update_engine=heldout_update_engine_2,
        reward_fn=reward_fn,
        observation_builder=observation_builder,
        instability_threshold=1.5,
    )

    policy.action_mapper.min_lr_multiplier = 0.1
    policy.action_mapper.max_lr_multiplier = 3.0

    print("\\n=== Held-out test 2: shifted quadratic ===")
    evaluate_policy(
        env=heldout_env_2,
        policy=policy,
        num_episodes=50,
        seed_offset=60000,
    )

    # 3. Held-out non-convex test: Himmelblau
    heldout_task_3 = make_himmelblau(
        horizon=60,
        init_scale=1.5,
        device=device,
    )
    heldout_update_engine_3 = UpdateEngine(
        optimizer_name="adam",
        base_lr=0.01,
    )
    heldout_env_3 = TrainingEnv(
        task=heldout_task_3,
        update_engine=heldout_update_engine_3,
        reward_fn=reward_fn,
        observation_builder=observation_builder,
        instability_threshold=1.5,
    )

    # Safer non-convex control range, similar in spirit to Rosenbrock
    policy.action_mapper.min_lr_multiplier = 0.1
    policy.action_mapper.max_lr_multiplier = 3

    print("\\n=== Held-out test 3: Himmelblau ===")
    evaluate_policy(
        env=heldout_env_3,
        policy=policy,
        num_episodes=50,
        seed_offset=70000,
    )

if __name__ == "__main__":
    main()
