"""
First runnable MNIST experiment for Hanno / HannoNet1.

Goal
----
This script keeps the same Hanno structure used in the analytical phase but
replaces the optimizee with a small MLP trained on MNIST.

The first proof-of-concept MNIST setup is intentionally modest:
- small MLP
- Adam backbone
- moderate horizon
- HannoNet1 still controls only the effective LR multiplier

Usage
-----
From the project root, run:

    python -m hanno.experiments.run_mnist

Dependencies
------------
This script requires torchvision in addition to torch and numpy:

    pip install torchvision
"""

from __future__ import annotations

from hanno.core.seeding import seed_everything
from hanno.core.types import EpisodeStep, EpisodeTrajectory, StepDiagnostics
from hanno.environment.observation import ObservationBuilder, ObservationConfig
from hanno.environment.reward import WindowedLogImprovementWithInstabilityPenalty
from hanno.environment.training_env import TrainingEnv
from hanno.environment.update_engine import UpdateEngine
from hanno.policy.hanno_net1 import HannoNet1
from hanno.policy.reinforce import REINFORCETrainer
from hanno.tasks.mnist import MnistTask, MnistTaskConfig


def run_episode(env: TrainingEnv, policy: HannoNet1, seed: int) -> tuple[EpisodeTrajectory, dict]:
    """
    Run one full MNIST rollout under the current controller.
    """

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


def main() -> None:
    """
    Assemble and run the first HannoNet1 MNIST experiment.
    """

    seed_everything(42, deterministic=True)
    device = "cpu"

    task = MnistTask(
        config=MnistTaskConfig(
            horizon=100,
            batch_size=64,
            train=True,
            data_root="./data",
            shuffle=True,
            device=device,
        )
    )

    update_engine = UpdateEngine(
        optimizer_name="adam",
        base_lr=1e-3,
    )

    reward_fn = WindowedLogImprovementWithInstabilityPenalty(
        window=3,
        eps=1e-8,
        penalty_weight=0.5,
        instability_threshold=1.5,
    )

    observation_builder = ObservationBuilder(
        config=ObservationConfig(window=3, eps=1e-8)
    )

    env = TrainingEnv(
        task=task,
        update_engine=update_engine,
        reward_fn=reward_fn,
        observation_builder=observation_builder,
        instability_threshold=1.5,
    )

    observation_dim = 10

    policy = HannoNet1(
        observation_dim=observation_dim,
        hidden_dim=64,
        num_layers=2,
        min_lr_multiplier=0.5,
        max_lr_multiplier=1.5,
    ).to(device)

    trainer = REINFORCETrainer(
        policy=policy,
        learning_rate=1e-3,
        gamma=1.0,
        entropy_weight=1e-3,
        normalize_returns=True,
        grad_clip_norm=1.0,
    )

    num_episodes = 50

    for episode_idx in range(1, num_episodes + 1):
        episode_seed = 2000 + episode_idx

        trajectory, episode_info = run_episode(env, policy, seed=episode_seed)
        update_stats = trainer.update(trajectory)
        cumulative_reward = sum(step.reward for step in trajectory.steps)

        print(
            f"[Episode {episode_idx:03d}] "
            f"task={episode_info['task_name']} | "
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


if __name__ == "__main__":
    main()
