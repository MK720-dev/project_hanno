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

from dataclasses import dataclass

from hanno.core.seeding import seed_everything
from hanno.core.types import EpisodeStep, EpisodeTrajectory, StepDiagnostics
from hanno.environment.observation import ObservationBuilder, ObservationConfig
from hanno.environment.reward import WindowedLogImprovementWithInstabilityPenalty
from hanno.environment.training_env import TrainingEnv
from hanno.environment.update_engine import UpdateEngine
from hanno.policy.hanno_net1 import HannoNet1
from hanno.policy.reinforce import REINFORCETrainer
from hanno.tasks.mnist import MnistTask, MnistTaskConfig


@dataclass(frozen=True)
class ResourceSchedule:
    """
    Lightweight training-budget preset for MNIST transfer experiments.
    """

    name: str
    horizon: int
    num_episodes: int


@dataclass(frozen=True)
class MnistTransferExperiment:
    """
    One MNIST experiment bundle containing a training family, a held-out family,
    and the episode-budget settings used for the run.
    """

    name: str
    train_variants: tuple[str, ...]
    eval_variants: tuple[str, ...]
    resource_schedule: ResourceSchedule


DEFAULT_RESOURCE_SCHEDULES: tuple[ResourceSchedule, ...] = (
    ResourceSchedule(name="short_low", horizon=75, num_episodes=30),
    ResourceSchedule(name="medium_mid", horizon=100, num_episodes=50),
    ResourceSchedule(name="long_high", horizon=150, num_episodes=80),
)


DEFAULT_EXPERIMENTS: tuple[MnistTransferExperiment, ...] = (
    MnistTransferExperiment(
        name="mlp_transfer_baseline",
        train_variants=("small", "wide", "deep"),
        eval_variants=("narrow", "bottleneck", "very_wide"),
        resource_schedule=DEFAULT_RESOURCE_SCHEDULES[1],
    ),
    MnistTransferExperiment(
        name="mlp_transfer_long_horizon",
        train_variants=("small", "wide", "deep"),
        eval_variants=("narrow", "bottleneck", "very_wide"),
        resource_schedule=DEFAULT_RESOURCE_SCHEDULES[2],
    ),
)


def run_episode(
    env: TrainingEnv,
    policy: HannoNet1,
    seed: int,
    expected_variant: str,
) -> tuple[EpisodeTrajectory, dict]:
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

    task = env.task
    resolved_variant = getattr(getattr(task, "config",None), "model_variant", None) 
    model_class = None
    try:
        model = env.state.parameters
        model_class = type(model).__name__ if model is not None else None
    except Exception:
        model_class = None
        
   
    episode_info = {
        "initial_loss": reset_info["initial_loss"],
        "final_loss": final_info.get("loss", None),
        "num_steps": len(trajectory),
        "task_name": final_info.get("task_name", reset_info["task_name"]),
        "expected_variant": expected_variant,
        "model_variant": resolved_variant,
        "model_class": model_class,
        "variant_match": resolved_variant == expected_variant,
        "mean_effective_lr": (
            sum(env.trace.effective_lrs) / len(env.trace.effective_lrs)
            if env.trace.effective_lrs
            else 0.0
        ),
        "instability_count": sum(1 for flag in env.trace.instability_flags if flag),
    }

    return trajectory, episode_info


def build_env(device: str, model_variant: str, horizon: int) -> TrainingEnv:
    """
    Assemble one MNIST environment instance for the requested architecture and
    resource setting.
    """

    task = MnistTask(
        config=MnistTaskConfig(
            horizon=horizon,
            batch_size=64,
            train=True,
            data_root="./data",
            shuffle=True,
            device=device,
            model_variant=model_variant,
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

    return TrainingEnv(
        task=task,
        update_engine=update_engine,
        reward_fn=reward_fn,
        observation_builder=observation_builder,
        instability_threshold=1.5,
    )


def train_transfer_family(
    policy: HannoNet1,
    trainer: REINFORCETrainer,
    device: str,
    experiment: MnistTransferExperiment,
    base_seed: int,
) -> None:
    """
    Train one controller across a rotating family of MNIST MLP architectures.
    """

    schedule = experiment.resource_schedule

    print(
        f"\n[Train] experiment={experiment.name} | "
        f"schedule={schedule.name} | "
        f"horizon={schedule.horizon} | "
        f"num_episodes={schedule.num_episodes} | "
        f"train_variants={','.join(experiment.train_variants)}"
    )

    for episode_idx in range(1, schedule.num_episodes + 1):
        model_variant = experiment.train_variants[(episode_idx - 1) % len(experiment.train_variants)]
        episode_seed = base_seed + episode_idx
        env = build_env(device=device, model_variant=model_variant, horizon=schedule.horizon)

        trajectory, episode_info = run_episode(
            env=env,
            policy=policy,
            seed=episode_seed,
            expected_variant=model_variant,
        )
        update_stats = trainer.update(trajectory)
        cumulative_reward = sum(step.reward for step in trajectory.steps)

        print(
            f"[Train Episode {episode_idx:03d}] "
            f"experiment={experiment.name} | "
            f"expected_variant={episode_info['expected_variant']} | "
            f"resolved_variant={episode_info['model_variant']} | "
            f"variant_match={episode_info['variant_match']} | "
            f"model_class={episode_info['model_class']} | "
            f"task_name={episode_info['task_name']} | "
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


def evaluate_transfer_family(
    policy: HannoNet1,
    device: str,
    experiment: MnistTransferExperiment,
    eval_episodes_per_variant: int = 3,
    base_seed: int = 5000,
) -> None:
    """
    Evaluate the trained controller on a held-out MNIST MLP family without
    updating the policy.
    """

    schedule = experiment.resource_schedule

    print(
        f"\n[Eval] experiment={experiment.name} | "
        f"schedule={schedule.name} | "
        f"held_out_variants={','.join(experiment.eval_variants)}"
    )

    for variant_idx, model_variant in enumerate(experiment.eval_variants):
        for episode_offset in range(eval_episodes_per_variant):
            episode_seed = base_seed + (variant_idx * 100) + episode_offset
            env = build_env(device=device, model_variant=model_variant, horizon=schedule.horizon)
            trajectory, episode_info = run_episode(
                env=env,
                policy=policy,
                seed=episode_seed,
                expected_variant=model_variant,
            )
            cumulative_reward = sum(step.reward for step in trajectory.steps)

            print(
                f"[Eval Episode {episode_offset + 1:03d}] "
                f"experiment={experiment.name} | "
                f"expected_variant={episode_info['expected_variant']} | "
                f"resolved_variant={episode_info['model_variant']} | "
                f"variant_match={episode_info['variant_match']} | "
                f"model_class={episode_info['model_class']} | "
                f"task_name={episode_info['task_name']} | "
                f"steps={episode_info['num_steps']} | "
                f"initial_loss={episode_info['initial_loss']:.6f} | "
                f"final_loss={episode_info['final_loss']:.6f} | "
                f"cum_reward={cumulative_reward:.6f} | "
                f"mean_effective_lr={episode_info['mean_effective_lr']:.6f} | "
                f"instability_count={episode_info['instability_count']}"
            )


def build_policy_and_trainer(device: str) -> tuple[HannoNet1, REINFORCETrainer]:
    """
    Build the controller and its REINFORCE trainer.
    """

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

    return policy, trainer


def main() -> None:
    """
    Assemble and run the first HannoNet1 MNIST transfer experiments.
    """

    seed_everything(42, deterministic=True)
    device = "cpu"

    for experiment_idx, experiment in enumerate(DEFAULT_EXPERIMENTS, start=1):
        policy, trainer = build_policy_and_trainer(device=device)
        train_transfer_family(
            policy=policy,
            trainer=trainer,
            device=device,
            experiment=experiment,
            base_seed=2000 + (experiment_idx * 1000),
        )
        evaluate_transfer_family(
            policy=policy,
            device=device,
            experiment=experiment,
            eval_episodes_per_variant=3,
            base_seed=8000 + (experiment_idx * 1000),
        )


if __name__ == "__main__":
    main()
