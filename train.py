"""
Train a MaskablePPO agent on the pallet packing environment.

Usage
    python train.py --data data.json --timesteps 500000

The script wraps the env with sb3_contrib ActionMasker so the policy never
samples illegal placements. A SubprocVecEnv is used when n_envs > 1.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from pallet_env import PalletPackingEnv, expand_items


def _mask_fn(env):
    return env.action_masks()


def _make_env(data_path: str, seed: int = 0):
    def _init():
        with open(data_path) as f:
            data = json.load(f)
        items = expand_items(data)
        env = PalletPackingEnv(items)
        env = ActionMasker(env, _mask_fn)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data.json")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--n_envs", type=int, default=4)
    parser.add_argument("--save_dir", default="./checkpoints")
    parser.add_argument("--log_dir", default="./logs")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n_steps", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--subproc", action="store_true",
                        help="Use SubprocVecEnv (faster on big runs, requires __main__ guard)")
    parser.add_argument("--no_tensorboard", action="store_true",
                        help="Disable tensorboard logging (use if tensorboard is not installed)")
    args = parser.parse_args()

    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    env_fns = [_make_env(args.data, seed=i) for i in range(args.n_envs)]
    vec_cls = SubprocVecEnv if args.subproc and args.n_envs > 1 else DummyVecEnv
    env = vec_cls(env_fns)
    eval_env = DummyVecEnv([_make_env(args.data, seed=999)])

    model = MaskablePPO(
        policy="MultiInputPolicy",
        env=env,
        verbose=1,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=args.ent_coef,
        tensorboard_log=None if args.no_tensorboard else args.log_dir,
    )

    callbacks = [
        CheckpointCallback(
            save_freq=max(50_000 // args.n_envs, 1),
            save_path=args.save_dir,
            name_prefix="ppo",
        ),
        MaskableEvalCallback(
            eval_env,
            best_model_save_path=args.save_dir,
            log_path=args.log_dir,
            eval_freq=max(10_000 // args.n_envs, 1),
            deterministic=True,
            n_eval_episodes=3,
        ),
    ]

    model.learn(total_timesteps=args.timesteps, callback=callbacks, tb_log_name="maskable_ppo")
    final_path = Path(args.save_dir) / "final_model"
    model.save(final_path)
    print(f"Training finished. Final model saved at {final_path}.zip")
    print(f"Best model saved at {Path(args.save_dir) / 'best_model.zip'}")


if __name__ == "__main__":
    main()
