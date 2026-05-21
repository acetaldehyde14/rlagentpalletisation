# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Reinforcement learning agent for 3D bin packing (pallet optimization). Uses MaskablePPO to learn how to pack mixed-SKU items onto pallets, minimizing pallet count while maximizing volume utilization.

## Setup & Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Train the agent
python train.py --timesteps 500000 --n_envs 4

# Multi-core training
python train.py --timesteps 1000000 --n_envs 8 --subproc

# Run baseline only (no model needed)
python evaluate.py --baseline_only

# Evaluate trained model with 3D visualization
python evaluate.py --model checkpoints/best_model.zip

# Training without TensorBoard
python train.py --timesteps 500000 --no_tensorboard
```

## Architecture

**`pallet_env.py` — core RL environment (`PalletPackingEnv`)**
- Observation: normalized heightmap (60×40 grid at 20 mm resolution) + current item features + progress metrics
- Action space: 4,800 discrete actions encoding (x_pos × y_pos × rotation) = 60×40×2
- Action masking: vectorized sliding-window validation enforcing bounds, flat support, height/weight limits
- Reward: per-item placement reward + new-pallet penalty + terminal utilization bonus

**`train.py` — training harness**
- `MaskablePPO` with `MultiInputPolicy` from `sb3-contrib` (handles Dict observation space)
- Parallel envs via `DummyVecEnv` or `SubprocVecEnv`
- Callbacks: `CheckpointCallback` (periodic saves) + `MaskableEvalCallback` (best model)
- Key hyperparameters: lr=3e-4, n_steps=512, batch_size=128, gamma=0.99

**`evaluate.py` — evaluation and visualization**
- First Fit Decreasing (FFD) baseline for comparison
- Deterministic inference from saved model
- 3D matplotlib visualization with per-pallet subplots and SKU color coding

**`data.json` — training data**
- 117 items across 6 SKUs; theoretical minimum 2 pallets (~2.153 m³ total volume)

## Key Design Constraints

- Items support only 90° horizontal rotation (2 orientations)
- Flat support required: items must rest on a single continuous height level
- Default pallet: EUR 1200×800×1500 mm, max 1000 kg
- Grid resolution is 20 mm (adjustable to 10 mm via `grid_res` for thin items)
- Items are sorted by volume descending before packing begins
