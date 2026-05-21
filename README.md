# Pallet Packing RL Agent

A MaskablePPO agent that learns to pack mixed SKU items into the fewest pallets at the highest volume utilisation. The environment uses a heightmap representation, action masking for legal placements, and a reward that rewards volume gain while penalising new pallets.

## Files

```
pallet_rl/
├── data.json          training data (your items)
├── pallet_env.py      Gymnasium environment + Item dataclass
├── train.py           MaskablePPO trainer
├── evaluate.py        runs agent vs First Fit Decreasing baseline + 3D plots
├── smoke_test.py      quick env sanity check (no RL stack needed)
├── requirements.txt
└── README.md
```

## Install

```bash
pip install -r requirements.txt
```

Optional for training logs:

```bash
pip install tensorboard
```

## Quick start

Run the baseline first to verify everything works without training:

```bash
python evaluate.py --baseline_only
```

Train the agent:

```bash
python train.py --timesteps 500000 --n_envs 4
```

For faster training on a multicore CPU:

```bash
python train.py --timesteps 1000000 --n_envs 8 --subproc
```

Skip tensorboard if it is not installed:

```bash
python train.py --timesteps 500000 --no_tensorboard
```

Evaluate the trained agent and visualise the result:

```bash
python evaluate.py --model checkpoints/best_model.zip
```

This writes `baseline_result.png` and `agent_result.png` in the current directory.

## How the environment works

Pallet dimensions default to 1200 mm × 800 mm × 1500 mm (EUR pallet) with a 1000 kg weight cap. The floor discretises into a 60 × 40 grid at 20 mm resolution. Items expand to one entry per unit of quantity (your data yields 117 items) and sort by volume descending so the agent always places the next largest unplaced item.

The state contains three pieces. The first is a normalised 60 × 40 heightmap of the current pallet's top surface. The second is the next item as a 4 vector of length, width, height, weight normalised against pallet limits. The third is a 2 vector of progress information (fraction of items placed, normalised pallet count).

The action is a single discrete index over `grid_l × grid_w × 2` (4800 actions), decoded into `(x, y, rotation)`. Rotation 0 keeps length along x, rotation 1 rotates the item 90 degrees around the vertical axis. A `sliding_window_view` computes the legal action mask in a vectorised pass: a placement is legal when the item fits within bounds, the footprint sits on a single flat height level (full support), the new stack height stays under the pallet limit, and the pallet weight cap is not exceeded.

When the agent's chosen position fails the check the env opens a new pallet and tries the corner `(0, 0)` with both rotations. If even an empty pallet cannot hold the item (oversized for any pallet) the item is skipped with a heavy penalty.

## Reward design

Per step:
1. Plus `(item_volume / pallet_volume) × 20` for every successful placement.
2. Minus `new_pallet_penalty` (default 5.0) when a fresh pallet opens.
3. Minus `skip_penalty` (default 10.0) when an item cannot be placed anywhere.

Episode terminal bonus:
4. Plus `utilisation × final_utilization_bonus` (default 50.0) where utilisation is `total_used_volume / (n_pallets × pallet_volume)`.
5. Minus `pallet_count × pallet_count_penalty` (default 2.0).

Tune these constants in `PalletPackingEnv.__init__` to bias toward fewer pallets vs higher utilisation. The two objectives correlate strongly (fewer pallets usually means higher fill) but you can shift the balance: raise `pallet_count_penalty` for fewer pallets at any cost, raise `final_utilization_bonus` to favour dense packing.

## Customising

Pallet geometry in `evaluate.py`:

```bash
python evaluate.py --pallet_length 1100 --pallet_width 1100 --pallet_height 1800
```

Pass the same kwargs into `PalletPackingEnv(...)` in `train.py` if you change pallet size for training. Singapore commonly uses 1100 × 1100 mm ISO pallets, so swap as needed.

Grid resolution lives in `pallet_env.py` (default 20 mm). Lower it to 10 mm for thin items at the cost of a 4× larger action space. Items get rounded up to the nearest cell so a 21.6 mm wide item occupies 2 cells at 20 mm resolution (40 mm slot).

## Expected results

For your 117 item dataset (2.153 m³ total) the theoretical minimum is 2 pallets. The First Fit Decreasing baseline reaches 3 pallets at 49.8% average utilisation. A trained MaskablePPO agent at 500K steps should match or beat this, typically 2 or 3 pallets at 55 to 75% utilisation depending on hyperparameters and reward weighting.

## Notes and limitations

The current support rule requires a fully flat footprint underneath the item. Real pallets often allow partial overhang, which would let the agent pack more densely. To relax this, edit `_check_placement` in `pallet_env.py` to require, for example, at least 80% of the footprint cells at the base height.

The env enforces axis aligned placement and 90 degree horizontal rotation only. Adding vertical rotation (6 orientations total) means swapping height with length or width, which is realistic for some items but unsafe for others (anything fragile, liquid, or "this way up"). Add a per item `allow_rotation` flag if you need that.

The agent is trained on a single instance (your 117 items in fixed sorted order). To generalise across orders, randomise quantities and SKU mix in a `make_env` factory before training.
