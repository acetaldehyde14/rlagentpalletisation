"""
Evaluate the trained RL agent against a First Fit Decreasing baseline
and render the final pallets as 3D plots.

Usage
    python evaluate.py --data data.json --model checkpoints/best_model.zip
    python evaluate.py --baseline_only   (no trained model needed)
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from pallet_env import PalletPackingEnv, expand_items


def _draw_box(ax, x, y, z, dx, dy, dz, color):
    verts = list(product([x, x + dx], [y, y + dy], [z, z + dz]))
    faces = [
        [0, 1, 3, 2],
        [4, 5, 7, 6],
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [0, 2, 6, 4],
        [1, 3, 7, 5],
    ]
    poly = [[verts[i] for i in face] for face in faces]
    ax.add_collection3d(
        Poly3DCollection(poly, facecolors=color, edgecolors="black", linewidths=0.25, alpha=0.85)
    )


def visualize(env: PalletPackingEnv, title: str, output_path: str):
    used = [p for p in env.pallets if p["placements"]]
    if not used:
        print("No placements to draw.")
        return
    n = len(used)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig = plt.figure(figsize=(6 * cols, 5 * rows))
    fig.suptitle(title, fontsize=14)

    cmap = plt.cm.tab20
    sku_colors = {}

    for i, pallet in enumerate(used):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        for placement in pallet["placements"]:
            item = placement["item"]
            base_sku = item.sku.split("#")[0]
            if base_sku not in sku_colors:
                sku_colors[base_sku] = cmap(len(sku_colors) % 20)
            color = sku_colors[base_sku]
            l_mm = placement["l_cells"] * env.grid_resolution
            w_mm = placement["w_cells"] * env.grid_resolution
            _draw_box(
                ax,
                placement["x_mm"], placement["y_mm"], placement["z_mm"],
                l_mm, w_mm, placement["stack_h"],
                color,
            )
        ax.set_xlim(0, env.pallet_length)
        ax.set_ylim(0, env.pallet_width)
        ax.set_zlim(0, env.pallet_height)
        ax.set_xlabel("Length (mm)")
        ax.set_ylabel("Width (mm)")
        ax.set_zlabel("Height (mm)")
        util = pallet["used_volume"] / env.pallet_volume * 100
        weight = pallet["used_weight"]
        ax.set_title(f"Pallet {i + 1}: {len(pallet['placements'])} items, {util:.1f}% util, {weight:.1f} kg")

    handles = [Patch(facecolor=c, edgecolor="black", label=s) for s, c in sku_colors.items()]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 6), bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(output_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def report(env: PalletPackingEnv, label: str):
    used = [p for p in env.pallets if p["placements"]]
    total_vol = sum(p["used_volume"] for p in used)
    n_used = len(used)
    capacity = n_used * env.pallet_volume
    util = total_vol / capacity * 100 if capacity > 0 else 0.0
    placed = sum(len(p["placements"]) for p in used)
    print()
    print(f"=== {label} ===")
    print(f"Pallets used        {n_used}")
    print(f"Items placed        {placed} / {env.total_items}")
    print(f"Avg utilisation     {util:.2f}%")
    for i, p in enumerate(used):
        pu = p["used_volume"] / env.pallet_volume * 100
        print(f"  Pallet {i + 1}: {len(p['placements'])} items, util {pu:5.1f}%, weight {p['used_weight']:.2f} kg")


def first_fit_decreasing(items, env_kwargs):
    env = PalletPackingEnv(items, **env_kwargs, sort_items=True)
    env.reset()
    while env.current_idx < env.total_items:
        masks = env.action_masks()
        valid_idx = np.where(masks)[0]
        if valid_idx.size == 0:
            break
        env.step(int(valid_idx[0]))
    return env


def extreme_point(items, env_kwargs):
    """Extreme Point heuristic: place each item at the valid position with the
    lowest base height, breaking ties by x then y (front-left-bottom preference).
    Items sorted by volume descending (same as FFD).
    """
    env = PalletPackingEnv(items, **env_kwargs, sort_items=True)
    env.reset()
    while env.current_idx < env.total_items:
        masks = env.action_masks()
        valid_idx = np.where(masks)[0]
        if valid_idx.size == 0:
            break

        item = env._current_item()
        pallet = env.pallets[-1]

        best_action = None
        best_score = None

        for action in valid_idx:
            x, y, rot = env._decode_action(int(action))
            result = env._check_placement(pallet, item, x, y, rot)
            if result is None:
                continue
            base, _, _, _ = result
            score = (base, x, y)
            if best_score is None or score < best_score:
                best_score = score
                best_action = int(action)

        # No valid placement on current pallet — fall back to first masked action
        # (env.step will open a new pallet automatically)
        if best_action is None:
            best_action = int(valid_idx[0])

        env.step(best_action)
    return env


def run_agent(model, items, env_kwargs):
    from sb3_contrib.common.wrappers import ActionMasker

    base_env = PalletPackingEnv(items, **env_kwargs, sort_items=True)
    env = ActionMasker(base_env, lambda e: e.action_masks())
    obs, _ = env.reset()
    done = False
    while not done:
        masks = env.action_masks()
        action, _ = model.predict(obs, action_masks=masks, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(int(action))
        done = terminated or truncated
    return base_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data.json")
    parser.add_argument("--model", default="./checkpoints/best_model.zip")
    parser.add_argument("--baseline_only", action="store_true")
    parser.add_argument("--pallet_length", type=float, default=1200.0)
    parser.add_argument("--pallet_width", type=float, default=1100.0)
    parser.add_argument("--pallet_height", type=float, default=1150.0)
    parser.add_argument("--max_weight", type=float, default=1000.0)
    args = parser.parse_args()

    with open(args.data) as f:
        data = json.load(f)
    items = expand_items(data)
    env_kwargs = dict(
        pallet_length=args.pallet_length,
        pallet_width=args.pallet_width,
        pallet_height=args.pallet_height,
        max_pallet_weight=args.max_weight,
    )

    print(f"Loaded {len(items)} items from {args.data}")
    total_volume = sum(it.length * it.width * it.height for it in items) / 1e9
    pallet_volume = args.pallet_length * args.pallet_width * args.pallet_height / 1e9
    print(f"Total item volume   {total_volume:.3f} m^3")
    print(f"Pallet capacity     {pallet_volume:.3f} m^3")
    print(f"Theoretical minimum {int(np.ceil(total_volume / pallet_volume))} pallets")

    baseline_env = first_fit_decreasing(items, env_kwargs)
    report(baseline_env, "First Fit Decreasing baseline")
    visualize(baseline_env, "First Fit Decreasing baseline", "baseline_result.png")

    ep_env = extreme_point(items, env_kwargs)
    report(ep_env, "Extreme Point baseline")
    visualize(ep_env, "Extreme Point baseline", "ep_result.png")

    if args.baseline_only:
        return
    if not Path(args.model).exists():
        print(f"\nModel {args.model} not found. Run train.py first or pass --baseline_only.")
        return

    from sb3_contrib import MaskablePPO
    model = MaskablePPO.load(args.model)
    agent_env = run_agent(model, items, env_kwargs)
    report(agent_env, "MaskablePPO agent")
    visualize(agent_env, "MaskablePPO agent", "agent_result.png")


if __name__ == "__main__":
    main()
