"""
3D Pallet Packing Environment for Reinforcement Learning.

State representation
    heightmap     : 2D grid storing the current top height at each cell
    item features : normalised dimensions and weight of the next item
    progress      : items placed ratio and pallet count

Action
    Single discrete index decoded into (grid_x, grid_y, rotation)
    6 rotations covering all axis-aligned orientations of (length, width, height)

Reward
    + per step  : item_volume / pallet_volume * 20
    - per step  : new_pallet_penalty when the agent triggers a fresh pallet
    + terminal  : utilisation * final_utilization_bonus
    - terminal  : pallet_count * pallet_count_penalty

Items rest on the highest occupied cell in their footprint (no flat-support
requirement). Max stack height and max pallet weight are enforced.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from numpy.lib.stride_tricks import sliding_window_view


@dataclass
class Item:
    sku: str
    length: float
    width: float
    height: float
    weight: float


def expand_items(data: dict) -> List[Item]:
    """Turn the JSON line items (with quantity) into a flat list of Item objects."""
    items: List[Item] = []
    for entry in data["items"]:
        for i in range(math.ceil(entry["quantity"])):
            items.append(
                Item(
                    sku=f"{entry['sku']}#{i:03d}",
                    length=float(entry["length_mm"]),
                    width=float(entry["width_mm"]),
                    height=float(entry["height_mm"]),
                    weight=float(entry["weight_kg"]),
                )
            )
    return items


class PalletPackingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        items: List[Item],
        pallet_length: float = 1200.0,
        pallet_width: float = 1100.0,
        pallet_height: float = 1150.0,
        max_pallet_weight: float = 1000.0,
        grid_resolution: float = 20.0,
        new_pallet_penalty: float = 5.0,
        final_utilization_bonus: float = 50.0,
        pallet_count_penalty: float = 2.0,
        skip_penalty: float = 10.0,
        sort_items: bool = True,
    ):
        super().__init__()
        self.original_items = items
        self.pallet_length = float(pallet_length)
        self.pallet_width = float(pallet_width)
        self.pallet_height = float(pallet_height)
        self.max_pallet_weight = float(max_pallet_weight)
        self.grid_resolution = float(grid_resolution)
        self.new_pallet_penalty = new_pallet_penalty
        self.final_utilization_bonus = final_utilization_bonus
        self.pallet_count_penalty = pallet_count_penalty
        self.skip_penalty = skip_penalty
        self.sort_items = sort_items

        self.grid_l = int(self.pallet_length // self.grid_resolution)
        self.grid_w = int(self.pallet_width // self.grid_resolution)
        self.pallet_volume = self.pallet_length * self.pallet_width * self.pallet_height
        self.num_rotations = 6

        self.action_space = spaces.Discrete(self.grid_l * self.grid_w * self.num_rotations)
        self.observation_space = spaces.Dict(
            {
                "heightmap": spaces.Box(
                    low=0.0, high=1.0,
                    shape=(self.grid_l, self.grid_w),
                    dtype=np.float32,
                ),
                "item": spaces.Box(
                    low=0.0, high=1.0,
                    shape=(4,),
                    dtype=np.float32,
                ),
                "progress": spaces.Box(
                    low=0.0, high=1.0,
                    shape=(2,),
                    dtype=np.float32,
                ),
            }
        )

        self.items: List[Item] = []
        self.current_idx = 0
        self.total_items = 0
        self.pallets: List[dict] = []
        self.reset()

    # ------------------------------------------------------------------ helpers

    def _new_pallet(self) -> dict:
        return {
            "heightmap": np.zeros((self.grid_l, self.grid_w), dtype=np.float32),
            "placements": [],
            "used_volume": 0.0,
            "used_weight": 0.0,
        }

    def _current_item(self) -> Optional[Item]:
        if self.current_idx >= len(self.items):
            return None
        return self.items[self.current_idx]

    def _item_dims(self, item: Item, rotation: int):
        """Return (footprint_l, footprint_w, stack_height) for each of 6 axis-aligned rotations."""
        l, w, h = item.length, item.width, item.height
        return [
            (l, w, h), (w, l, h),  # upright: swap horizontal
            (l, h, w), (h, l, w),  # tipped onto width face
            (w, h, l), (h, w, l),  # tipped onto length face
        ][rotation]

    def _item_cells(self, item: Item, rotation: int):
        fp_l, fp_w, _ = self._item_dims(item, rotation)
        l_cells = int(np.ceil(fp_l / self.grid_resolution))
        w_cells = int(np.ceil(fp_w / self.grid_resolution))
        return l_cells, w_cells

    def _check_placement(self, pallet: dict, item: Item, x: int, y: int, rotation: int):
        fp_l, fp_w, stack_h = self._item_dims(item, rotation)
        l_cells = int(np.ceil(fp_l / self.grid_resolution))
        w_cells = int(np.ceil(fp_w / self.grid_resolution))
        if x + l_cells > self.grid_l or y + w_cells > self.grid_w:
            return None
        if pallet["used_weight"] + item.weight > self.max_pallet_weight:
            return None
        region = pallet["heightmap"][x:x + l_cells, y:y + w_cells]
        base = float(region.max())
        if base + stack_h > self.pallet_height:
            return None
        return base, l_cells, w_cells, stack_h

    def _decode_action(self, action: int):
        rot = action % self.num_rotations
        pos = action // self.num_rotations
        y = pos % self.grid_w
        x = pos // self.grid_w
        return x, y, rot

    # ------------------------------------------------------------------ gym API

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.items = list(self.original_items)
        if self.sort_items:
            self.items.sort(key=lambda it: -(it.length * it.width * it.height))
        self.current_idx = 0
        self.total_items = len(self.items)
        self.pallets = [self._new_pallet()]
        return self._get_obs(), {}

    def _get_obs(self):
        item = self._current_item()
        if item is None:
            item_feat = np.zeros(4, dtype=np.float32)
        else:
            item_feat = np.array(
                [
                    min(item.length / self.pallet_length, 1.0),
                    min(item.width / self.pallet_width, 1.0),
                    min(item.height / self.pallet_height, 1.0),
                    min(item.weight / self.max_pallet_weight, 1.0),
                ],
                dtype=np.float32,
            )
        heightmap = (self.pallets[-1]["heightmap"] / self.pallet_height).astype(np.float32)
        progress = np.array(
            [
                self.current_idx / max(self.total_items, 1),
                min(len(self.pallets) / 20.0, 1.0),
            ],
            dtype=np.float32,
        )
        return {"heightmap": heightmap, "item": item_feat, "progress": progress}

    def action_masks(self) -> np.ndarray:
        """Vectorised valid action mask using a sliding window over the heightmap."""
        mask = np.zeros(self.action_space.n, dtype=bool)
        item = self._current_item()
        if item is None:
            mask[0] = True
            return mask

        pallet = self.pallets[-1]
        if pallet["used_weight"] + item.weight > self.max_pallet_weight:
            mask[:] = True
            return mask

        hm = pallet["heightmap"]
        for rot in range(self.num_rotations):
            _, _, stack_h = self._item_dims(item, rot)
            l_cells, w_cells = self._item_cells(item, rot)
            max_x = self.grid_l - l_cells + 1
            max_y = self.grid_w - w_cells + 1
            if max_x <= 0 or max_y <= 0:
                continue
            windows = sliding_window_view(hm, (l_cells, w_cells))
            max_h = windows.max(axis=(2, 3))
            fits = max_h + stack_h <= self.pallet_height
            xs, ys = np.where(fits)
            if xs.size:
                indices = (xs * self.grid_w + ys) * self.num_rotations + rot
                mask[indices] = True

        if not mask.any():
            mask[:] = True
        return mask

    def step(self, action):
        info = {}
        item = self._current_item()
        if item is None:
            return self._get_obs(), 0.0, True, False, info

        x, y, rot = self._decode_action(int(action))
        pallet = self.pallets[-1]
        result = self._check_placement(pallet, item, x, y, rot)
        reward = 0.0

        if result is None:
            self.pallets.append(self._new_pallet())
            pallet = self.pallets[-1]
            reward -= self.new_pallet_penalty
            for try_rot in range(self.num_rotations):
                cand = self._check_placement(pallet, item, 0, 0, try_rot)
                if cand is not None:
                    rot = try_rot
                    x, y = 0, 0
                    result = cand
                    break

        if result is None:
            self.current_idx += 1
            reward -= self.skip_penalty
            info["skipped"] = item.sku
            terminated = self.current_idx >= self.total_items
            if terminated:
                self._apply_final_reward(info)
                reward += info["final_reward"]
            return self._get_obs(), reward, terminated, False, info

        base, l_cells, w_cells, stack_h = result
        new_h = base + stack_h
        pallet["heightmap"][x:x + l_cells, y:y + w_cells] = new_h
        pallet["placements"].append(
            {
                "item": item,
                "x_mm": x * self.grid_resolution,
                "y_mm": y * self.grid_resolution,
                "z_mm": base,
                "rotation": rot,
                "l_cells": l_cells,
                "w_cells": w_cells,
                "stack_h": stack_h,
            }
        )
        vol = item.length * item.width * item.height
        pallet["used_volume"] += vol
        pallet["used_weight"] += item.weight
        reward += (vol / self.pallet_volume) * 20.0

        self.current_idx += 1
        terminated = self.current_idx >= self.total_items
        if terminated:
            self._apply_final_reward(info)
            reward += info["final_reward"]
        return self._get_obs(), reward, terminated, False, info

    def _apply_final_reward(self, info: dict):
        used = [p for p in self.pallets if p["placements"]]
        n_used = max(len(used), 1)
        total_vol = sum(p["used_volume"] for p in used)
        capacity = n_used * self.pallet_volume
        util = total_vol / capacity if capacity > 0 else 0.0
        info["final_reward"] = util * self.final_utilization_bonus - n_used * self.pallet_count_penalty
        info["num_pallets"] = n_used
        info["utilization"] = util
        info["placed"] = sum(len(p["placements"]) for p in used)
