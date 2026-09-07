"""Open-loop evaluation for Psi0 on a single episode.

Mirrors examples/simple/openloop_eval.ipynb as a runnable script.
"""

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoProcessor

# ---------------------------------------------------------------------------
# Locate project root and chdir into it so relative paths (.runs/, etc.) work
# ---------------------------------------------------------------------------
project_root = Path(__file__).resolve().parent
while project_root != project_root.parent and not (project_root / "pyproject.toml").exists():
    project_root = project_root.parent
os.chdir(project_root)

import json
from psi.utils import parse_args_to_tyro_config, seed_everything, apply_legacy_model_config_defaults
from psi.config.config import LaunchConfig

CKPT_STEP = 40000

# Action component labels and split boundaries (36-dim action)
LABELS = [
    "hand_joints",   # [0:14]
    "arm_joints",    # [14:28]
    "torsor_roll",   # [28:31]  (torso rpy, 3-dim)
    "torsor_pitch",  # [31:32]
    "torsor_yaw",    # [32:33]
    "height",        # [33:34]
    "vx",            # [34:35]
    "vy",            # [35:36]
    "torso_vyaw",    # unused in print split but used in plot split
    "target_yaw",
]
PRINT_SPLITS = [14, 28, 31, 32, 33, 34, 35]   # 8 groups
PLOT_SPLITS  = [14, 28, 29, 30, 31, 32, 33, 34, 35]  # 10 groups


def parse_args():
    p = argparse.ArgumentParser(description="Open-loop eval for Psi0")
    p.add_argument("--run-dir", type=Path, required=True,
                   help="Path to the run directory (contains argv.txt and run_config.json)")
    p.add_argument("--gpu", type=int, default=0,
                   help="GPU index to use (e.g. 0 → cuda:0)")
    p.add_argument("--eps-idx", type=int, default=None,
                   help="Episode index to evaluate (random if not set)")
    p.add_argument("--stride", type=int, default=4,
                   help="Frame stride within the episode")
    p.add_argument("--num-inference-steps", type=int, default=10)
    p.add_argument("--output-dir", type=str, default=".",
                   help="Directory to save the output plot")
    return p.parse_args()


def load_config(run_dir: Path) -> LaunchConfig:
    config_: LaunchConfig = parse_args_to_tyro_config(run_dir / "argv.txt")  # type: ignore
    conf = apply_legacy_model_config_defaults(
        json.loads((run_dir / "run_config.json").read_text()))
    launch_config = config_.model_validate(conf)
    return launch_config


def main():
    args = parse_args()
    run_dir = args.run_dir

    launch_config = load_config(run_dir)
    seed_everything(launch_config.seed or 42)

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------
    from psi.models.psi0 import Psi0Model

    device = f"cuda:{args.gpu}"
    psi0 = Psi0Model.from_pretrained(run_dir, CKPT_STEP, launch_config, device=device)
    psi0.to(device)
    psi0.eval()
    print("Model loaded successfully.")

    # ------------------------------------------------------------------
    # Load dataset
    # ------------------------------------------------------------------
    from psi.config.data_lerobot import LerobotDataConfig

    data_cfg: LerobotDataConfig = launch_config.data  # type: ignore
    maxmin = data_cfg.transform.field

    transform_kwargs = dict(vlm_processor=psi0.vlm_processor)
    # vlm_processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
    # transform_kwargs = dict(vlm_processor=vlm_processor)
    dataset = data_cfg(split="train", transform_kwargs=transform_kwargs)

    # ------------------------------------------------------------------
    # Pick episode
    # ------------------------------------------------------------------
    total_episodes = dataset.raw_dataset.meta.total_episodes
    eps_idx = args.eps_idx if args.eps_idx is not None else np.random.randint(0, total_episodes)
    print(f"Episode {eps_idx} / {total_episodes}")

    start_frame = dataset.raw_dataset.base_dataset.episode_data_index["from"][eps_idx].item()
    end_frame   = dataset.raw_dataset.base_dataset.episode_data_index["to"][eps_idx].item()
    print(f"Frames {start_frame}..{end_frame}  ({end_frame - start_frame} frames)")

    # ------------------------------------------------------------------
    # Eval loop
    # ------------------------------------------------------------------
    per_frame_errors = []

    for i in tqdm(
        range(start_frame, end_frame, args.stride),
        desc="Evaluating frames",
        unit="frame",
    ):
        frame = dataset[i]
        # print(frame.keys()); break

        batch_images       = [frame["raw_images"]]
        batch_instructions = [frame["instruction"]]
        batch_states       = torch.from_numpy(frame["states"]).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_actions = psi0.predict_action(
                observations=batch_images,
                states=batch_states,
                instructions=batch_instructions,
                num_inference_steps=args.num_inference_steps,
                traj2ds=None,
            )

        gt_action = torch.from_numpy(frame["raw_actions"]).unsqueeze(0).to(device)
        denorm_pred = maxmin.denormalize(pred_actions)

        error_l1 = (denorm_pred - gt_action).abs().detach().cpu().numpy()
        error_l1 = error_l1.reshape(-1, gt_action.shape[-1])   # (Tp, Da)
        per_frame_errors.append(error_l1.mean(0))              # (Da,)

    per_frame_errors = np.stack(per_frame_errors, axis=0)   # (T, Da)
    mean_error       = per_frame_errors.mean(axis=0)        # (Da,)

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print("\n---------------------------\n")
    for i, seg in enumerate(np.split(mean_error, PRINT_SPLITS)):
        print(f"denormed_err_l1_{LABELS[i]} {seg.shape}:  {np.linalg.norm(seg):.6f}")

    # ------------------------------------------------------------------
    # Plot per-step error curves
    # ------------------------------------------------------------------
    error_groups   = np.split(per_frame_errors, PLOT_SPLITS, axis=-1)
    per_label_norm = [np.linalg.norm(g, axis=-1) for g in error_groups]
    curve_map      = dict(zip(LABELS, per_label_norm))

    plot_groups = [
        ("hand_joints + arm_joints (rad)", ["hand_joints", "arm_joints"]),
        ("torso rpy (rad)",                ["torsor_roll", "torsor_pitch", "torsor_yaw"]),
        ("height (m)",                     ["height"]),
        ("vx + vy (m/s)",                  ["vx", "vy"]),
        ("target_yaw (rad)",               ["target_yaw"]),
    ]

    fig, axes = plt.subplots(5, 1, figsize=(12, 16), sharex=True)
    for ax, (title, keys) in zip(axes, plot_groups):
        for key in keys:
            if key in curve_map:
                ax.plot(curve_map[key], label=key)
        ax.set_title(title)
        ax.set_ylabel("Error norm")
        ax.grid(True, alpha=0.3)
        if len(keys) > 1:
            ax.legend()
    axes[-1].set_xlabel(f"Sample step in episode (stride={args.stride})")
    plt.suptitle(f"Episode {eps_idx}  |  ckpt step {CKPT_STEP}", y=1.01)
    plt.tight_layout()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"openloop_eval_eps{eps_idx}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out_path}")


if __name__ == "__main__":
    main()
