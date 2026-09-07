"""Mock HTTP client for serve_psi0_sonic_http.py (the non-RTC POST /act sonic server).

Replays a validation episode: for each frame it POSTs one observation to /act, receives a
denormalized action chunk (Ta, Da), and compares it to the first Ta ground-truth steps,
reporting per-modality denormalized L1 and round-trip latency.

This mirrors mock_psi0_client_rtc.py for the sonic data path (image_keys, raw-state
round-trip via the ActionStateTransform field) but uses synchronous HTTP like
mock_psix_client_http.py instead of the RTC WebSocket stream.

Usage:
    serve_psi0_sonic_http --policy psi0 --port 8014 --ckpt-step 40000 \
        --run-dir .runs/finetune/g1neck30fps622.sonic.flow1000.cosine.lr1.0e-04.b128.gpus8.2606270114
    python src/psi/deploy/mock_psi0_client_sonic_http.py \
        --run-dir .runs/finetune/g1neck30fps622.sonic.flow1000.cosine.lr1.0e-04.b128.gpus8.2606270114 \
        --host localhost --port 8014 --eps-idx 18 --target-hz 30
"""

import argparse
import time
from pathlib import Path

from dotenv import load_dotenv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
from tqdm.auto import tqdm
from transformers import AutoProcessor

import json
from psi.utils import parse_args_to_tyro_config, seed_everything, apply_legacy_model_config_defaults
from psi.utils.overwatch import initialize_overwatch
from psi.config.config import LaunchConfig
from psi.deploy.helpers import RequestMessage, ResponseMessage

overwatch = initialize_overwatch(__name__)

_env_path = Path(".env")
if _env_path.exists():
    load_dotenv(_env_path)
else:
    overwatch.warning(".env not found in the current directory — environment variables may be missing.")

# Modality split of the action vector, matching mock_psi0_client_rtc.py /
# examples/openloop/psi0_inference_g1neck.py.
ACTION_SPLITS = [64, 78]
ACTION_LABELS = ["latent_action", "hand_joints", "neck_joints"]


def parse_args():
    p = argparse.ArgumentParser(description="Mock HTTP client for serve_psi0_sonic_http.py")
    p.add_argument("--run-dir",   type=Path, required=True,
                   help="Run directory (contains argv.txt and run_config.json) — MUST be the "
                        "same run the server was started with, so stats/image_keys match.")
    p.add_argument("--host",      type=str,   default="localhost")
    p.add_argument("--port",      type=int,   default=8014)
    p.add_argument("--eps-idx",   type=int,   default=0,
                   help="Validation episode index (18 matches the open-loop reference).")
    p.add_argument("--split",     type=str,   default="val",
                   help="Dataset split to replay (open-loop reference uses 'val').")
    p.add_argument("--target-hz", type=float, default=30.0,
                   help="Observation send frequency (Hz).")
    p.add_argument("--timeout",   type=float, default=60.0,
                   help="Per-request timeout (s); first request waits through model warmup.")
    p.add_argument("--output-dir", type=str,  default=".")
    return p.parse_args()


def load_config(run_dir: Path) -> LaunchConfig:
    config_: LaunchConfig = parse_args_to_tyro_config(run_dir / "argv.txt")  # type: ignore
    conf = apply_legacy_model_config_defaults(
        json.loads((run_dir / "run_config.json").read_text()))
    return config_.model_validate(conf)


def denormalize_state(norm_state: np.ndarray, field) -> np.ndarray:
    """Invert normalize_state_func so the server's re-normalization recovers the same state
    the open-loop path feeds the model (ill dims map to 0 both ways, so the round-trip is exact)."""
    state_min = np.asarray(field.state_min, dtype=np.float32)
    state_max = np.asarray(field.state_max, dtype=np.float32)
    norm_state = np.asarray(norm_state, dtype=np.float32).reshape(-1)
    assert norm_state.shape[0] == state_min.shape[0], (
        f"state dim {norm_state.shape[0]} != stats dim {state_min.shape[0]}; "
        f"run-dir mismatch with the server?")
    return ((norm_state + 1.0) / 2.0 * (state_max - state_min) + state_min).astype(np.float32)


def build_request(frame: dict, image_keys: list[str], field, dataset_name: str) -> dict:
    """Serialize one dataset frame into the RequestMessage dict the psi0 sonic server expects."""
    observations = frame["observations"]  # PIL images, one per repack image key, in order
    assert len(observations) == len(image_keys), (
        f"{len(observations)} obs images but {len(image_keys)} image_keys {image_keys}")
    image_dict = {k: np.asarray(img, dtype=np.uint8) for k, img in zip(image_keys, observations)}

    # frame state is normalized; the server expects RAW and re-normalizes it
    state_vec = np.asarray(frame["states"], dtype=np.float32)
    if state_vec.ndim == 2:
        state_vec = state_vec[-1]
    raw_state = denormalize_state(state_vec, field)

    msg = RequestMessage(
        image=image_dict,
        instruction=frame["instruction"],
        history={},
        state={"states": raw_state},
        condition={},
        gt_action=[],
        dataset_name=dataset_name,
        timestamp=str(time.time()),
    )
    return msg.serialize()


def run_episode(base_url, frames, image_keys, field, dataset_name, target_hz, recv_timeout):
    """POST each frame's obs to /act and collect the returned action chunk.

    Returns (gt_actions, pred_actions, send_times, recv_times), index-aligned. Each pred is the
    server's (Ta, Da) chunk; each gt is the matching first Ta GT steps for that frame.
    """
    gt_chunks   = [np.asarray(f["raw_actions"], dtype=np.float32) for f in frames]  # (Tp, Da) each
    gt_actions:   list[np.ndarray] = []
    pred_actions: list[np.ndarray] = []
    send_times:   list[float]      = []
    recv_times:   list[float]      = []

    interval = 1.0 / target_hz
    overwatch.info(f"POST {base_url}/act — first frame absorbs the server-side model build")
    for k, frame in enumerate(tqdm(frames, desc="Replaying frames", unit="frame")):
        payload = build_request(frame, image_keys, field, dataset_name)
        send_t = time.perf_counter()
        try:
            resp = requests.post(f"{base_url}/act", json=payload, timeout=recv_timeout)
        except requests.Timeout:
            overwatch.warning(f"Timeout after {recv_timeout}s at frame {k} — stopping "
                              f"({len(pred_actions)}/{len(frames)} received)")
            break
        recv_times.append(time.perf_counter())
        send_times.append(send_t)

        pred = np.asarray(ResponseMessage.deserialize(resp.json()).action, dtype=np.float32)  # (Ta, Da)
        pred = pred.reshape(-1, pred.shape[-1])
        Ta = pred.shape[0]
        pred_actions.append(pred)
        gt_actions.append(gt_chunks[k][:Ta])

        sleep = interval - (time.perf_counter() - send_t)
        if sleep > 0:
            time.sleep(sleep)

    return gt_actions, pred_actions, send_times, recv_times


def main():
    args = parse_args()

    launch_config = load_config(args.run_dir)
    seed_everything(launch_config.seed or 42)

    # Build the dataset with just the VLM processor (CPU) — no model, so no GPU contention.
    from psi.config.data_lerobot import LerobotDataConfig
    from psi.models.psi0 import QWEN3VL_VARIANT

    data_cfg: LerobotDataConfig = launch_config.data  # type: ignore
    vlm_processor = AutoProcessor.from_pretrained(QWEN3VL_VARIANT)
    dataset = data_cfg(split=args.split, transform_kwargs=dict(vlm_processor=vlm_processor, no_aug=True))
    overwatch.info(f"{args.split} dataset size: {len(dataset)}")

    repack = launch_config.data.transform.repack          # type: ignore
    image_keys: list[str] = repack.image_keys
    field = launch_config.data.transform.field            # type: ignore
    assert field.normalize_state, "server asserts normalize_state; field has it disabled"

    # Slice the requested episode like the open-loop reference does.
    episode_index = dataset.raw_dataset.base_dataset.episode_data_index
    start = int(episode_index["from"][args.eps_idx].item())
    end   = int(episode_index["to"][args.eps_idx].item())
    overwatch.info(f"Episode {args.eps_idx}: frames [{start}, {end}) -> {end - start} frames")

    frames = [dataset[i] for i in tqdm(range(start, end), desc="Loading episode frames", unit="frame")]
    if not frames:
        raise RuntimeError("No frames found for the given episode index.")
    dataset_name = frames[0].get("dataset_name")
    overwatch.info(f"image_keys: {image_keys}  |  action_dim: {launch_config.model.action_dim}")  # type: ignore

    base_url = f"http://{args.host}:{args.port}"
    gt_actions, pred_actions, send_times, recv_times = run_episode(
        base_url, frames, image_keys, field, dataset_name, args.target_hz, args.timeout
    )

    n_pairs = min(len(gt_actions), len(pred_actions))
    if n_pairs == 0:
        raise RuntimeError("No action pairs received — is the server running?")
    overwatch.info(f"Paired {n_pairs}/{len(frames)} frames with received actions")

    # Per-frame denormalized L1 over the returned chunk, split by modality.
    errors = np.stack([
        np.abs(gt_actions[i] - pred_actions[i]).mean(axis=0)  # (Da,) averaged over the chunk
        for i in range(n_pairs)
    ])  # (T, Da)
    mean_err = errors.mean(axis=0)                            # (Da,)
    per_modality = np.split(mean_err, ACTION_SPLITS, axis=-1)

    rtt_ms = np.array([(recv_times[i] - send_times[i]) * 1000 for i in range(n_pairs)])
    mean_rtt, std_rtt = float(rtt_ms.mean()), float(rtt_ms.std())

    overwatch.info("--- per-modality denormalized L1 (vs first Ta GT steps) ---")
    for label, chunk in zip(ACTION_LABELS, per_modality):
        overwatch.info(f"  denormed_err_l1_{label}: {chunk.mean():.6f}")
    overwatch.info(f"round-trip latency: {mean_rtt:.1f} ± {std_rtt:.1f} ms  (n={n_pairs})")

    # Plot per-modality L1 curves over the episode.
    curve_map = {
        label: chunk.mean(axis=1)
        for label, chunk in zip(ACTION_LABELS, np.split(errors, ACTION_SPLITS, axis=-1))
    }
    n_groups = len(curve_map)
    fig, axes = plt.subplots(n_groups, 1, figsize=(12, 3 * n_groups), sharex=True)
    if n_groups == 1:
        axes = [axes]
    for ax, (label, curve) in zip(axes, curve_map.items()):
        mean_val = float(curve.mean())
        ax.plot(curve)
        # Horizontal line marking the per-modality average denormalized L1.
        ax.axhline(mean_val, color="tab:red", linestyle="--", linewidth=1,
                   label=f"mean {mean_val:.4f}")
        ax.set_title(f"{label}  (mean {mean_val:.4f})")
        ax.set_ylabel("Denormalized L1")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Frame step in episode")
    plt.suptitle(
        f"psi0 sonic HTTP mock  |  {dataset_name}  |  Episode {args.eps_idx}  |  {args.host}:{args.port}"
        f"\nRound-trip {mean_rtt:.1f} ± {std_rtt:.1f} ms  (n={n_pairs} pairs)",
        y=1.01,
    )
    plt.tight_layout()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"mock_psi0_sonic_http_eps{args.eps_idx}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    overwatch.info(f"Plot saved to {out_path}")


if __name__ == "__main__":
    main()
