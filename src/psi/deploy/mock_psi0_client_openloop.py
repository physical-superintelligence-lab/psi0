"""Open-loop HTTP client for serve_psi0_simple.py / serve_psi0_simple_multi.py.

Replays a validation episode frame by frame over `POST /act` and reports per-modality
denormalized L1 between the returned action chunk and the ground-truth chunk of the SAME
frame. Unlike `mock_psi0_client_rtc.py` (real-robot, WebSocket, one executed step per tick),
this is a synchronous request/response loop: send frame k, wait for its chunk, score it,
move on. Nothing is executed, so there is no version pairing and no timing pressure — the
number it reports is the open-loop full-chunk error.

The dataset loading, request building, response parsing and L1 accounting mirror
`mock_psi0_client_rtc.py`; only the transport and the scoring horizon differ.

Usage:
    serve_psi0_simple --host 0.0.0.0 --port 22085 \
        --run-dir=.runs/finetune/mix33-qknorm-100k.simple.flow1000.cosine.lr1.0e-04.b256.gpus8.2608081545 \
        --ckpt-step=60000 --action-exec-horizon=24 --rtc

    python src/psi/deploy/mock_psi0_client_openloop.py \
        --run-dir .runs/finetune/mix33-qknorm-100k.simple.flow1000.cosine.lr1.0e-04.b256.gpus8.2608081545 \
        --host localhost --port 22085 --eps-idx 0 --max-frames 60
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
from tqdm.auto import tqdm
from transformers import AutoProcessor

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

# Modality split of the 36-D simple-embodiment action, matching
# ActionStateTransform.denormalize_L1_action_err:
#   hand_joints(14) + arm_joints(14) + rpy(3) + height(1) + vx + vy + vyaw + dyaw
ACTION_SPLITS = [14, 28, 31, 32, 33, 34, 35]
ACTION_LABELS = ["hand_joints", "arm_joints", "torso_rpy", "height",
                 "vx", "vy", "vyaw", "target_yaw"]


def parse_args():
    p = argparse.ArgumentParser(description="Open-loop HTTP client for the psi0-simple servers")
    p.add_argument("--run-dir",    type=Path, required=True,
                   help="Run directory (contains argv.txt and run_config.json) — MUST be the "
                        "same run the server was started with, so stats/transforms match.")
    p.add_argument("--host",       type=str,   default="localhost")
    p.add_argument("--port",       type=int,   default=22085)
    p.add_argument("--eps-idx",    type=int,   nargs="+", default=[0],
                   help="Validation episode index/indices. Several indices are replayed by that "
                        "many CONCURRENT clients (one per episode), which is how the batching "
                        "server gets exercised; add --sequential to replay them one at a time.")
    p.add_argument("--sequential", action="store_true",
                   help="Replay multiple episodes back to back instead of concurrently — gives "
                        "the batch-size-1 baseline to A/B the concurrent run against.")
    p.add_argument("--split",      type=str,   default="val")
    p.add_argument("--max-frames", type=int,   default=0, help="0 = whole episode.")
    p.add_argument("--stride",     type=int,   default=1, help="Replay every Nth frame.")
    p.add_argument("--timeout",    type=float, default=120.0,
                   help="Per-request timeout (s); the first request absorbs GPU warmup.")
    p.add_argument("--image-keys", type=str,   nargs="+", default=["rgb_head_stereo_left"],
                   help="Keys for the image dict; the server consumes values in order.")
    p.add_argument("--client-id",  type=str,   default="openloop",
                   help="Sent as history['client_id']; the multi server keys RTC state on it "
                        "(ignored by the single-client server).")
    p.add_argument("--reset-every-frame", action="store_true",
                   help="Send history['reset'] on every frame so each prediction is independent "
                        "of the previous chunk. Default resets only on the first frame, i.e. the "
                        "server chains RTC across the episode as it would on the robot.")
    p.add_argument("--repo-id",    type=str,   default=None,
                   help="Override the run's val repo id — useful when the training dataset is "
                        "not on this machine. Normalization stats still come from --run-dir, so "
                        "the numbers stay comparable across servers but are off-distribution.")
    p.add_argument("--root-dir",   type=str,   default=None, help="Override the dataset root dir.")
    p.add_argument("--save-preds", action="store_true",
                   help="Dump per-episode gt/pred chunks to .npz — lets you check offline that a "
                        "batched server routed each client its OWN prediction (cross-pair the L1).")
    p.add_argument("--output-dir", type=str,   default=".")
    p.add_argument("--tag",        type=str,   default="",
                   help="Suffix for output filenames, e.g. 'simple' vs 'multi'.")
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


def build_request(frame: dict, image_keys: list[str], field, dataset_name: str,
                  history: dict) -> dict:
    """Serialize one dataset frame into the RequestMessage payload the psi0 server expects."""
    observations = frame["observations"]  # PIL images, one per repack image key, in order
    assert len(observations) == len(image_keys), (
        f"{len(observations)} obs images but {len(image_keys)} image_keys {image_keys}")
    image_dict = {k: np.asarray(img, dtype=np.uint8) for k, img in zip(image_keys, observations)}

    # frame state is normalized; the server expects RAW and re-normalizes it
    state_vec = np.asarray(frame["states"], dtype=np.float32)
    if state_vec.ndim == 1:
        state_vec = state_vec[None, :]
    raw_state = np.stack([denormalize_state(s, field) for s in state_vec])  # (To, Ds)

    msg = RequestMessage(
        image=image_dict,
        instruction=frame["instruction"],
        history=history,
        state={"states": raw_state},
        condition={},
        gt_action=[],
        dataset_name=dataset_name,
        timestamp=str(time.time()),
    )
    return msg.serialize()


def parse_response(resp: requests.Response) -> np.ndarray:
    """Pull the (Ta, Da) action chunk out of a /act response, surfacing server-side errors."""
    body = resp.json()
    if isinstance(body, str):  # serve_psi0_simple's error path double-encodes '{"status": ...}'
        body = json.loads(body)
    if "action" not in body:
        raise RuntimeError(f"server returned an error: {body}")
    return np.asarray(ResponseMessage.deserialize(body).action, dtype=np.float32)


def replay_episode(url, frames, image_keys, field, dataset_name, args,
                   client_id=None, position=0):
    """Send every frame in turn and pair each response with that frame's GT chunk.

    One call == one client. Several calls running in threads == several concurrent clients,
    which is what makes the batching server actually batch.

    Returns (gt_chunks, pred_chunks, rtts_ms) index-aligned.
    """
    gt_chunks: list[np.ndarray] = []
    pred_chunks: list[np.ndarray] = []
    rtts_ms: list[float] = []
    client_id = client_id or args.client_id

    session = requests.Session()
    bar = tqdm(frames, desc=f"Replay [{client_id}]", unit="frame", position=position, leave=False)
    for k, frame in enumerate(bar):
        history = {"client_id": client_id}
        if k == 0 or args.reset_every_frame:
            history["reset"] = True

        payload = build_request(frame, image_keys, field, dataset_name, history)
        t0 = time.perf_counter()
        resp = session.post(url, json=payload, timeout=args.timeout)
        rtt = (time.perf_counter() - t0) * 1000.0
        resp.raise_for_status()

        pred = parse_response(resp)                                    # (Ta, Da)
        gt = np.asarray(frame["raw_actions"], dtype=np.float32)        # (Tp, Da) unnormalized
        assert gt.shape[-1] == pred.shape[-1], f"action dim {gt.shape} vs {pred.shape}"
        Ta = min(pred.shape[0], gt.shape[0])

        pred_chunks.append(pred[:Ta])
        gt_chunks.append(gt[:Ta])
        rtts_ms.append(rtt)

    return gt_chunks, pred_chunks, rtts_ms


def report(gt_chunks, pred_chunks, rtts_ms, dataset_name, args, url, eps_idx) -> dict:
    errors = np.stack([np.abs(g - p) for g, p in zip(gt_chunks, pred_chunks)])  # (T, Ta, Da)
    n_frames, horizon, _ = errors.shape

    mean_err = errors.mean(axis=(0, 1))                       # (Da,) over frames and horizon
    first_step_err = errors[:, 0, :].mean(axis=0)             # (Da,) first step of each chunk

    per_modality = dict(zip(ACTION_LABELS,
                            [c.mean() for c in np.split(mean_err, ACTION_SPLITS, axis=-1)]))
    per_modality_first = dict(zip(ACTION_LABELS,
                                  [c.mean() for c in np.split(first_step_err, ACTION_SPLITS, axis=-1)]))

    rtt = np.asarray(rtts_ms)
    warm = rtt[1:] if len(rtt) > 1 else rtt  # frame 0 absorbs GPU warmup

    overwatch.info(f"--- eps {eps_idx}: open-loop denormalized L1 over {n_frames} frames "
                   f"x {horizon} steps ---")
    for label in ACTION_LABELS:
        overwatch.info(f"  err_l1_{label:<12} full-chunk {per_modality[label]:.6f}   "
                       f"first-step {per_modality_first[label]:.6f}")
    overwatch.info(f"  overall mean L1: {mean_err.mean():.6f}")
    overwatch.info(f"round-trip: first {rtt[0]:.0f} ms | warm {warm.mean():.1f} ± {warm.std():.1f} ms "
                   f"(n={len(warm)})")

    summary = {
        "url": url,
        "dataset": dataset_name,
        "eps_idx": eps_idx,
        "n_frames": int(n_frames),
        "horizon": int(horizon),
        "reset_every_frame": bool(args.reset_every_frame),
        "l1_full_chunk": {k: float(v) for k, v in per_modality.items()},
        "l1_first_step": {k: float(v) for k, v in per_modality_first.items()},
        "l1_overall": float(mean_err.mean()),
        "rtt_ms": {"first": float(rtt[0]), "warm_mean": float(warm.mean()),
                   "warm_std": float(warm.std())},
    }

    # Per-modality L1 over the episode + error growth along the chunk horizon.
    curves = {label: c.mean(axis=(1, 2))
              for label, c in zip(ACTION_LABELS, np.split(errors, ACTION_SPLITS, axis=-1))}
    n_rows = len(curves) + 1
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 2.2 * n_rows), sharex=False)
    for ax, (label, curve) in zip(axes, curves.items()):
        ax.plot(curve)
        ax.set_title(f"{label}  (mean {curve.mean():.4f})", fontsize=9)
        ax.set_ylabel("L1")
        ax.grid(True, alpha=0.3)
    axes[-2].set_xlabel("Frame step in episode")
    axes[-1].plot(errors.mean(axis=(0, 2)))
    axes[-1].set_title("L1 vs position within the predicted chunk", fontsize=9)
    axes[-1].set_xlabel("Step within chunk")
    axes[-1].grid(True, alpha=0.3)
    plt.suptitle(
        f"psi0 open-loop  |  {dataset_name}  |  Episode {eps_idx}  |  {url}"
        f"\noverall L1 {mean_err.mean():.4f}  |  warm RTT {warm.mean():.0f} ms  "
        f"|  {n_frames} frames x {horizon} steps",
        y=1.005,
    )
    plt.tight_layout()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    plot_path = out_dir / f"openloop_eps{eps_idx}{suffix}.png"
    json_path = out_dir / f"openloop_eps{eps_idx}{suffix}.json"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    json_path.write_text(json.dumps(summary, indent=2))
    if args.save_preds:
        np.savez_compressed(out_dir / f"openloop_eps{eps_idx}{suffix}.npz",
                            gt=np.stack(gt_chunks), pred=np.stack(pred_chunks))
    overwatch.info(f"Plot saved to {plot_path}  |  summary saved to {json_path}")
    return summary


def main():
    args = parse_args()

    launch_config = load_config(args.run_dir)
    seed_everything(launch_config.seed or 42)

    url = f"http://{args.host}:{args.port}/act"
    info = requests.get(f"http://{args.host}:{args.port}/info", timeout=30).json()
    overwatch.info(f"Server /info: {json.dumps(info)}")
    assert info["action"]["action_dim"] == launch_config.model.action_dim, (  # type: ignore
        "server action_dim does not match --run-dir; are they the same run?")

    # Build the dataset with just the VLM processor (CPU) — no model, so no GPU contention.
    from psi.config.data_lerobot import LerobotDataConfig
    from psi.models.psi0 import QWEN3VL_VARIANT

    data_cfg: LerobotDataConfig = launch_config.data  # type: ignore
    if args.root_dir:
        data_cfg.root_dir = args.root_dir
    if args.repo_id:
        data_cfg.train_repo_ids = [args.repo_id]
        data_cfg.val_repo_ids = [args.repo_id]
        overwatch.warning(f"dataset overridden to {data_cfg.root_dir}/{args.repo_id} — L1 is "
                          f"off-distribution vs the training data, use it for A/B only")
    vlm_processor = AutoProcessor.from_pretrained(QWEN3VL_VARIANT)
    dataset = data_cfg(split=args.split, transform_kwargs=dict(vlm_processor=vlm_processor, no_aug=True))
    overwatch.info(f"{args.split} dataset size: {len(dataset)}")

    repack = launch_config.data.transform.repack          # type: ignore
    image_keys: list[str] = getattr(repack, "image_keys", None) or args.image_keys
    field = launch_config.data.transform.field            # type: ignore
    assert field.normalize_state, "server asserts normalize_state; field has it disabled"

    # Slice the requested episodes like the open-loop reference does.
    episode_index = dataset.raw_dataset.base_dataset.episode_data_index
    episodes: dict[int, list] = {}
    for eps in args.eps_idx:
        start = int(episode_index["from"][eps].item())
        end   = int(episode_index["to"][eps].item())
        idxs = list(range(start, end, args.stride))
        if args.max_frames:
            idxs = idxs[: args.max_frames]
        overwatch.info(f"Episode {eps}: frames [{start}, {end}) -> replaying {len(idxs)} "
                       f"(stride {args.stride})")
        episodes[eps] = [dataset[i] for i in
                         tqdm(idxs, desc=f"Loading eps {eps}", unit="frame")]
        if not episodes[eps]:
            raise RuntimeError(f"No frames found for episode index {eps}.")

    dataset_name = episodes[args.eps_idx[0]][0].get("dataset_name")
    overwatch.info(f"image_keys: {image_keys}  |  action_dim: {launch_config.model.action_dim}  "  # type: ignore
                   f"|  server exec horizon: {info['action']['action_exec_horizon']}")

    def _one(eps, position):
        return eps, replay_episode(url, episodes[eps], image_keys, field, dataset_name, args,
                                   client_id=f"{args.client_id}-eps{eps}", position=position)

    t0 = time.perf_counter()
    if len(episodes) == 1 or args.sequential:
        mode = "sequential"
        results = [_one(eps, i) for i, eps in enumerate(episodes)]
    else:
        # Concurrent clients — this is what gives the batching server something to batch.
        mode = "concurrent"
        with ThreadPoolExecutor(max_workers=len(episodes)) as ex:
            futs = [ex.submit(_one, eps, i) for i, eps in enumerate(episodes)]
            results = [f.result() for f in futs]
    wall = time.perf_counter() - t0

    summaries = {}
    for eps, (gt_chunks, pred_chunks, rtts_ms) in results:
        summaries[eps] = report(gt_chunks, pred_chunks, rtts_ms, dataset_name, args, url, eps)
        summaries[eps]["mode"] = mode

    n_req = sum(s["n_frames"] for s in summaries.values())
    overwatch.info(f"{mode}: {len(episodes)} client(s), {n_req} requests in {wall:.1f}s "
                   f"({n_req / wall:.2f} req/s aggregate)")
    if len(summaries) > 1:
        combined = Path(args.output_dir) / (
            f"openloop_{mode}{'_' + args.tag if args.tag else ''}.json")
        combined.write_text(json.dumps(
            {"mode": mode, "wall_s": wall, "requests_per_s": n_req / wall,
             "episodes": {str(k): v for k, v in summaries.items()}}, indent=2))
        overwatch.info(f"Combined summary saved to {combined}")


if __name__ == "__main__":
    main()
