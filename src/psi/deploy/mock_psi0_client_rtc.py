"""Mock WebSocket client for serve_psi0_sonic.py — replays a validation episode and reports
per-modality denormalized L1 vs the open-loop reference (examples/openloop/psi0_inference_g1neck.py).

Each prediction is paired to its obs via the server's monotonic `action_version` (see
run_episode). Note this measures the RTC executed-action error (one stale-chunk step), not
the open-loop full-chunk number — same ballpark, different quantity.

Usage:
    serve_psi0_sonic --policy psi0 --port 8014 --ckpt-step 40000 --rtc \
        --run-dir .runs/finetune/g1neck.sonic.flow1000.cosine.lr1.0e-04.b128.gpus8.2606211349
    python src/psi/deploy/mock_psi0_client_rtc.py \
        --run-dir .runs/finetune/g1neck.sonic.flow1000.cosine.lr1.0e-04.b128.gpus8.2606211349 \
        --host localhost --port 8014 --eps-idx 18 --target-hz 30

With --save-replay the executed actions are also dumped as a record_sonic.py-style pickle,
so the episode the policy just produced can be re-executed in the MuJoCo sim:

    ... --save-replay --save-replay-gt          # pred + GT baseline
    hongyi-wbc/sim_replay/replay_in_mujoco.sh <out>/replay_eps18_pred.pkl

The dump undoes the repack permutation (psi trains on token-first actions, the WBC expects
hand+neck+token) and FSQ-quantizes the token channel, exactly as the live client does.
"""

import argparse
import asyncio
import json
import pickle
import re
import time
from pathlib import Path
import requests

from dotenv import load_dotenv

# Before anything that touches huggingface_hub: it snapshots HF_HOME/HF_TOKEN at import
# time, so a .env loaded after this point never takes effect and the cache silently misses.
_env_path = Path(".env")
_env_loaded = _env_path.exists()
if _env_loaded:
    load_dotenv(_env_path)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import websockets
from tqdm.auto import tqdm
from transformers import AutoProcessor

from psi.utils import parse_args_to_tyro_config, seed_everything, apply_legacy_model_config_defaults
from psi.utils.overwatch import initialize_overwatch
from psi.config.config import LaunchConfig
from psi.deploy.helpers import RequestMessage, ResponseMessage

overwatch = initialize_overwatch(__name__)

if not _env_loaded:
    overwatch.warning(".env not found in the current directory — environment variables may be missing.")

# Modality split of the 80-D action, matching examples/openloop/psi0_inference_g1neck.py.
ACTION_SPLITS = [64, 78]
ACTION_LABELS = ["latent_action", "hand_joints", "neck_joints"]

# --- replay-pickle constants, mirroring hongyi-wbc/g1_sonic_client.py ---
HAND_DIM, NECK_DIM, TOKEN_DIM = 14, 2, 64
QPOS_DIM = 29                      # states = qpos(29) + hands(14) + neck(2), per meta/modality.json
# FSQ grid the WBC expects on the token channel. Recorded dataset tokens sit exactly on
# it (16 levels); a policy prediction is continuous, so every live client quantizes
# before publishing (psi_rtc_sonic_client.py:360) and a replay dump must do the same.
FSQ_MIN, FSQ_MAX, FSQ_STEP = -0.625, 0.625, 0.0625


def parse_args():
    p = argparse.ArgumentParser(description="Mock RTC WebSocket client for serve_psi0_sonic.py")
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
                   help="Observation send frequency (Hz); the server control loop runs at 30 Hz.")
    p.add_argument("--timeout",   type=float, default=60.0,
                   help="Per-action receive timeout (s); first action waits through model warmup.")
    p.add_argument("--output-dir", type=str,  default=None)
    p.add_argument("--rollout", type=Path, default=None)
    p.add_argument("--max-allowed-frames", type=int, default=None)
    p.add_argument("--save-replay", action="store_true",
                   help="Dump the predicted actions as a record_sonic.py-style pickle that "
                        "hongyi-wbc/sim_replay/replay_in_mujoco.sh can stream into the sim.")
    p.add_argument("--save-replay-gt", action="store_true",
                   help="Also dump the ground-truth actions in the same format, as the A/B "
                        "baseline (equivalent to sim_replay/lerobot_to_replay.py).")
    p.add_argument("--no-fsq", action="store_true",
                   help="Do not FSQ-quantize the predicted token channel when dumping a replay "
                        "pickle. Off-grid tokens are off-distribution for the WBC — debug only.")
    return p.parse_args()


def load_vlm_processor(variant: str):
    """Load the VLM processor from the local HF cache, without asking the hub first.

    processing_auto's initial cached_file call ignores local_files_only, so it always HEADs
    the hub; when that connection stalls it burns ~80s in retries before the client has even
    connected, for files that are already on disk. Force offline for the cached attempt and
    only reach the network if the cache genuinely lacks the processor.
    """
    from huggingface_hub import constants as hf_constants

    was_offline = hf_constants.HF_HUB_OFFLINE
    hf_constants.HF_HUB_OFFLINE = True
    try:
        return AutoProcessor.from_pretrained(variant)
    except Exception:
        overwatch.warning(f"{variant} not in the local HF cache — fetching it from the hub")
        hf_constants.HF_HUB_OFFLINE = was_offline
        return AutoProcessor.from_pretrained(variant)
    finally:
        hf_constants.HF_HUB_OFFLINE = was_offline


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


def fsq_quantize(x: np.ndarray) -> np.ndarray:
    """Snap the token channel onto the WBC's FSQ grid (same math as every hongyi-wbc client)."""
    q = np.round(np.clip(x, FSQ_MIN, FSQ_MAX) / FSQ_STEP) * FSQ_STEP
    return np.clip(q, FSQ_MIN, FSQ_MAX).astype(np.float32)


_ACTION_KEY_RE = re.compile(r"^action\[(\d*):(\d*)\]$")


def raw_action_spans(action_keys: list[str], action_dim: int) -> list[tuple[int, int]]:
    """Parse repack.action_keys (e.g. ["action[16:80]", "action[:14]", "action[14:16]"]) into
    the raw-dataset spans they concatenate, in model-vector order."""
    spans = []
    for key in action_keys:
        m = _ACTION_KEY_RE.match(str(key).strip())
        assert m, (f"action key {key!r} is not an action[a:b] slice — cannot invert the "
                   f"permutation into the replay layout")
        lo = int(m.group(1)) if m.group(1) else 0
        hi = int(m.group(2)) if m.group(2) else action_dim
        spans.append((lo, hi))
    return spans


def model_to_raw_action(actions: np.ndarray, action_keys: list[str]) -> np.ndarray:
    """Undo the repack permutation: model-order actions -> raw dataset action layout.

    psi trains on a reordered action vector (token first, per repack.action_keys), while the
    replay pickle -- and TokenPublisher._token_start -- want the recorded layout
    hand(14) + neck(2) + token(64). Inverting the declared spans keeps this correct for the
    neckless (78-D) runs too, instead of hardcoding one run's permutation.
    """
    actions = np.asarray(actions, dtype=np.float32)
    spans = raw_action_spans(action_keys, actions.shape[1])
    width = max(hi for _, hi in spans)
    assert sum(hi - lo for lo, hi in spans) == actions.shape[1], (
        f"action_keys cover {sum(hi - lo for lo, hi in spans)} dims but actions are "
        f"{actions.shape[1]}-D — run-dir mismatch?")

    raw = np.zeros((len(actions), width), dtype=np.float32)
    filled = np.zeros(width, dtype=bool)
    pos = 0
    for lo, hi in spans:
        raw[:, lo:hi] = actions[:, pos:pos + (hi - lo)]
        filled[lo:hi] = True
        pos += hi - lo
    assert filled.all(), f"action_keys leave raw dims {np.flatnonzero(~filled).tolist()} unset"
    return raw


def write_replay_pickle(path: Path, actions: np.ndarray, states: np.ndarray, action_keys: list[str],
                        fps: float, task: str, quantize: bool = True) -> None:
    """Write a record_sonic.py-style replay pickle from model-order actions + raw states.

    Mirrors sim_replay/lerobot_to_replay.py: replay_sonic.py itself only reads include_neck,
    freq and ticks{t, action}; the measured-state fields come from the 45-D observation so the
    pickle stays self-describing. Note those states are what the policy SAW, not the result of
    executing the predicted actions.
    """
    raw = model_to_raw_action(actions, action_keys)
    width = raw.shape[1]
    include_neck = width == HAND_DIM + NECK_DIM + TOKEN_DIM
    assert include_neck or width == HAND_DIM + TOKEN_DIM, (
        f"raw action width {width} is neither the 78-D nor the 80-D sonic layout")

    token_start = HAND_DIM + (NECK_DIM if include_neck else 0)
    if quantize:
        raw[:, token_start:token_start + TOKEN_DIM] = fsq_quantize(
            raw[:, token_start:token_start + TOKEN_DIM])

    n = len(raw)
    # Uniform grid: every frame is sent in order and paired 1:1, so index k IS episode frame k.
    t = np.arange(n, dtype=np.float64) / fps
    ticks = {
        "t": t,
        "action": raw,
        "base_quat": np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1)),
        "is_repeat": np.zeros(n, dtype=bool),
    }

    states = np.asarray(states, dtype=np.float32)
    expected_state = QPOS_DIM + HAND_DIM + (NECK_DIM if include_neck else 0)
    if states.shape[1] == expected_state:
        ticks["qpos"] = states[:, :QPOS_DIM]
        ticks["left_hand_q"] = states[:, QPOS_DIM:QPOS_DIM + 7]
        ticks["right_hand_q"] = states[:, QPOS_DIM + 7:QPOS_DIM + HAND_DIM]
        if include_neck:
            ticks["neck_state"] = states[:, QPOS_DIM + HAND_DIM:expected_state]
    else:
        # replay_sonic.py never reads these; the ticks above are enough to stream.
        overwatch.warning(f"state dim {states.shape[1]} != {expected_state} — writing the replay "
                          f"pickle without measured qpos/hand/neck fields")

    data = {
        "include_neck": include_neck,
        "action_dim": int(width),
        "freq": int(round(fps)),
        "ticks": ticks,
        "commands": [],
        "task": str(task),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=4)

    tok = raw[:, token_start:token_start + TOKEN_DIM]
    overwatch.info(f"wrote {path}: {n} ticks @ {data['freq']} Hz ({t[-1] - t[0]:.1f}s), "
                   f"include_neck={include_neck}, token range "
                   f"[{tok.min():.3f}, {tok.max():.3f}]{'' if quantize else ' (unquantized)'}")


def _slim_frame(item: dict) -> dict:
    """Drop the VLM tensors from a dataset item, keeping only the replay inputs.

    `observations` becomes uint8 arrays (build_request converts them anyway) and only the
    first GT action step is retained — that is all run_episode pairs against."""
    return {
        "observations": [np.asarray(img, dtype=np.uint8) for img in item["observations"]],
        "states": np.asarray(item["states"], dtype=np.float32),
        "raw_actions": np.asarray(item["raw_actions"], dtype=np.float32)[:1],
        "instruction": item["instruction"],
        "dataset_name": item.get("dataset_name"),
    }


def build_request(frame: dict, image_keys: list[str], field, dataset_name: str
                  ) -> tuple[str, np.ndarray]:
    """Serialize one dataset frame into the RequestMessage JSON the psi0 server expects.

    Returns (payload, raw_state) — the caller keeps the state so a replay dump records exactly
    what was sent, index-aligned with the actions it came back with."""
    observations = frame["observations"]  # np.uint8 arrays, one per repack image key, in order
    # observations = [np.load("recorded_obs_0.npy")] 
    assert len(observations) == len(image_keys), (
        f"{len(observations)} obs images but {len(image_keys)} image_keys {image_keys}")
    image_dict = {k: np.asarray(img, dtype=np.uint8) for k, img in zip(image_keys, observations)}

    # frame state is normalized; the server expects RAW and re-normalizes it
    state_vec = np.asarray(frame["states"], dtype=np.float32)
    # state_vec = np.load("recorded_state_0.npy")
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
    return json.dumps(msg.serialize()), raw_state


def _parse_version(text: str, default: int) -> tuple[int, np.ndarray]:
    """Pull (action_version, action) from a raw message — `version` must be read off the raw
    dict since ResponseMessage.deserialize drops it."""
    resp_dict = json.loads(text)
    version = int(resp_dict.get("version", default))
    action = np.asarray(ResponseMessage.deserialize(resp_dict).action).reshape(-1)
    return version, action


async def _flush_stale(ws, last_version: int, flush_timeout: float = 0.005) -> int:
    """Drain actions still buffered from the previous obs; return the newest version seen."""
    while True:
        try:
            text = await asyncio.wait_for(ws.recv(), timeout=flush_timeout)
        except asyncio.TimeoutError:
            return last_version
        v, _ = _parse_version(text, default=last_version)
        last_version = max(last_version, v)


async def run_episode(uri, frames, image_keys, field, dataset_name, target_hz, recv_timeout):
    """Replay the episode, pairing each frame with the action the server produced from it.

    Per frame: flush stale buffered actions, send the frame, then block for the first action
    with a strictly newer `action_version` (produced by a tick that read this frame). Version
    gaps are counted, not mis-paired. No warmup phase — the first frame's recv absorbs the
    server's synchronous model build, so `--timeout` must cover it.

    Returns (gt_actions, pred_actions, sent_states, send_times, recv_times), paired
    index-aligned.
    """
    gt_first = [np.asarray(f["raw_actions"], dtype=np.float32)[0] for f in frames]  # first GT step (Da,)
    gt_actions:   list[np.ndarray] = []
    pred_actions: list[np.ndarray] = []
    sent_states:  list[np.ndarray] = []
    send_times:   list[float]      = []
    recv_times:   list[float]      = []
    dropped_ticks = 0   # control ticks the server produced but coalesced away (version gaps)
    stale_skipped = 0   # buffered actions from earlier obs we discarded

    async with websockets.connect(uri, max_size=16 * 1024 * 1024) as ws:
        overwatch.info(f"Connected to {uri} — first frame absorbs the server-side model build")

        last_version = 0  # action_version starts at 0; first action is version 1
        interval = 1.0 / target_hz
        for k, frame in enumerate(tqdm(frames, desc="Replaying frames", unit="frame")):
            payload, raw_state = build_request(frame, image_keys, field, dataset_name)

            last_version = await _flush_stale(ws, last_version)
            send_t = time.perf_counter()
            await ws.send(payload)

            action = None
            try:
                while True:  # accept the first action from a tick that read this frame
                    text = await asyncio.wait_for(ws.recv(), timeout=recv_timeout)
                    version, act = _parse_version(text, default=last_version + 1)
                    if version <= last_version:
                        stale_skipped += 1
                        continue
                    if version > last_version + 1:
                        dropped_ticks += version - last_version - 1
                    last_version = version
                    action = act
                    break
            except asyncio.TimeoutError:
                overwatch.warning(f"Timeout after {recv_timeout}s at frame {k} — stopping "
                                  f"({len(pred_actions)}/{len(frames)} paired)")
                break

            recv_times.append(time.perf_counter())
            send_times.append(send_t)
            pred_actions.append(action)
            gt_actions.append(gt_first[k])
            sent_states.append(raw_state)

            sleep = interval - (time.perf_counter() - send_t)
            if sleep > 0:
                await asyncio.sleep(sleep)

    overwatch.info(f"Stream sync: {stale_skipped} stale actions skipped, "
                   f"{dropped_ticks} control ticks dropped/coalesced over {len(pred_actions)} pairs")
    return gt_actions, pred_actions, sent_states, send_times, recv_times


def main():
    args = parse_args()

    launch_config = load_config(args.run_dir)
    seed_everything(launch_config.seed or 42)
    
    info = requests.get(f"http://{args.host}:{args.port}/info", timeout=30).json()
    overwatch.info(f"Server /info: {json.dumps(info)}")
    assert info["action"]["action_dim"] == launch_config.model.action_dim, (  # type: ignore
        "server action_dim does not match --run-dir; are they the same run?")
    
    assert "ckpt_step" in info, "Server info does not contain 'ckpt_step'"
    
    if not args.output_dir:
        args.output_dir = f"{args.run_dir}/checkpoints/ckpt_{info['ckpt_step']}"

    # Build the dataset with just the VLM processor (CPU) — no model, so no GPU contention.
    from psi.config.data_lerobot import LerobotDataConfig
    from psi.models.psi0 import QWEN3VL_VARIANT

    data_cfg: LerobotDataConfig = launch_config.data  # type: ignore
    
    # load inference rollout data
    if args.rollout is not None:
        data_cfg.root_dir = args.rollout.parent
        data_cfg.train_repo_ids = [args.rollout.name]

    vlm_processor = load_vlm_processor(QWEN3VL_VARIANT)
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
    if args.max_allowed_frames is not None:
        end = min(end, start + args.max_allowed_frames)
        
    overwatch.info(f"Episode {args.eps_idx}: frames [{start}, {end}) -> {end - start} frames")

    # Keep only what the replay actually sends. A full dataset item drags the VLM
    # processor output along (pixel_values is ~7 MB/frame for a 384x672 image), which
    # OOMs the box on a ~1k-frame episode; the slim view is ~1 MB/frame.
    frames = [_slim_frame(dataset[i])
              for i in tqdm(range(start, end), desc="Loading episode frames", unit="frame")]
    if not frames:
        raise RuntimeError("No frames found for the given episode index.")
    dataset_name = frames[0].get("dataset_name")
    overwatch.info(f"image_keys: {image_keys}  |  action_dim: {launch_config.model.action_dim}")  # type: ignore

    uri = f"ws://{args.host}:{args.port}/ws"
    gt_actions, pred_actions, sent_states, send_times, recv_times = asyncio.run(
        run_episode(uri, frames, image_keys, field, dataset_name, args.target_hz, args.timeout)
    )

    n_pairs = min(len(gt_actions), len(pred_actions))
    if n_pairs == 0:
        raise RuntimeError("No action pairs received — is the server running?")
    overwatch.info(f"Paired {n_pairs}/{len(frames)} frames with received actions")

    # Per-frame denormalized L1, split by modality like the open-loop reference.
    errors = np.stack([np.abs(gt_actions[i] - pred_actions[i]) for i in range(n_pairs)])  # (T, Da)
    mean_err = errors.mean(axis=0)                                                          # (Da,)
    per_modality = np.split(mean_err, ACTION_SPLITS, axis=-1)

    rtt_ms = np.array([(recv_times[i] - send_times[i]) * 1000 for i in range(n_pairs)])
    mean_rtt, std_rtt = float(rtt_ms.mean()), float(rtt_ms.std())

    overwatch.info("--- per-modality denormalized L1 (vs first GT step) ---")
    for label, chunk in zip(ACTION_LABELS, per_modality):
        overwatch.info(f"  denormed_err_l1_{label}: {chunk.mean():.6f}")
    overwatch.info(f"round-trip latency: {mean_rtt:.1f} ± {std_rtt:.1f} ms  (n={n_pairs})")

    if args.save_replay or args.save_replay_gt:
        # Recorded rate, not --target-hz: the pickle is replayed at the episode's own rate.
        meta = getattr(dataset.raw_dataset, "meta", None)
        fps = float(getattr(meta, "fps", args.target_hz)) if meta is not None else args.target_hz
        task = frames[0]["instruction"]
        states = np.stack(sent_states[:n_pairs])
        out_dir = Path(args.output_dir)
        # Name the pickles after the rollout episode dir if given, else the split and episode index.
        stem = (args.rollout.name if args.rollout is not None
                else f"openloop_rtc_{args.split}_eps{args.eps_idx}")
        if args.save_replay:
            write_replay_pickle(out_dir / f"{stem}_pred.pkl",
                                np.stack(pred_actions[:n_pairs]), states, repack.action_keys,
                                fps, task, quantize=not args.no_fsq)
        if args.save_replay_gt:
            # GT tokens are already on the FSQ grid, so quantizing is a no-op -- keep it off
            # to stay byte-identical to sim_replay/lerobot_to_replay.py's output.
            write_replay_pickle(out_dir / f"{stem}_gt.pkl",
                                np.stack(gt_actions[:n_pairs]), states, repack.action_keys,
                                fps, task, quantize=False)

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
        ax.plot(curve, label=label)
        # Episode average as a flat reference, so a frame reads as above/below the
        # mean at a glance instead of against the per-panel autoscaled y-range.
        mean_l1 = float(curve.mean())
        ax.axhline(mean_l1, color="tab:red", linestyle="--", linewidth=1.2,
                   label=f"mean {mean_l1:.4f}")
        ax.set_title(f"{label}  (mean {mean_l1:.4f})")
        ax.set_ylabel("Denormalized L1")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Frame step in episode")
    plt.suptitle(
        f"psi0 RTC mock  |  {dataset_name}  |  Episode {args.eps_idx}  |  {args.host}:{args.port}"
        f"\nRound-trip {mean_rtt:.1f} ± {std_rtt:.1f} ms  (n={n_pairs} pairs)",
        y=1.01,
    )
    plt.tight_layout()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.split}_eps{args.eps_idx}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    overwatch.info(f"Plot saved to {out_path}")


if __name__ == "__main__":
    main()
