import argparse
import json
import logging
import math
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from datasets import Dataset, Features, Sequence, Value
from datasets.utils.logging import set_verbosity_error
from huggingface_hub import create_repo, create_tag, upload_large_folder
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SOURCE_VECTOR_DIM = 43
MOTION_TOKEN_DIM = 64
PSI_STATE_DIM = 43
PSI_ACTION_DIM = 78
QPOS_SLICES = ((0, 15), (15, 22), (29, 36))
HAND_SLICES = ((22, 29), (36, 43))
SOURCE_TO_ACTUATED_HAND7 = (4, 5, 6, 0, 1, 2, 3)
ACTUATED_TO_SOURCE_HAND7 = (3, 4, 5, 6, 0, 1, 2)

CODE_VERSION = "v2.1"
FPS = 30

# disable_progress_bar()
set_verbosity_error()

# Silence pyarrow/parquet messages
logging.getLogger("pyarrow").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)

# --- source (collection) column names ---
SRC_VIDEO_KEY = "observation.images.ego_view"
SRC_STATE = "observation.state"            # 43, joint-angle layout below
SRC_ACTION_WBC = "action.wbc"              # 43, same layout as state
SRC_MOTION_TOKEN = "action.motion_token"   # 64
SRC_TELEOP_LEFT_HAND = "teleop.left_hand_joints"   # 7, actuated order
SRC_TELEOP_RIGHT_HAND = "teleop.right_hand_joints"  # 7, actuated order

@dataclass
class InfoDict:
    codebase_version: str
    robot_type: str
    total_episodes: int
    total_frames: int
    total_tasks: int
    total_videos: int
    total_chunks: int
    chunks_size: int
    fps: int
    data_path: str
    video_path: str
    features: Dict[str, Any]


def append_jsonl_line_atomic(path: Path, obj: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, separators=(",", ":"), ensure_ascii=False) + "\n"
    fd = os.open(str(path), os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "a", encoding="utf-8") as f:
        try:
            import fcntl

            fcntl.flock(f, fcntl.LOCK_EX)
        except Exception:
            pass
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def take_slices(values: np.ndarray, slices: tuple[tuple[int, int], ...]) -> np.ndarray:
    return np.concatenate([values[start:end] for start, end in slices])


def source_full_hand14(source43: np.ndarray) -> np.ndarray:
    source_hands = take_slices(source43, HAND_SLICES).reshape(2, 7)
    return source_hands[:, list(SOURCE_TO_ACTUATED_HAND7)].reshape(14)


def source_hand7_from_actuated(hand7: np.ndarray) -> np.ndarray:
    hand = np.asarray(hand7, dtype=np.float32)
    if hand.shape != (7,):
        raise ValueError(f"hand7 must have shape (7,), got {hand.shape}")
    return hand[list(ACTUATED_TO_SOURCE_HAND7)]


def pack_psi_state(source43: np.ndarray) -> np.ndarray:
    return np.concatenate([take_slices(source43, QPOS_SLICES), source_full_hand14(source43)])


class Sonic2LeRobotConverter:
    """Convert a SONIC deploy-collected LeRobot dataset into the Psi0 training format.

    Source has the GR00T-WholeBodyControl schema (observation.state[43],
    action.wbc[43], action.motion_token[64], teleop.* columns) with a single
    video key ``observation.images.ego_view``. Output matches multi_task_psi0_sonic:
    states[43]=qpos(29)+hand(14), action[78]=motion_token(64)+hand(14), one
    ``observation.images.egocentric`` video. Frames are kept 1:1; video is copied.
    """

    def __init__(self):
        self.features = Features(
            {
                "states": Sequence(Value("float32")),
                "action": Sequence(Value("float32")),
                "timestamp": Value("float32"),
                "frame_index": Value("int64"),
                "episode_index": Value("int64"),
                "index": Value("int64"),
                "task_index": Value("int64"),
                "next.done": Value("bool"),
            }
        )
        self.tasks_meta: Dict[int, str] = {}                  # task_index -> description
        self.episode_sources: List[Tuple[int, Path, Path, int]] = []  # (task_idx, parquet, video, out_ep)
        self.lengths_by_episode: Dict[int, int] = {}
        self.chunks_size: int = 1000
        self.end_effector: str = "dex3"
        self.num_episodes: int = 0
        self.total_frames: int = 0

    def build_obs(self, state43: np.ndarray) -> Dict[str, Any]:
        states = pack_psi_state(state43)
        return {"states": states.astype(np.float32).tolist()}  # 29 + 14 = 43

    def build_act(
        self,
        token64: np.ndarray,
        wbc43: np.ndarray,
        *,
        hand14_override: np.ndarray | None = None,
    ) -> List[float]:
        if hand14_override is not None:
            token = np.asarray(token64, dtype=np.float32)
            hand = np.asarray(hand14_override, dtype=np.float32)
            if token.shape != (MOTION_TOKEN_DIM,) or not np.all(np.isfinite(token)):
                raise ValueError(f"motion_token64 must be finite ({MOTION_TOKEN_DIM},), got {token.shape}")
            if hand.shape != (14,) or not np.all(np.isfinite(hand)):
                raise ValueError(f"hand14_override must be finite (14,), got {hand.shape}")
            return np.concatenate([token, hand]).astype(np.float32).tolist()
        action = np.concatenate([token64, source_full_hand14(wbc43)])
        return action.astype(np.float32).tolist()  # 64 + 14 = 78

    def make_one_episode(
        self,
        task_index: int,
        episode_index: int,
        src_parquet: Path,
        src_video: Path,
        out_base: Path,
        chunks_size: int,
    ) -> Tuple[int, int]:
        chunk_path = out_base / f"chunk-{episode_index // chunks_size:03d}"
        chunk_path.mkdir(parents=True, exist_ok=True)
        parquet_path = chunk_path / f"episode_{episode_index:06d}.parquet"

        ego_dir = out_base.parent / "videos" / f"chunk-{episode_index // chunks_size:03d}" / "observation.images.egocentric"
        ego_dir.mkdir(parents=True, exist_ok=True)
        vid_path = ego_dir / f"episode_{episode_index:06d}.mp4"

        df = pd.read_parquet(src_parquet)
        n = len(df)
        assert n > 0, f"empty parquet {src_parquet}"
        state = np.vstack([np.asarray(x, dtype=np.float64) for x in df[SRC_STATE]])
        wbc = np.vstack([np.asarray(x, dtype=np.float64) for x in df[SRC_ACTION_WBC]])
        token = np.vstack([np.asarray(x, dtype=np.float64) for x in df[SRC_MOTION_TOKEN]])
        if state.shape[1] != SOURCE_VECTOR_DIM:
            raise ValueError(f"{src_parquet}: {SRC_STATE} must be {SOURCE_VECTOR_DIM}-D, got {state.shape[1]}")
        if wbc.shape[1] != SOURCE_VECTOR_DIM:
            raise ValueError(f"{src_parquet}: {SRC_ACTION_WBC} must be {SOURCE_VECTOR_DIM}-D, got {wbc.shape[1]}")
        if token.shape[1] != MOTION_TOKEN_DIM:
            raise ValueError(f"{src_parquet}: {SRC_MOTION_TOKEN} must be {MOTION_TOKEN_DIM}-D, got {token.shape[1]}")

        hand_targets = None
        if self.end_effector == "dex1_virtual14":
            missing_hand_columns = [
                key
                for key in (SRC_TELEOP_LEFT_HAND, SRC_TELEOP_RIGHT_HAND)
                if key not in df.columns
            ]
            if missing_hand_columns:
                raise ValueError(
                    f"{src_parquet}: dex1_virtual14 action requires "
                    + ", ".join(missing_hand_columns)
                )
            left_hand = np.vstack(
                [np.asarray(x, dtype=np.float32) for x in df[SRC_TELEOP_LEFT_HAND]]
            )
            right_hand = np.vstack(
                [np.asarray(x, dtype=np.float32) for x in df[SRC_TELEOP_RIGHT_HAND]]
            )
            if left_hand.shape != (n, 7) or right_hand.shape != (n, 7):
                raise ValueError(
                    f"{src_parquet}: teleop hand targets must be ({n}, 7), "
                    f"got {left_hand.shape} and {right_hand.shape}"
                )
            hand_targets = np.concatenate([left_hand, right_hand], axis=1)
            if not np.all(np.isfinite(hand_targets)):
                raise ValueError(f"{src_parquet}: teleop hand targets contain NaN/Inf")

        rows: List[Dict[str, Any]] = []
        for i in range(n):
            rows.append(
                {
                    **self.build_obs(state[i]),
                    "action": self.build_act(
                        token[i],
                        wbc[i],
                        hand14_override=None if hand_targets is None else hand_targets[i],
                    ),
                    "timestamp": i * (1.0 / FPS),
                    "frame_index": i,
                    "episode_index": episode_index,
                    "index": i,
                    "task_index": task_index,
                    "next.done": (i == n - 1),
                }
            )

        tmp_dir = out_base / f"_tmp_ep_{episode_index:06d}"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        parquet_tmp = tmp_dir / "episode.parquet"
        Dataset.from_list(rows, features=self.features).to_parquet(str(parquet_tmp))
        os.replace(parquet_tmp, parquet_path)
        shutil.copyfile(src_video, vid_path)  # already h264/yuv420p
        shutil.rmtree(tmp_dir, ignore_errors=True)

        # per-episode action/timestamp stats -> episodes_stats.jsonl
        acts = np.array([r["action"] for r in rows], dtype=np.float32)
        episode_stats = {
            "episode_index": episode_index,
            "stats": {
                "action": {
                    "min": acts.min(0).tolist(), "max": acts.max(0).tolist(),
                    "mean": acts.mean(0).tolist(), "std": acts.std(0).tolist(),
                    "count": [n],
                },
                "timestamp": {
                    "min": [0.0], "max": [(n - 1) / FPS],
                    "mean": [((n - 1) / 2) / FPS],
                    "std": [n / (2 * FPS * math.sqrt(3))], "count": [n],
                },
            },
        }
        append_jsonl_line_atomic(out_base.parent / "meta" / "episodes_stats.jsonl", episode_stats)
        return episode_index, n

    def run(
        self,
        data_root: Path,
        work_dir: Path,
        chunks_size: int,
        num_workers: int,
        robot_type: str,
        end_effector: str,
    ):
        self.chunks_size = chunks_size
        self.end_effector = end_effector
        print(f"Using end_effector={self.end_effector}")
        data_dir = work_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        # task string -> index, and episode -> task description, from the source meta
        task_to_idx: Dict[str, int] = {}
        for r in read_jsonl(data_root / "meta" / "tasks.jsonl"):
            task_to_idx[r["task"]] = r["task_index"]
            self.tasks_meta[r["task_index"]] = r["task"]
        ep_task: Dict[int, str] = {}
        for r in read_jsonl(data_root / "meta" / "episodes.jsonl"):
            tasks = r.get("tasks", [])
            ep_task[r["episode_index"]] = tasks[0] if tasks else ""

        self.episode_sources = []
        out_ep = 0
        for pq in sorted((data_root / "data").rglob("episode_*.parquet")):
            src_ep = int(pq.stem.split("_")[1])
            desc = ep_task.get(src_ep, "")
            task_idx = task_to_idx.get(desc, 0)
            video = (
                data_root / "videos" / f"chunk-{src_ep // chunks_size:03d}"
                / SRC_VIDEO_KEY / f"episode_{src_ep:06d}.mp4"
            )
            assert video.is_file(), f"missing source video: {video}"
            self.episode_sources.append((task_idx, pq, video, out_ep))
            out_ep += 1

        print(f"Found {len(self.episode_sources)} episodes, {len(self.tasks_meta)} tasks.")
        if not self.episode_sources:
            print("No episodes found.")
            return

        with ProcessPoolExecutor(max_workers=num_workers) as ex:
            futures = [
                ex.submit(self.make_one_episode, task_idx, oi, pq, vid, data_dir, chunks_size)
                for (task_idx, pq, vid, oi) in self.episode_sources
            ]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing episodes", unit="ep"):
                ep_idx, n_frames = fut.result()
                self.lengths_by_episode[ep_idx] = n_frames

        self.num_episodes = len(self.lengths_by_episode)
        self.total_frames = sum(self.lengths_by_episode.values())
        print(f"Now total episodes: {self.num_episodes}, frames: {self.total_frames}")

    def write_meta(self, out_dir: Path):
        meta_dir = out_dir / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)

        dataset_cursor = 0
        ep_rows_meta = []
        for (task_idx, _pq, _vid, ep_index) in sorted(self.episode_sources, key=lambda x: x[3]):
            n = self.lengths_by_episode.get(ep_index, 0)
            if n <= 0:
                continue
            ep_rows_meta.append(
                {
                    "episode_index": ep_index,
                    "tasks": [task_idx],
                    "length": n,
                    "dataset_from_index": dataset_cursor,
                    "dataset_to_index": dataset_cursor + (n - 1),
                    "robot_type": "g1",
                    "instruction": self.tasks_meta.get(task_idx, ""),
                }
            )
            dataset_cursor += n
        episodes_df = pd.DataFrame(ep_rows_meta).sort_values("episode_index").reset_index(drop=True)

        task_rows = [
            {"task_index": ti, "task": desc, "category": "default", "description": desc}
            for ti, desc in sorted(self.tasks_meta.items())
        ]
        tasks_df = pd.DataFrame(task_rows).sort_values("task_index").reset_index(drop=True)

        video_info = {
            "video.fps": float(FPS), "video.codec": "h264", "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False, "has_audio": False,
        }
        features_meta = {
            "observation.images.egocentric": {
                "dtype": "video", "shape": [480, 640, 3],
                "names": ["height", "width", "channel"], "video_info": video_info,
            },
            "states": {"dtype": "float32", "shape": [PSI_STATE_DIM]},
            "action": {"dtype": "float32", "shape": [PSI_ACTION_DIM]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "next.done": {"dtype": "bool", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
        }

        info = InfoDict(
            codebase_version=CODE_VERSION,
            robot_type="g1",
            total_episodes=self.num_episodes,
            total_frames=self.total_frames,
            total_tasks=len(self.tasks_meta),
            total_videos=self.num_episodes,
            total_chunks=math.ceil(self.num_episodes / self.chunks_size),
            chunks_size=self.chunks_size,
            fps=FPS,
            data_path="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            video_path="videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            features=features_meta,
        )
        (meta_dir / "info.json").write_text(json.dumps(asdict(info), indent=4))
        with open(meta_dir / "tasks.jsonl", "w") as f:
            for row in tasks_df.to_dict(orient="records"):
                json.dump(row, f)
                f.write("\n")
        with open(meta_dir / "episodes.jsonl", "w") as f:
            for row in episodes_df.to_dict(orient="records"):
                json.dump(row, f)
                f.write("\n")
        print(f"\nWrote meta and {self.num_episodes} episode(s) into: {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True,
                        help="Source SONIC-collected LeRobot dataset, e.g. ~/test_collect")
    parser.add_argument("--work-dir", type=str, default="_lerobot_build")
    parser.add_argument("--repo-id", type=str)
    parser.add_argument("--chunks-size", type=int, default=1000)
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--repo-exist-ok", action="store_true")
    parser.add_argument("--num-workers", type=int, default=os.cpu_count(), help="Max parallel workers")
    parser.add_argument("--robot-type", type=str, choices=["g1"], default="g1")
    parser.add_argument(
        "--end-effector",
        choices=["dex3", "dex1_virtual14"],
        default="dex3",
        help=(
            "Source end-effector. dex1_virtual14 preserves the dense virtual Dex3 "
            "hand representation produced by the Psi0 SONIC Dex1 wrapper."
        ),
    )
    args = parser.parse_args()
    data_root = Path(args.data_root).expanduser().resolve()
    work_dir = Path(args.work_dir).expanduser().resolve()
    if args.repo_id:
        work_dir = work_dir / args.repo_id
    for d in [work_dir / "data", work_dir / "videos", work_dir / "meta"]:
        d.mkdir(parents=True, exist_ok=True)

    pipeline = Sonic2LeRobotConverter()
    pipeline.run(
        data_root,
        work_dir,
        args.chunks_size,
        args.num_workers,
        args.robot_type,
        args.end_effector,
    )
    pipeline.write_meta(work_dir)

    if args.push:
        if not args.repo_id:
            raise ValueError("--repo-id is required when --push is set")
        create_repo(args.repo_id, repo_type="dataset", private=args.private, exist_ok=args.repo_exist_ok)
        upload_large_folder(repo_id=args.repo_id, repo_type="dataset", folder_path=str(work_dir))
        create_tag(args.repo_id, tag=CODE_VERSION, repo_type="dataset")
        print(f"\nUploaded to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
