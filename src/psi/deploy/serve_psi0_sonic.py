import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from pydantic import BaseModel
from collections import deque
import threading
import time
import copy

import os
import sys
import json
import tyro
import uvicorn
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, Any, List, Optional, TYPE_CHECKING, cast
import os.path as osp

from psi.models.psi0 import Psi0Model 

if TYPE_CHECKING:
    from transformers import CLIPTextModelWithProjection, CLIPTokenizer
    from psi.config.model_psi0 import Psi0ModelConfig

from rich.console import Console
console = Console()

# The live dashboard owns the terminal (rich.Live alt-screen), so every stray
# write would scribble over it. Set once in Server.__init__; while it is on,
# dprint/color_print route to overwatch instead, which __init__ has already
# redirected to a log file (same contract as serve_psix.py).
_DASH_ACTIVE = False


def dprint(*args, **kwargs):
    """print(), unless the dashboard is up — then log the line to the file."""
    if _DASH_ACTIVE:
        overwatch.info(" ".join(str(a) for a in args))
    else:
        print(*args, **kwargs)


def color_print(*args, markup=False, style="red"):
    if _DASH_ACTIVE:
        overwatch.info(" ".join(str(a) for a in args))
        return
    console.print(*args, style=style, markup=markup)

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from psi.config.config import LaunchConfig, ServerConfig
from psi.deploy.helpers import *
from psi.utils import parse_args_to_tyro_config, pad_to_len, seed_everything, apply_legacy_model_config_defaults
from psi.deploy.dashboard import (
    update_dashboard, reset_dashboard, dashboard_loop, log_every, redirect_logs_to_file,
)
from psi.utils.overwatch import initialize_overwatch 
overwatch = initialize_overwatch(__name__)

PREDICT_HORIZON = 30          # H
MIN_EXEC_HORIZON = 15         # s_min
DELAY_BUFFER_SIZE = 6         # delay_buffer_size
D_INIT = 6                    # d_init 
CTRL_PERIOD_SEC = 1. / 30     # 30Hz


def shift_and_pad_action_chunk(action_chunk: np.ndarray, steps: int) -> np.ndarray:
    """Drop the first `steps` actions of a (H, D) chunk and zero-pad back to (H, D).
    """
    tail = action_chunk[steps:, :]
    pad = np.zeros((steps, action_chunk.shape[1]), dtype=action_chunk.dtype)
    return np.concatenate([tail, pad], axis=0)


class PooledTextEncoderCache:
    """Deploy-side `pooled_projections` for combined_temb models.

    Loads the frozen instruction embeddings that were precomputed during training
    (saved next to run_config.json as the trainer's pooled cache). On a cache miss —
    i.e. a user passes an unseen instruction — it lazily loads the SAME frozen text
    encoder and computes+caches the embedding on the fly. Cache-hit path touches no
    model. Keys are lowercased to match the training transform.
    """

    def __init__(self, cache_path, encoder: str, encoder_path: str,
                 projection_dim: int, device):
        self.device = torch.device(device)
        self.encoder = encoder                # e.g. "clip"
        self.encoder_path = encoder_path
        self.projection_dim = projection_dim
        self._cache: Dict[str, torch.Tensor] = {}
        self._model: Optional["CLIPTextModelWithProjection"] = None
        self._tokenizer: Optional["CLIPTokenizer"] = None

        if cache_path is not None and osp.exists(cache_path):
            data = torch.load(cache_path, map_location="cpu")
            self._cache = {k: v.to(self.device) for k, v in data.items()}
            overwatch.info(f"Loaded pooled text cache ({len(self._cache)} entries) <- {cache_path}")
        else:
            overwatch.warning(
                f"No pooled text cache at {cache_path}; every instruction will be "
                f"computed on the fly with the frozen {encoder} encoder."
            )

    def _lazy_load_encoder(self):
        """Load (once) and return the frozen text encoder + its tokenizer."""
        if self._model is None or self._tokenizer is None:
            assert self.encoder == "clip", f"unsupported pooled_text_encoder: {self.encoder}"
            from transformers import CLIPTextModelWithProjection, CLIPTokenizer
            tokenizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(self.encoder_path)
            m = CLIPTextModelWithProjection.from_pretrained(self.encoder_path, dtype=torch.bfloat16)
            m.requires_grad_(False)
            m.eval()
            assert m.config.projection_dim == self.projection_dim, (
                f"pooled_projection_dim={self.projection_dim} but encoder projection_dim="
                f"{m.config.projection_dim}"
            )
            self._tokenizer = tokenizer
            self._model = m.to(self.device)  # type: ignore
            overwatch.info(f"Loaded FROZEN {self.encoder} encoder for on-the-fly pooled embeddings <- {self.encoder_path}")
        return self._model, self._tokenizer

    @torch.no_grad()
    def __call__(self, instruction: str) -> torch.Tensor:
        """instruction (str) -> pooled_projections (1, projection_dim) on device."""
        key = instruction.lower()
        emb = self._cache.get(key)
        if emb is None:  # unseen instruction -> compute with the frozen encoder and cache
            model, tokenizer = self._lazy_load_encoder()
            toks = tokenizer(
                [key], padding=True, truncation=True,
                max_length=tokenizer.model_max_length, return_tensors="pt",
            ).to(self.device)
            emb = model(**toks).text_embeds[0]
            self._cache[key] = emb
            overwatch.info(f"pooled cache MISS -> computed embedding for instruction: {instruction!r}")
        return emb.unsqueeze(0)  # (1, D)


class RealTimeChunkController:

    def __init__(
        self,
        policy: Psi0Model,
        o_first: np.ndarray,
        prediction_horizon: int = PREDICT_HORIZON,
        min_exec_horizon: int = MIN_EXEC_HORIZON,
        delay_buf_size: int = DELAY_BUFFER_SIZE,
        d_init: int = D_INIT,
        trained_rtc: bool = False,
        max_delay: int = 8,
        init_prev_action: np.ndarray | None = None,
    ):

        self.policy = policy
        self.device: torch.device = cast(torch.device, self.policy.device)
        self.H = prediction_horizon
        self.s_min = min_exec_horizon
        self.t = 0
        # Which RTC path the checkpoint supports. 
        self.trained_rtc = trained_rtc
        self.max_delay = max_delay
        # Optional normalized pseudo prev-action (Da,) shipped by the client on its first frame 
        self.init_prev_action = (
            None if init_prev_action is None
            else np.asarray(init_prev_action, dtype=np.float32).reshape(-1)
        )

        # Peak forward-pass latency, surfaced on the dashboard.
        self._max_infer_ms = 0.0

        # Warm up on a THROWAWAY chunk first. 
        update_dashboard(status="warming up: first chunk (1/3)")
        A_warm = self._predict_action(copy.deepcopy(o_first)) # (H, D)
        for i in range (2):
            update_dashboard(status=f"warming up: RTC pass {i + 1}/2 ({i + 2}/3)")
            _ = self._predict_action_rtc(
                copy.deepcopy(o_first),
                shift_and_pad_action_chunk(A_warm, self.s_min),
                d_init,
                self.s_min,
            )
        dprint("Model warmed up")

        # The chunk that will actually be executed, predicted from `o_first`. No action
        # has left the server yet (the control loop only starts once this returns), so
        # the robot is not driven by the policy during the warm-up above and `o_first`
        # still describes the pose the first chunk is spliced onto.
        if self.init_prev_action is not None:
            # Condition on the client's current-pose pseudo action, tiled over the chunk.
            # d=1: only the first row is treated as "already committed", which is exactly
            # the pose the robot is standing in right now.
            prev = np.tile(self.init_prev_action[np.newaxis, :], (self.H, 1)).astype(np.float32)
            color_print(f"[init-prev] warm-starting first chunk via RTC: prev={prev.shape}, d=1", style="cyan")
            A_first = self._predict_action_rtc(o_first, prev, 1, self.s_min) # (H, D)
        else:
            A_first = self._predict_action(o_first) # (H, D)

        update_dashboard(status="live", error=None)

        self.A_cur = A_first # (H, D)
        self.o_cur: Dict[str, Any] | None = None 

        self.Q = deque([d_init], maxlen=delay_buf_size)

        self.M = threading.Lock()
        self.C = threading.Condition(self.M)

        # Set by stop() on client disconnect to break the inference loop so the
        # controller (and its stale A_cur/t/Q) can be dropped and rebuilt fresh.
        self._stop_event = threading.Event()
        self._infer_th = threading.Thread(target=self._inference_loop, daemon=True)
        self._infer_th.start()

    def stop(self, join_timeout: float = 5.0) -> None:
        """Signal the inference thread to exit and wait for it. Wakes the thread if
        it is parked in C.wait(); if a forward pass is in flight, it exits right after."""
        self._stop_event.set()
        with self.C:
            self.C.notify_all()
        self._infer_th.join(timeout=join_timeout)
        if self._infer_th.is_alive():
            color_print(f"[RealTimeChunkController] inference thread did not stop within {join_timeout}s", style="yellow")

        
    def step(self, obs_next: Dict[str, Any]): # consume a_(t-1) and provide o_t
        with self.C:
            self.t += 1
            self.o_cur = obs_next
            self.C.notify()
            if self.t-1 >= len(self.A_cur):
                single_action = self.A_cur[-1]
                dprint("failed")
            else:
                single_action = self.A_cur[self.t - 1]
            return single_action[np.newaxis, :] # (1, D)

    def _inference_loop(self):
        while not self._stop_event.is_set():
            with self.C:
                try:
                    while self.t < self.s_min and not self._stop_event.is_set():
                        self.C.wait() # wait until notified and get the lock
                    if self._stop_event.is_set():  # woken by stop(): exit cleanly
                        return
                    s = self.t

                    assert (s-2) >= 0

                    o = copy.deepcopy(self.o_cur)
                    d = max(self.Q)
                    A_prev = shift_and_pad_action_chunk(self.A_cur, s) # (H, D)

                    inference_start = time.perf_counter()
                    self.C.release()
                    # print(f"[inference]  latency={time.perf_counter()-inference_start:.4f}s  s={s}  d={d}  self.t={self.t}")
                    A_new = self._predict_action_rtc(o, A_prev, d, s)
                    infer_ms = (time.perf_counter() - inference_start) * 1000
                    self.C.acquire()

                    self.A_cur = A_new
                    self.t = self.t - s
                    self.Q.append(self.t)

                    # `s` ticks of the previous chunk were executed while this forward
                    # ran; `self.t` is how far into the NEW chunk the control loop
                    # already is (the delay this replan cost).
                    self._max_infer_ms = max(self._max_infer_ms, infer_ms)
                    update_dashboard(infer_ms=infer_ms, max_infer_ms=self._max_infer_ms,
                                     infer_executed=s, infer_delay=d, infer_ticks=self.t,
                                     status="live")
                    log_every("inference",
                              f"[inference] latency={infer_ms / 1000:.4f}s  executed={s}  "
                              f"d={d}  ticks_since_replan={self.t}", freq=2.0)
                    # print(f"[inference]  latency={time.perf_counter()-inference_start:.4f}s  s={s}  d={d}  self.t={self.t}")
                except Exception as e:
                    dprint(f"\n[ERROR] Inference loop crashed!")
                    dprint(f"Error: {e}")
                    import traceback
                    traceback.print_exc()
                    dprint("\n[FATAL] Stopping program...")
                    update_dashboard(status="server crashed", error=f"{type(e).__name__}: {e}")
                    time.sleep(0.5)
                    os._exit(1)
    
    def _predict_action_rtc(self, o, A_prev, d, s):

        common: Dict[str, Any] = dict(
            observations=o['imgs'],
            states=torch.from_numpy(o['obs']).to(self.device),
            traj2ds=None,
            instructions=o['text_instructions'],
            num_inference_steps=8,
            prev_actions=torch.from_numpy(A_prev[np.newaxis, :, :]).to(self.device), # (H, D) -> (1, H, D)
            inference_delay=d,
            pooled_projections=o.get('pooled_projections'),
        )
        if self.trained_rtc:
            # Train-time RTC: the first d actions are hard-frozen to the previous
            # chunk, exactly as the checkpoint was trained. No execution horizon or
            # guidance knobs here -- continuity comes from the weights.
            A_new = self.policy.predict_action_with_training_rtc_flow(
                        **common,
                        max_delay=self.max_delay,
                    )
        else:
            # Test-time RTC: no RTC training in the checkpoint, so continuity has to
            # be imposed at inference by soft-mask guidance towards the previous chunk.
            A_new = self.policy.predict_action_with_rtc_flow(
                        **common,
                        execution_horizon=s,
                        guidance_alpha=0.9,
                    )
        return A_new[0].float().detach().cpu().numpy() # (1, H, D) -> (H, D)
    
    def _predict_action(self, o):
        normalized_actions = self.policy.predict_action(
                    observations=o['imgs'],
                    states=torch.from_numpy(o['obs']).to(self.device),
                    traj2ds=None,
                    instructions=o['text_instructions'],
                    num_inference_steps = 8,
                    pooled_projections=o.get('pooled_projections'),
                )[0].float().detach().cpu().numpy() # (1, H, D) -> (H, D)
        
        return normalized_actions

class Server:

    def __init__(
        self, 
        policy:str, 
        run_dir: Path, 
        ckpt_step: int | str  = "latest", 
        device: str = "cuda:0", 
        enable_rtc: bool = False,
        action_exec_horizon: int | None = None,
        dashboard: bool = False,
    ):
        # Set up the dashboard BEFORE any logging: it only engages on a TTY, and when
        # it does the log handlers are swapped for a file one first, so the model-load
        # and config messages below are not lost under the alt-screen (serve_psix.py
        # does the same). _DASH_ACTIVE also silences dprint/color_print module-wide.
        global _DASH_ACTIVE
        self.dashboard = dashboard
        self._dash_active = dashboard and sys.stdout.isatty()
        _DASH_ACTIVE = self._dash_active
        self._log_path = (redirect_logs_to_file(prefix="serve_psi0_sonic")
                          if self._dash_active else None)
        if self._log_path is not None:
            print(f"Dashboard active — routing logs to {self._log_path}")

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please check your CUDA installation.")
        
        self.device = torch.device(device)
        overwatch.info(f"Using device: {self.device}")
        overwatch.info(f"Serving {policy}")

        # Kept so /info can describe what is being served without re-reading run_config.
        self.policy = policy
        self.run_dir = run_dir
        self.ckpt_step = ckpt_step
        self.enable_rtc = enable_rtc

        assert osp.exists(run_dir), f"run_dir {run_dir} does not exist!"
        assert osp.exists(run_dir / "checkpoints" / f"ckpt_{ckpt_step}"), f"ckpt {ckpt_step} does not exist!"
        assert osp.exists(run_dir / "run_config.json"), f"run config does not exist!"
        
        # first build dynamic config 
        config_: LaunchConfig = parse_args_to_tyro_config(run_dir / "argv.txt") # type: ignore
        # then load it from previsously saved json
        conf = apply_legacy_model_config_defaults(
            json.loads((run_dir / "run_config.json").read_text()))
        launch_config = config_.model_validate(conf)
        seed_everything(launch_config.seed or 42)

        # Opt-in (PSI_RTC_INIT_PREV=1): let the client seed the FIRST chunk with a
        # pseudo prev-action encoding the robot's current pose
        self.use_init_prev = os.environ.get("PSI_RTC_INIT_PREV", "0").strip().lower() in (
            "1", "true", "yes", "on")
        # Sticky across frames: the client sends it exactly once, on its first frame,
        # while the controller is only built after warm-up several frames later.
        self._pending_init_prev: np.ndarray | None = None
        overwatch.info(f"[init-prev] client-seeded first chunk: {'ENABLED' if self.use_init_prev else 'disabled'}")

        self.trained_rtc = bool(getattr(launch_config.model, "rtc", False))
        self.rtc_max_delay = int(getattr(launch_config.model, "max_delay", 8))
        if enable_rtc:
            overwatch.info(
                f"RTC replan path: {'train-time (frozen prefix)' if self.trained_rtc else 'test-time (guidance)'}"
                f"{f', max_delay={self.rtc_max_delay}' if self.trained_rtc else ''}")

        overwatch.info("loading action model...")
        from psi.models.psi0 import Psi0Model 
        self.model = Psi0Model.from_pretrained(run_dir, ckpt_step, launch_config, device=device)
        self.model.to(device)
        self.model.eval()

        from psi.config.transform import SimpleRepackTransform, Psi0ModelTransform, ActionStateTransform
        self.maxmin:ActionStateTransform = launch_config.data.transform.field # type:ignore
        self.model_transform:Psi0ModelTransform = launch_config.data.transform.model # type:ignore

        self.Da = launch_config.model.action_dim # type:ignore
        self.Tp = launch_config.model.action_chunk_size # type:ignore
        self.Ta = action_exec_horizon or launch_config.model.action_exec_horizon # type:ignore
        assert self.Ta <= self.Tp, "action_exec_horizon is too big"
        self.launch_cfg = launch_config
        self.count = 0

        # combined_temb models need a pooled_projections vector per instruction. Load the
        # cache the trainer precomputed (relative path saved into run_dir/project_dir), and
        # fall back to computing unseen instructions on the fly (see PooledTextEncoderCache).
        mc: Psi0ModelConfig = launch_config.model  # type: ignore
        self.pooled_helper: Optional[PooledTextEncoderCache] = None
        if mc.pooled_text_encoder is not None:
            cache_rel = getattr(mc, "pooled_cache_path", None)
            cache_file = None
            if cache_rel is not None:
                cache_file = cache_rel if osp.isabs(cache_rel) else str(run_dir / cache_rel)
            self.pooled_helper = PooledTextEncoderCache(
                cache_path=cache_file,
                encoder=mc.pooled_text_encoder,
                encoder_path=mc.pooled_text_encoder_path,
                projection_dim=mc.pooled_projection_dim,
                device=self.device,
            )

        # control - shared state with locks
        self.latest_obs = None
        self.latest_action = None
        self.action_version = 0  # Used by client to check if there's a new action
        
        self.obs_lock = threading.Lock()
        self.action_lock = threading.Lock()

        self.controller = None
        self._control_loop_started = False
        self._control_thread: threading.Thread | None = None
        self._control_stop = threading.Event()  # set on disconnect to break the control loop

        # number the connection for reset
        self._conn_generation = 0
        self._conn_setup_lock: asyncio.Lock | None = None  # created in async context

        # WebSocket: asyncio event to notify when new action is ready
        self.app = FastAPI()
        self._setup_routes()
        
        self._action_ready_event: asyncio.Event = None  # Will be created in async context
        self._active_websocket: WebSocket = None
        self._loop = None  # asyncio event loop reference for thread-safe notification
        self.start_time = time.time()
        self.start_time_obs = time.time()
        # EMA of the incoming obs rate, for the dashboard's "obs in" row.
        self._obs_hz_ema: float | None = None

    def _init_controller(self, o_first):
        controller = RealTimeChunkController(
            policy=self.model, o_first=o_first,
            trained_rtc=self.trained_rtc, max_delay=self.rtc_max_delay,
            init_prev_action=self._pending_init_prev,
        )
        return controller

    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        """Raw action -> [-1, 1], the exact inverse of _postprocess_action.

        ActionStateTransform.normalize() cannot be used here: it routes through
        __call__, which unconditionally normalizes data["states"] when the checkpoint
        normalizes states -- and this payload has no states in it.
        """
        lo = np.asarray(self.maxmin.action_min, dtype=np.float32)
        hi = np.asarray(self.maxmin.action_max, dtype=np.float32)
        # Same near-degenerate-range guard as the training transform: a dim whose
        # min == max is passed through untouched instead of dividing by ~0.
        ill = np.abs(hi - lo) < 1e-4 * (np.abs(hi) + np.abs(lo) + 1e-8)
        hi = np.where(ill, 1.0, hi)
        normalized = np.where(ill, action, (action - lo) / (hi - lo) * 2 - 1)
        # The sonic transform has no per-dim mask at all, so read it defensively: an
        # absent mask means "normalize every dim", which is what its __call__ does.
        _masks = getattr(self.maxmin, "action_norm_masks", None)
        if getattr(self.maxmin, "use_norm_mask", False) and _masks is not None:
            normalized = np.where(np.asarray(_masks), normalized, action)
        return np.clip(normalized, -1, 1).astype(np.float32)

    def _postprocess_action(self, action):
        # return self.launch_cfg.data.data_transforms.denormalize_action(action)
        return self.maxmin.denormalize(action) # denormalization is done in the pipeline

    def preprocess_image(self, image_dict: Dict[str, Any]) -> Dict[str, Any]:
        imgs = {}
        for k in image_dict.keys():
            imgs[k] = self._process_img(image_dict[k])
        return imgs

    def _process_img(self, img):
        from torchvision.transforms import v2
        transforms = [self.model_transform.resize(), self.model_transform.center_crop()]
        t = v2.Compose(transforms)
        return [t(img)]

    def _parse_obs_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Parse observation payload and return processed obs dict"""
        request = RequestMessage.deserialize(payload)
        image_dict, instruction, history_dict, state_dict, gt_action, dataset_name = \
                    request.image, request.instruction, request.history, request.state, request.gt_action, request.dataset_name
        
        instruction = instruction.lower()
        update_dashboard(instruction=instruction)

        imgs = {}

        for cam_idx, img_key in enumerate(self.launch_cfg.data.transform.repack.image_keys):
            imgs[f"cam{cam_idx}"] = Image.fromarray(np.clip(image_dict[img_key], 0, 255).astype(np.uint8))

        # Image-feed liveness for the dashboard (which camera, and when it arrived).
        if self.dashboard and image_dict:
            update_dashboard(frame_cam=next(iter(image_dict)), frame_t=time.time())
        
        # First frame only: the client may ship a RAW pseudo prev-action (its current
        # pose in action space, action-dim ordered). Normalize it with the action stats
        # -- the model's RTC path consumes normalized actions -- and pad to the model's
        # action_dim, mirroring serve_psix.py's handling.
        init_prev_action = None
        _raw_init = state_dict.get("init_prev_action")
        if _raw_init is not None and self.use_init_prev:
            _a = np.asarray(_raw_init, dtype=np.float32).reshape(-1)
            if self.maxmin.pad_action_dim is not None and self.maxmin.pad_action_dim != len(_a):
                _a = pad_to_len(_a, self.maxmin.pad_action_dim, dim=0)[0]
            assert self.maxmin.action_min is not None
            assert len(_a) == len(self.maxmin.action_min), (
                f"init_prev_action has {len(_a)} dims but the action stats have "
                f"{len(self.maxmin.action_min)}; the client must send a RAW action-space "
                f"vector in the checkpoint's action layout")
            init_prev_action = self._normalize_action(_a)
            if init_prev_action.shape[-1] < self.Da:
                init_prev_action = pad_to_len(init_prev_action, self.Da, dim=0)[0]
            self._pending_init_prev = init_prev_action
            overwatch.info(f"[init-prev] got first-frame pseudo prev-action, shape={init_prev_action.shape}")
        elif _raw_init is not None:
            log_once = getattr(self, "_init_prev_ignored_logged", False)
            if not log_once:
                self._init_prev_ignored_logged = True
                overwatch.warning("[init-prev] client sent init_prev_action but "
                                  "PSI_RTC_INIT_PREV is not set; ignoring it")

        states = state_dict["states"].copy() # shape (43,)
        obs = states

        # normalize states
        assert self.maxmin.normalize_state, "check if state is normalized"
        if self.maxmin.pad_state_dim is not None and self.maxmin.pad_state_dim != len(obs):
            obs = pad_to_len(obs, self.maxmin.pad_state_dim, dim=0)[0]
        obs = self.maxmin.normalize_state_func(obs) # shape (43,)
        obs = obs[np.newaxis, np.newaxis, :] # (43,) -> (1, 1, 43)

        image_input = self.preprocess_image(imgs)
        batch_images = [image_input['cam0']] # batch size == 1

        conditions = {}
        text_instructions = [instruction] # len == 1
        # pooled_projections (1, D) for combined_temb; None when the model has no pooled
        # text encoder configured. Cache hit is a dict lookup; miss computes via CLIP.
        pooled_projections = self.pooled_helper(instruction) if self.pooled_helper is not None else None
        return {'imgs': batch_images, 'text_instructions': text_instructions, 'obs': obs,
                'conditions': conditions, 'pooled_projections': pooled_projections, 'init_prev_action': init_prev_action}

    async def websocket_handler(self, websocket: WebSocket):
        """
        WebSocket handler for bidirectional communication:
        - Receive obs from client at high frequency (only the newest frame is kept)
        - Send action to client immediately when new action is ready
        """
        await websocket.accept()

        # Claim the server for this connection: bump the generation and tear down
        # whatever the previous connection left behind BEFORE any obs is accepted.
        gen = await self._begin_connection(websocket)
        action_ready = self._action_ready_event
        update_dashboard(connected=True, error=None, status="waiting for warmup",
                         conn_id=gen)
        dprint(f"[WebSocket] Client connected (generation {gen})")

        # Newest raw frame received but not yet parsed. Parsing costs real time (JSON +
        # PIL + resize), so the reader never parses: it overwrites this slot and lets the
        # parser skip whatever went stale meanwhile.
        newest: List[str | None] = [None]
        newest_ready = asyncio.Event()
        disconnected = asyncio.Event()

        async def receive_obs():
            """Reader: drain the socket as fast as it can, keep only the newest frame."""
            try:
                while True:
                    newest[0] = await websocket.receive_text()
                    newest_ready.set()
            except WebSocketDisconnect:
                dprint("[WebSocket] Client disconnected (receive)")
            except Exception as e:
                import traceback
                overwatch.error(f"[WebSocket] Receive error: {e}\n{traceback.format_exc()}")
                update_dashboard(status="error", error=f"{type(e).__name__}: {e}")
            finally:
                disconnected.set()
                newest_ready.set()  # unpark the parser

        async def parse_obs():
            """Parser: only ever parses the most recent frame, so a backlog that built up
            during a slow stretch (controller warmup, a long forward pass) is dropped
            instead of being replayed into the policy as seconds-old state."""
            try:
                while not disconnected.is_set():
                    await newest_ready.wait()
                    newest_ready.clear()
                    data, newest[0] = newest[0], None
                    if data is None:
                        continue

                    interval = time.time() - self.start_time_obs
                    self.start_time_obs = time.time()
                    if interval > 0:
                        inst_hz = 1.0 / interval
                        self._obs_hz_ema = inst_hz if self._obs_hz_ema is None \
                            else 0.9 * self._obs_hz_ema + 0.1 * inst_hz
                        update_dashboard(obs_hz=self._obs_hz_ema,
                                         obs_hz_target=1.0 / CTRL_PERIOD_SEC)
                    # print(f"[WebSocket] receive_obs interval: {interval} seconds")

                    payload = json.loads(data)
                    # Off the event loop, so parsing never stalls the sender.
                    this_o = await asyncio.to_thread(self._parse_obs_payload, payload)
                    if gen != self._conn_generation:  # replaced while we parsed
                        return
                    with self.obs_lock:
                        self.latest_obs = this_o

                    # If control loop hasn't started, start it automatically.
                    if not self._control_loop_started:
                        self._control_loop_started = True
                        asyncio.create_task(self._start_control_loop(gen))
            except Exception as e:
                import traceback
                overwatch.error(f"[WebSocket] Parse error: {e}\n{traceback.format_exc()}")
                update_dashboard(status="error", error=f"{type(e).__name__}: {e}")

        async def send_action():
            """Send action to client when new action is ready"""
            try:
                while not disconnected.is_set():
                    # Wait for new action to be ready
                    await action_ready.wait()
                    action_ready.clear()

                    interval = time.time() - self.start_time
                    self.start_time = time.time()
                    # print(f"[WebSocket] send_action interval: {interval} seconds")

                    # Get the action
                    with self.action_lock:
                        action = self.latest_action
                        version = self.action_version
                        self.latest_action = None  # Reset after sending

                    if action is None:
                        # Teardown wiped it between notify and wake -- not fatal.
                        continue

                    # Send action to client
                    response = ResponseMessage(action, err=0.0)
                    resp_dict = response.serialize()
                    resp_dict["version"] = version
                    await websocket.send_text(json.dumps(resp_dict))
                    # print(f"[WebSocket] Sent action, version={version}")
            except WebSocketDisconnect:
                dprint("[WebSocket] Client disconnected (send)")
            except Exception as e:
                dprint(f"[WebSocket] Send error: {e}")

        tasks = [asyncio.create_task(c) for c in (receive_obs(), parse_obs(), send_action())]
        try:
            # First task to finish ends the connection. The reader returning on
            # disconnect is enough, so teardown no longer depends on a send failing --
            # a client that drops before any action exists used to park the sender
            # forever here, and the handler never reached its teardown at all.
            _, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for t in pending:
                t.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        except Exception as e:
            dprint(f"[WebSocket] Connection closed: {e}")
            update_dashboard(connected="disconnected")
        finally:
            await self._end_connection(gen)
            update_dashboard(connected=False, status="idle")
            dprint("[WebSocket] Handler finished")

    async def _begin_connection(self, websocket: WebSocket) -> int:
        """Take ownership of the server for a newly accepted connection: retire the
        previous generation (tearing down its controller off the event loop) and reset
        every piece of per-connection state, so this client always starts from scratch."""
        if self._conn_setup_lock is None:
            self._conn_setup_lock = asyncio.Lock()
        async with self._conn_setup_lock:
            self._conn_generation += 1
            gen = self._conn_generation
            # Blocking joins (up to ~7s) go to a worker thread: doing them inline used
            # to freeze the event loop for the whole teardown.
            await asyncio.to_thread(self._reset_connection_state)
            self._active_websocket = websocket
            self._action_ready_event = asyncio.Event()
            self.start_time = time.time()
            self.start_time_obs = time.time()
            return gen

    async def _end_connection(self, gen: int) -> None:
        """Tear down this connection's state -- but only if it is still the live one.
        A handler that lost a reconnect race must not touch the controller that the
        newer connection is already driving."""
        if self._conn_setup_lock is None:
            self._conn_setup_lock = asyncio.Lock()
        async with self._conn_setup_lock:
            if gen != self._conn_generation:
                dprint(f"[WebSocket] generation {gen} superseded by {self._conn_generation}; "
                      f"leaving live state alone")
                return
            await asyncio.to_thread(self._reset_connection_state)
            self._active_websocket = None

    async def _start_control_loop(self, gen: int):
        """Build the controller for generation `gen` and start its control thread.

        The caller sets `_control_loop_started` before spawning this, so obs keep being
        consumed while the warmup runs -- the warmup is several forward passes and used
        to run inline on the event loop, which froze the reader and left the policy
        replaying a seconds-long backlog of stale frames once it resumed.
        """
        update_dashboard(status="warming up")
        try:
            with self.obs_lock:
                o_first = copy.deepcopy(self.latest_obs)

            controller = await asyncio.to_thread(
                self._init_controller, o_first
            )  # wait for model warm up, off the event loop
        except BaseException as e:
            # Let a later obs retry instead of wedging the server with the flag stuck on
            # and no controller behind it.
            self._control_loop_started = False
            update_dashboard(status="error", error=f"controller init failed: {e}")
            dprint(f"[control loop] controller init failed: {e}")
            raise

        if gen != self._conn_generation:
            # Client went away (or was replaced) during warmup: drop what we just built
            # rather than driving a dead connection.
            await asyncio.to_thread(controller.stop)
            self._control_loop_started = False
            dprint(f"[control loop] generation {gen} superseded during warmup; discarded")
            return

        self.controller = controller

        # Start control loop thread (clear any stop signal left from a prior connection)
        self._control_stop.clear()
        self._control_thread = threading.Thread(
            target=self._control_loop, args=(gen,), daemon=True
        )
        self._control_thread.start()
        dprint("[control loop] started")

    def _reset_connection_state(self) -> None:
        """Tear down the control loop + inference thread and wipe ALL per-connection
        state, so the next connection starts from a clean slate (a fresh, re-warmed
        controller with no carry-over of the previous episode's action chunk).

        Blocking (thread joins); callers run it via asyncio.to_thread.
        """
        # Wipe obs first: the control loop's `obs_next is None` check is its intended
        # exit, so it stops even if it misses the stop event.
        with self.obs_lock:
            self.latest_obs = None
        self._pending_init_prev = None  # next client re-sends it on its own first frame
        with self.action_lock:
            self.latest_action = None
            self.action_version = 0

        if self._control_loop_started:
            self._control_stop.set()
            if self._control_thread is not None:
                self._control_thread.join(timeout=2.0)
                if self._control_thread.is_alive():
                    color_print("[control loop] did not stop within 2s", style="yellow")
                self._control_thread = None
            if self.controller is not None:
                self.controller.stop()
                self.controller = None
            self._control_loop_started = False

        reset_dashboard()
        dprint("[WebSocket] control + inference stopped; per-connection state reset")

    def _control_loop(self, gen: int):
        """
        Control loop: Execute controller.step strictly every CTRL_PERIOD_SEC
        And expect at next time, the obs_next sent from client is the one after executing the action

        Owned by connection generation `gen`; it must never drive, or post actions to,
        a connection that has since replaced it.
        """
        next_tick = time.perf_counter()
        prev_tick = time.perf_counter()
        missed_ticks = 0
        update_dashboard(ctrl_target_ms=CTRL_PERIOD_SEC * 1000)

        while not self._control_stop.is_set() and gen == self._conn_generation:
            # loop_start = time.time()

            # 1. Get latest obs
            with self.obs_lock:
                obs_next = copy.deepcopy(self.latest_obs)
            if obs_next is None:  # teardown in progress (latest_obs wiped)
                break

            # 2. Execute step
            controller = self.controller
            if controller is None:  # teardown won the race
                break
            action = controller.step(obs_next) # (1, D)
            pred_action = self._postprocess_action(action) # (1, D)

            # 3. Update latest_action
            with self.action_lock:
                self.latest_action = pred_action
                self.action_version += 1
                update_dashboard(action_version=self.action_version)

            # 4. Notify WebSocket that new action is ready
            if self._action_ready_event is not None and gen == self._conn_generation:
                # Thread-safe way to set asyncio event from another thread
                try:
                    assert self._loop is not None, "asyncio event loop is not set"
                    self._loop.call_soon_threadsafe(self._action_ready_event.set)
                except Exception as e:
                    dprint(f"[control loop] Failed to notify WebSocket: {e}")
            
            # elapsed = (time.time() - loop_start) * 1000
            # print(f"[control loop] step took {elapsed:.1f}ms, version={self.action_version}")
            
            # 5. Wait until next ctrl period
            next_tick += CTRL_PERIOD_SEC
            sleep_time = next_tick - time.perf_counter()
            now = time.perf_counter()
            interval = now - prev_tick
            prev_tick = now
            update_dashboard(ctrl_ms=interval * 1000)
            # print(f"[control loop] interval: {interval} seconds")
            if sleep_time > 0:
                time.sleep(sleep_time)
                # delay_ms(sleep_time*1000)
            else:
                missed_ticks += 1
                update_dashboard(ctrl_missed=missed_ticks)
                dprint(f"[control loop] WARNING: missed tick by {-sleep_time*1000:.1f}ms")
                next_tick = time.perf_counter()
        dprint("[control loop] stopped (client disconnected)")


    def _setup_routes(self):
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            self._loop = asyncio.get_event_loop()
            await self.websocket_handler(websocket)
        
        @self.app.get("/info")
        async def info():
            return self.info()
        
        @self.app.get("/health")
        async def health_check():
            return {"status": "ok"}
        
    def info(self) -> JSONResponse:
        """Describe the served policy and the request format the /ws endpoint expects."""
        # timestamp == last segment of the run-dir name, split by "."
        timestamp = Path(self.run_dir).name.split(".")[-1]

        # image transforms applied to every observation before inference
        # (mirrors preprocess_image/_process_img: [resize, center_crop])
        transforms = [
            {"name": "resize", "size": self.model_transform.resize.size},
            {"name": "center_crop", "size": self.model_transform.center_crop.size},
        ]

        repack = self.launch_cfg.data.transform.repack
        image_keys = list(getattr(repack, "image_keys", []))
        state_key = getattr(repack, "state_key", "states")

        # Model-side (padded) state width. The wire width is whatever the robot sends
        # before pad_to_len, which the server only learns per request.
        state_dim = self.maxmin.pad_state_dim
        if state_dim is None and self.maxmin.state_min is not None:
            state_dim = len(self.maxmin.state_min)

        content = {
            "policy": self.policy,
            "timestamp": timestamp,
            "run_dir": str(self.run_dir),
            "ckpt_step": self.ckpt_step,
            "dataset_name": repack.dataset_name,
            "transforms": transforms,
            "expected_keys": {
                "image": {k: "HxWx3 uint8 image array" for k in image_keys},
                "state": {state_key: f"1x{state_dim} unnormalized state vector"},
            },
            "observation": {
                "state_dim": state_dim,
                "normalize_state": self.maxmin.normalize_state,
            },
            "action": {
                "action_dim": self.Da,
                "action_chunk_size": self.Tp,
                "action_exec_horizon": self.Ta,
            },
            "rtc_enabled": self.enable_rtc,
            "rtc": {
                "prediction_horizon": PREDICT_HORIZON,
                "min_exec_horizon": MIN_EXEC_HORIZON,
                "delay_buffer_size": DELAY_BUFFER_SIZE,
                "d_init": D_INIT,
                "ctrl_period_sec": CTRL_PERIOD_SEC,
                "mode": "train" if self.trained_rtc else "test_time",
                "init_prev_enabled": self.use_init_prev,
                "max_delay": self.rtc_max_delay,
            },
        }
        return JSONResponse(content=content)

    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        dprint(f"Server listens on {host}:{port}")
        dprint(f"WebSocket endpoint: ws://{host}:{port}/ws")

        # Logs were already redirected to a file in __init__ (so startup messages aren't lost). 
        dash_active = self._dash_active
        stop_event = threading.Event()
        dash_thread = None
        uvicorn_log_level = "info"
        if self.dashboard and not dash_active:
            overwatch.warning("dashboard requested but stdout is not a TTY — falling back to line logs")
        if dash_active:
            update_dashboard(obs_hz_target=1.0 / CTRL_PERIOD_SEC,
                             ctrl_target_ms=CTRL_PERIOD_SEC * 1000,
                             device=str(self.device), status="waiting for first obs")
            uvicorn_log_level = "warning"
            dash_thread = threading.Thread(
                target=dashboard_loop, args=(stop_event, time.monotonic()),
                kwargs={"log_path": self._log_path,
                        "title": f"[bold]Psi0 Sonic RTC Server[/bold] [dim]{host}:{port}[/dim]"},
                daemon=True,
            )
            dash_thread.start()

        try:
            uvicorn.run(self.app, host=host, port=port, log_level=uvicorn_log_level,
                        access_log=not dash_active)
        except Exception as e:
            dprint(f"Server crashed, {e}")
        finally:
            stop_event.set()
            if dash_thread is not None:
                dash_thread.join(timeout=1.0)
            dprint("Server stopped.")
            exit(1)

def serve(cfg: ServerConfig) -> None:
    overwatch.info("Server :: Initializing Policy")
    assert cfg.policy is not None, "which policy to serve?"
    assert cfg.rtc, "this server is for rtc"
    assert type(cfg.ckpt_step) == int, "ckpt_step must be specified"
    server = Server(
        cfg.policy, 
        Path(cfg.run_dir), 
        cfg.ckpt_step, 
        cfg.device,
        cfg.rtc,
        cfg.action_exec_horizon,
        dashboard=cfg.dashboard)
    
    dprint("Server :: Spinning Up")
    server.run(cfg.host, cfg.port)

def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()  # take environment variables from .env file
    config = tyro.cli(ServerConfig, config=(tyro.conf.ConsolidateSubcommandArgs,))
    serve(config)

if __name__ == "__main__":
    main()