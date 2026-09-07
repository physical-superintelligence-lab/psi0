import os
import sys
import threading
import tyro
import torch
import time
import numpy as np
import os.path as osp
from pathlib import Path
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from PIL import Image
from typing import Union, Dict, Any, List
from base64 import b64decode, b64encode
from fastapi.responses import JSONResponse
from numpy.lib.format import descr_to_dtype, dtype_to_descr
from torchvision.transforms import v2

from psi.deploy.helpers import *
from psi.config.config import LaunchConfig, ServerConfig
from psi.config.transform import SimpleRepackTransform, Psi0ModelTransform, ActionStateTransform
import json
from psi.utils import parse_args_to_tyro_config, pad_to_len, seed_everything, apply_legacy_model_config_defaults
from psi.utils.overwatch import initialize_overwatch 

overwatch = initialize_overwatch(__name__)

class Server:
    
    def __init__(
        self, 
        policy:str, 
        run_dir: Path, 
        ckpt_step: int | str  = "latest", 
        device: str = "cuda:0", 
        enable_rtc: bool = False,
        action_exec_horizon: int | None = None,
        rtc_mode: str = "auto",
        pig_mask_schedule: str = "exponential",
        pig_guidance_alpha: float = 0.9,
        num_inference_steps: int = 10,
        rtc_session_timeout: float = 30.0,
        rtc_inference_delay: int | None = None,
        min_exec_horizon: int | None = None,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please check your CUDA installation.")
         
        self.device = torch.device(device)
        overwatch.info(f"Using device: {self.device}")
        overwatch.info(f"Serving {policy}")

        self.policy = policy
        self.run_dir = run_dir
        self.ckpt_step = ckpt_step

        assert osp.exists(run_dir), f"run_dir {run_dir} does not exist!"
        assert osp.exists(run_dir / "checkpoints" / f"ckpt_{ckpt_step}"), f"ckpt {ckpt_step} does not exist!"
        assert osp.exists(run_dir / "run_config.json"), f"run config does not exist!"

        # load launch config 
        config_: LaunchConfig = parse_args_to_tyro_config(run_dir / "argv.txt") # type: ignore
        conf = apply_legacy_model_config_defaults(
            json.loads((run_dir / "run_config.json").read_text()))
        launch_config = config_.model_validate(conf)
        seed_everything(launch_config.seed or 42)

        from psi.models.psi0 import Psi0Model 
        self.model = Psi0Model.from_pretrained(run_dir, ckpt_step, launch_config, device=device)
        self.model.to(device)
        self.model.eval()

        self.maxmin:ActionStateTransform = launch_config.data.transform.field # type:ignore
        self.repack_transform:SimpleRepackTransform = launch_config.data.transform.repack # type:ignore
        self.model_transform:Psi0ModelTransform = launch_config.data.transform.model # type:ignore

        """ compatible for the psix dataloader & transforms """
        if hasattr(self.maxmin, "state_transform"):
            self.state_transform = self.maxmin.state_transform
            self.action_transform = self.maxmin.action_transform
            embodiment_tag = "simple_g1"
            _norm_path = run_dir / f"{embodiment_tag}_norm_stats.json"
            assert osp.exists(_norm_path), f"{_norm_path} not found (training writes it next to the ckpt)"
            _stats_for_tag = json.loads(Path(_norm_path).read_text())
            self.state_transform.init_from_norm_stats(_stats_for_tag)
            self.action_transform.init_from_norm_stats(_stats_for_tag)
            overwatch.info(f"Initialized state/action normalizers from {_norm_path}")

        # The sonic checkpoints (SonicRepackTransform / SonicActionStateTransform) keep
        # pad_state_dim on the field transform too; either one is authoritative, so take
        # whichever the config actually carries.
        self.pad_state_dim = (getattr(self.repack_transform, "pad_state_dim", None)
                              or getattr(self.maxmin, "pad_state_dim", None))
        # Image keys as the checkpoint was trained with them. A client that sends the
        # same keys gets its views ordered deterministically; one that sends different
        # keys (the sonic robot client posts `observation.images.head` while the pack
        # recorded `observation.images.egocentric`) falls back to payload order.
        self.image_keys: List[str] = list(getattr(self.repack_transform, "image_keys", []) or [])
        self._img_key_warned = False
        self.To = int(getattr(launch_config.model, "observation_horizon", 1) or 1)  # type:ignore

        # Print number of total/trainable model parameters
        num_params = sum(p.numel() for p in self.model.parameters())
        overwatch.info(f"Parameters (in millions): {num_params*1e-6:.3f} Total", ctx_level=1)

        # self.previous_rpy = np.array([0.0, 0.0, 0.0], dtype=np.float32) # FIXME 
        # self.previous_height = np.array([0.74], dtype=np.float32)

        self.Da = launch_config.model.action_dim # type:ignore
        self.Tp = launch_config.model.action_chunk_size # type:ignore
        self.Ta = action_exec_horizon or launch_config.model.action_exec_horizon # type:ignore
        assert self.Ta <= self.Tp, "action_exec_horizon is too big"
        self.launch_config = launch_config
        self.count = 0
        
        self.num_inference_steps = int(num_inference_steps)
        self.pig_mask_schedule = pig_mask_schedule
        self.pig_guidance_alpha = float(pig_guidance_alpha)

        # RTC flavour. "train" is the hard-frozen-prefix sampler and needs a checkpoint
        # trained with --model.rtc; "test_time" is inference-time guidance towards the
        # previous chunk and works on any flow checkpoint (that is the only option for a
        # --model.no-rtc run, e.g. the sonic-wbcbox neckless finetunes). "auto" picks
        # whichever the checkpoint supports.
        trained_rtc = bool(getattr(launch_config.model, "rtc", False))  # type:ignore
        if rtc_mode not in ("auto", "off", "train", "test_time"):
            raise ValueError(f"rtc_mode must be auto|off|train|test_time, got {rtc_mode!r}")
        # An explicit --rtc-mode train|test_time turns RTC on by itself: asking for a
        # flavour and silently getting open loop because --rtc was not also passed is a
        # trap worth closing. "auto" is the default value, so it cannot be told apart
        # from "not passed" and still requires --rtc.
        if rtc_mode == "off":
            self.rtc_mode = "off"
        elif rtc_mode in ("train", "test_time"):
            self.rtc_mode = rtc_mode
            if not enable_rtc:
                overwatch.info(f"--rtc-mode {rtc_mode} implies --rtc; enabling RTC")
        elif not enable_rtc:
            self.rtc_mode = "off"
        else:  # auto + --rtc
            self.rtc_mode = "train" if trained_rtc else "test_time"
        self.enable_rtc = self.rtc_mode != "off"
        if self.rtc_mode == "train" and not trained_rtc:
            raise ValueError(
                "rtc_mode=train needs a checkpoint trained with --model.rtc; this run has "
                "model.rtc=False. Serve with rtc_mode=test_time (or rtc_mode=auto) or --no-rtc.")
        self.rtc_max_delay = int(getattr(launch_config.model, "max_delay", 8) or 8)  # type:ignore
        if self.enable_rtc:
            # Both RTC paths condition on the previous chunk shifted left by exec_shift
            # (= Ta) and zero padded back to Tp. At Ta == Tp the shift consumes the whole
            # chunk, leaving the "previous chunk" all padding with nothing for the
            # guidance/freeze to hold on to -- silently open loop. Refuse instead.
            assert self.Ta < self.Tp, (
                f"RTC needs action_exec_horizon < action_chunk_size, got Ta={self.Ta}, "
                f"Tp={self.Tp}: the previous chunk shifted by Ta would be entirely padding. "
                f"Pass --action-exec-horizon (e.g. {self.Tp // 2}) or serve with --no-rtc.")
            self.previous_action = None #np.zeros((self.Tp, self.Da), dtype=np.float32)
            # Rows the client consumed before this request: /act returns Ta and the
            # client runs them all. A protocol fact, not a knob -- s and d are the knobs.
            self.exec_shift = self.Ta
            self.min_exec_horizon = (self.exec_shift if min_exec_horizon is None
                                     else int(min_exec_horizon))
            if rtc_inference_delay is not None:
                self.rtc_inference_delay = int(rtc_inference_delay)
            elif self.rtc_mode == "train":
                # Half the executed window: the first d returned rows continue the
                # previous plan, the rest are fresh. Bounded by d + s <= H and by the
                # prefix width the checkpoint was trained on.
                self.rtc_inference_delay = max(1, min(self.exec_shift // 2,
                                                      self.Tp - self.min_exec_horizon,
                                                      self.rtc_max_delay - 1))
                overwatch.info(
                    f"rtc_mode=train without --rtc-inference-delay: defaulting to d="
                    f"{self.rtc_inference_delay}. The frozen prefix is this sampler's "
                    f"only conditioning channel, so d=0 would ignore the previous chunk "
                    f"entirely and smooth nothing.")
            else:
                self.rtc_inference_delay = 0
            _d, _s, _H = self.rtc_inference_delay, self.min_exec_horizon, self.Tp
            if _s < self.exec_shift:
                raise ValueError(
                    f"RTC needs s >= exec_shift, got s={_s} (--min-exec-horizon) and "
                    f"exec_shift={self.exec_shift} (= action_exec_horizon, the rows the "
                    f"client consumes). The previous chunk is zero padded in its last "
                    f"{self.exec_shift} rows, so a free region starting at H-s={_H - _s} "
                    f"would run the ramp over {self.exec_shift - _s} padding row(s) and "
                    f"guide those actions towards zero.")
            if not 0 <= _d or _d + _s > _H:
                raise ValueError(
                    f"RTC needs 0 <= d and d + s <= H, got d={_d} "
                    f"(--rtc-inference-delay), s={_s} (--min-exec-horizon), H={_H} "
                    f"(action_chunk_size). With s={_s} the delay must satisfy "
                    f"d <= {_H - _s}; beyond that the hard region reaches into the zero "
                    f"padding, which holds no previous action to reproduce.")
            if _d >= self.exec_shift:
                overwatch.warning(
                    f"inference_delay d={_d} >= exec_shift={self.exec_shift}: every row "
                    f"/act returns falls inside the hard region, so the client replays "
                    f"the previous plan verbatim and never sees a reaction to the "
                    f"observation it just sent. Lower --rtc-inference-delay below "
                    f"{self.exec_shift}.")
            if self.rtc_mode == "train":
                if _d < 1:
                    raise ValueError(
                        "rtc_mode=train needs --rtc-inference-delay >= 1: the frozen "
                        "prefix is its only conditioning channel, so d=0 makes the call "
                        "identical to open-loop predict_action (prev_actions is never "
                        "read). Pass a delay, or serve with rtc_mode=test_time, whose "
                        "ramp smooths at d=0.")
                if _d >= min(_H, self.rtc_max_delay):
                    raise ValueError(
                        f"rtc_inference_delay={_d} is wider than the prefixes this "
                        f"checkpoint was trained on: d must stay under max_delay="
                        f"{self.rtc_max_delay} (and under H={_H}). Lower it to "
                        f"<= {min(_H, self.rtc_max_delay) - 1}, or serve with "
                        f"rtc_mode=test_time, which takes no max_delay at all.")
            # _create_soft_mask decays over [d, H-s) and is 0 beyond. At d == H-s that
            # interval is empty, so every decaying schedule collapses to "hard" and
            # pig_mask_schedule stops meaning anything. Say so rather than let it look
            # configured.
            if (self.rtc_mode != "train" and _d == _H - _s
                    and self.pig_mask_schedule in ("exponential", "linear")):
                overwatch.warning(
                    f"mask_schedule={self.pig_mask_schedule} has no effect: at d == H-s "
                    f"== {_d} the ramp region [d, H-s) is empty, so the mask is a hard "
                    f"freeze over all {_d} overlapping rows. Lower --rtc-inference-delay "
                    f"to restore the ramp, or pass --pig-mask-schedule hard to make the "
                    f"freeze explicit.")

            _delay_note = (f"max_delay={self.rtc_max_delay}, " if self.rtc_mode == "train"
                           else f"mask_schedule={self.pig_mask_schedule}, alpha={self.pig_guidance_alpha}, ")
            overwatch.info(f"RTC enabled: mode={self.rtc_mode}, \n"
                           f"{_delay_note}\n"
                           f"action_dim={self.Da}, \n"
                           f"action_chunk_size={self.Tp}, \n"
                           f"action_exec_horizon={self.Ta} (returned rows), \n"
                           f"exec_shift={self.exec_shift} (prev-chunk shift), \n"
                           f"inference_delay(d)={self.rtc_inference_delay}, \n"
                           f"min_exec_horizon(s)={self.min_exec_horizon}")
        else:
            self.previous_action = None
        self.last_serve_time = time.monotonic()
        self._infer_lock = threading.Lock()
        self._rtc_session: str | None = None   # id of the client that currently owns RTC state
        self._session_lock = threading.Lock()
        # seconds of silence after which another client may take the RTC session over
        self.rtc_session_timeout = float(rtc_session_timeout)


    def _claim_rtc_session(self, history_dict: Dict[str, Any], peer: str) -> None:
        """Reject a second RTC client instead of silently giving it wrong actions.
        """
        if not self.enable_rtc:
            return  # no cross-request state -> concurrent clients are merely serialized
        history_dict = history_dict if isinstance(history_dict, dict) else {}
        # `session_id` is what the SIMPLE eval client sends (simple/baselines/psi0.py);
        # `client_id` is what serve_psi0_simple_multi documents. Accept either.
        client = str(history_dict.get("client_id")
                     or history_dict.get("session_id")
                     or peer)

        with self._session_lock:
            idle = time.monotonic() - self.last_serve_time
            stale = self._rtc_session is not None and client != self._rtc_session \
                and idle > self.rtc_session_timeout
            if self._rtc_session is None or "reset" in history_dict or stale:
                if self._rtc_session not in (None, client):
                    overwatch.info(
                        f"RTC session handed over: {self._rtc_session} -> {client}"
                        + (f" (previous owner idle {idle:.1f}s)" if stale else ""))
                    self.previous_action = None
                self._rtc_session = client
                return
            if client != self._rtc_session:
                raise HTTPException(
                    status_code=409,
                    detail=(f"RTC session is owned by '{self._rtc_session}'; '{client}' cannot "
                            f"share it. This server keeps ONE global previous_action, so a second "
                            f"RTC client would silently get actions conditioned on the first "
                            f"client's chunk. Send history['reset'] to take the session over, "
                            f"wait {self.rtc_session_timeout:.0f}s for the owner to go idle, or "
                            f"use serve_psi0_simple_multi for per-client RTC state."),
                )

    def _order_images(self, image_dict: Dict[str, Any]) -> List[Image.Image]:
        """Payload images as PIL, in the view order the checkpoint was trained with."""
        keys = [k for k in self.image_keys if k in image_dict]
        if len(keys) != len(self.image_keys) or not keys:
            keys = list(image_dict.keys())[:len(self.image_keys) or None]
            if not self._img_key_warned:
                self._img_key_warned = True
                overwatch.warning(
                    f"payload image keys {list(image_dict.keys())} do not match the "
                    f"checkpoint's {self.image_keys}; falling back to payload order")
        return [Image.fromarray(np.clip(np.asarray(image_dict[k]), 0, 255).astype(np.uint8))
                for k in keys]

    def _prepare_states(self, state_dict: Dict[str, Any]) -> torch.Tensor:
        """Raw client state -> normalized (To, Ds) tensor on device.

        Accepts a flat (Ds,) vector (what the sonic robot client posts) as well as the
        (N, Ds) history stack the simple client posts; only the last To frames are kept,
        since that is all the model was trained to read.
        """
        states = np.asarray(state_dict["states"], dtype=np.float32).copy()
        if states.ndim == 1:
            states = states[np.newaxis, :]
        assert states.ndim == 2, f"states must be (Ds,) or (N, Ds), got {states.shape}"
        if states.shape[0] > self.To:
            states = states[-self.To:]
        states, _ = pad_to_len(states, self.pad_state_dim, dim=1)
        if self.maxmin.normalize_state:  # type:ignore
            states = self.maxmin.normalize_state_func(states)
        return torch.as_tensor(np.asarray(states, dtype=np.float32)).to(self.device)

    def _shifted_prev_actions(self) -> torch.Tensor:
        """Previous chunk advanced by exec_shift rows, re-padded to Tp.

        The shift is what the client executed, not the mask's min_exec_horizon. Startup
        enforces s >= exec_shift, so the tail padding always lands in the mask's free
        region (i >= H-s) and never enters the guidance/freeze term.
        """
        s = self.exec_shift
        prev = np.concatenate([
            self.previous_action[None, s:, :],
            np.zeros((1, s, self.Da), dtype=np.float32)
        ], axis=1)  # (1, Tp, Da)
        return torch.from_numpy(prev).to(self.device)

    def predict_action(self, payload: Dict[str, Any], http_request: Request) -> JSONResponse:
        # overwatch.info(f"Received request with payload: {payload}")
        # Last-resort identity, used only when the client sends no session/client id.
        # It must be the host alone: a client that does not keep the connection alive
        # (requests.post per step, as the SIMPLE eval client does) draws a fresh
        # ephemeral port every call, so host:port would read as a new client on every
        # request and 409 the whole episode away.
        peer = http_request.client.host if http_request.client else "unknown"
        try:
            request = RequestMessage.deserialize(payload)
            image_dict, instruction, history_dict, state_dict, gt_action, dataset_name = \
                request.image, request.instruction, request.history, request.state, request.gt_action, request.dataset_name
            
            # the sonic robot client posts history/condition as JSON null, not {}
            history_dict = history_dict if isinstance(history_dict, dict) else {}
            # every repack transform lowercases the instruction before tokenizing
            # (transform.py / transform_psi0_sonic.py), so serving must too
            instruction = str(instruction).lower()
            overwatch.info(f"Instruction: {instruction}")
            overwatch.info(f"history_dict: {history_dict}")
            self._claim_rtc_session(history_dict, peer)

            transforms = [self.model_transform.resize(), self.model_transform.center_crop()]
            t = v2.Compose(transforms)

            images = [[t(img) for img in self._order_images(image_dict)]]  # B=1
            states = self._prepare_states(state_dict)  # (To, Ds)

            # uvicorn runs this sync handler in a threadpool, so two clients hitting /act
            # at once would enter the model concurrently and corrupt state shared across
            # requests -- the diffusers scheduler's step index (IndexError in
            # noise_scheduler.step) and self.previous_action. Serialize inference; only
            # request decoding and image transforms run in parallel.
            with self._infer_lock:
                common: Dict[str, Any] = dict(
                    observations=images,
                    states=states.unsqueeze(0), # B, To, Ds
                    instructions=[instruction], # [Task] * B
                    num_inference_steps=self.num_inference_steps,
                    traj2ds=None,
                )
                current_time = time.monotonic()
                if (not self.enable_rtc or self.previous_action is None
                        or "reset" in history_dict):
                    #  or (current_time - self.last_serve_time) > 30  #if idle more than 60s, reset previous action
                    if self.enable_rtc:
                        overwatch.info("===Reset or first step, without condition===")
                    raw_pred_actions = self.model.predict_action(**common)
                else:
                    overwatch.info(f"RTC enabled ({self.rtc_mode}), using RTC inference")
                    overwatch.info("Last chunk execution loop time: {:.2f}s ago".format(current_time - self.last_serve_time))
                    prev_actions = self._shifted_prev_actions()  # (1, Tp, Da)

                    if self.rtc_mode == "test_time":
                        raw_pred_actions = self.model.predict_action_with_rtc_flow(
                            **common,
                            prev_actions=prev_actions,
                            inference_delay=self.rtc_inference_delay,
                            execution_horizon=self.min_exec_horizon,
                            mask_schedule=self.pig_mask_schedule,
                            guidance_alpha=self.pig_guidance_alpha,
                        )
                    else:
                        raw_pred_actions = self.model.predict_action_with_training_rtc_flow(
                            **common,
                            prev_actions=prev_actions,
                            inference_delay=self.rtc_inference_delay,
                            max_delay=self.rtc_max_delay
                        )

                raw_pred_actions = raw_pred_actions.reshape(-1, self.Da).float().detach().cpu().numpy() # (Tp, Da)
                pred_actions = self.maxmin.denormalize(raw_pred_actions) # (Tp, Da)
                self.previous_action = raw_pred_actions.copy().astype(np.float32) # for rtc
                pred_actions = pred_actions[:self.Ta] # type:ignore
                overwatch.info(f"Return Action ({pred_actions.shape})") # : {pred_actions}

                self.last_serve_time = time.monotonic()
            response = ResponseMessage(pred_actions, 0.0) # type:ignore
            return JSONResponse(content=response.serialize())

        except HTTPException as e:
            # session rejection: let FastAPI return the real status code instead of the
            # catch-all below, which would hide it behind a 200
            overwatch.warning(f"rejected {peer}: {e.detail}")
            raise

        except Exception as e:
            import traceback
            overwatch.warning(traceback.format_exc())
            return JSONResponse(content=f'{{"status": "{e}"}}')

    def info(self) -> JSONResponse:
        """Describe the served policy and the request format the /act endpoint expects."""
        # timestamp == last segment of the run-dir name, split by "."
        timestamp = Path(self.run_dir).name.split(".")[-1]

        # image transforms applied to every observation before inference
        # (mirrors predict_action: [resize, center_crop])
        transforms = [
            {"name": "resize", "size": self.model_transform.resize.size},
            {"name": "center_crop", "size": self.model_transform.center_crop.size},
        ]

        content = {
            "policy": self.policy,
            "timestamp": timestamp,
            "run_dir": str(self.run_dir),
            "ckpt_step": self.ckpt_step,
            "transforms": transforms,
            "expected_keys": {
                "image": {k: "HxWx3 uint8 image array" for k in self.image_keys},
                "state": {"states": f"(Ds,) or (N, Ds) unnormalized; padded to "
                                    f"{self.pad_state_dim}, last {self.To} frame(s) used"},
                "history": {"reset": "optional"},
                "dataset_name": "str"
            },
            "action": {
                "action_dim": self.Da,
                "action_chunk_size": self.Tp,
                "action_exec_horizon": self.Ta,
                # "normalize_state": getattr(self.maxmin, "normalize_state", True),
                # "pad_state_dim": self.maxmin.pad_state_dim,
            },
            "rtc_enabled": self.enable_rtc,
            "rtc_mode": self.rtc_mode,
            # RTC geometry, all in rows. exec_shift is what the server assumes the
            # client consumed (always action_exec_horizon); d/s are tuning.
            "rtc_exec_shift": self.exec_shift if self.enable_rtc else None,
            "rtc_inference_delay": self.rtc_inference_delay if self.enable_rtc else None,
            "rtc_min_exec_horizon": self.min_exec_horizon if self.enable_rtc else None,
            "rtc_max_delay": self.rtc_max_delay if self.rtc_mode == "train" else None,
            "pig_mask_schedule": self.pig_mask_schedule if self.rtc_mode == "test_time" else None,
            "pig_guidance_alpha": self.pig_guidance_alpha if self.rtc_mode == "test_time" else None,
            "num_inference_steps": self.num_inference_steps,
        }
        return JSONResponse(content=content)


    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self.app = FastAPI()
        self.app.post("/act")(self.predict_action)
        self.app.get("/info")(self.info)
        self.app.get("/health")(lambda: JSONResponse(content={"status": "ok"}))
        overwatch.info(f"Server listens on {host}:{port}")
        try:
            uvicorn.run(self.app, host=host, port=port)
        except Exception as e:
            overwatch.warning(f"Server crashed, {e}")
        finally:
            overwatch.info("Server stopped.")
            exit(1)

def serve(cfg: ServerConfig) -> None:
    overwatch.info("Server :: Initializing Psi0")
    assert cfg.policy is not None, "which policy to serve?"
    server = Server(
        cfg.policy, 
        Path(cfg.run_dir), 
        cfg.ckpt_step, 
        cfg.device, 
        cfg.rtc,
        cfg.action_exec_horizon,
        rtc_mode=cfg.rtc_mode,
        pig_mask_schedule=cfg.pig_mask_schedule,
        pig_guidance_alpha=cfg.pig_guidance_alpha,
        rtc_inference_delay=cfg.rtc_inference_delay,
        min_exec_horizon=cfg.min_exec_horizon,
        num_inference_steps=cfg.num_inference_steps,
    )
    
    overwatch.info("Server :: Spinning Up")
    server.run(cfg.host, cfg.port)

def main():
    overwatch.info("Start Serving from uv")
    overwatch.info(f"Args: {sys.argv}")
    from dotenv import load_dotenv
    assert load_dotenv() 
    config = tyro.cli(ServerConfig, config=(tyro.conf.ConsolidateSubcommandArgs,), args=sys.argv[1:])
    serve(config)

if __name__ == "__main__":
    from dotenv import load_dotenv
    assert load_dotenv()
    config = tyro.cli(ServerConfig, config=(tyro.conf.ConsolidateSubcommandArgs,))
    serve(config)