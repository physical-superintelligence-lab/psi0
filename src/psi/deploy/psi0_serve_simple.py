import os
import sys
import tyro
import torch
import time
import numpy as np
import os.path as osp
from pathlib import Path
import uvicorn
from fastapi import FastAPI
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
        action_exec_horizon: int | None = None
    ):
        _dt = str(device).split(":")[0]
        if _dt == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please check your CUDA installation.")
         
        self.device = torch.device(device)
        overwatch.info(f"Using device: {self.device}")
        overwatch.info(f"Serving {policy}")

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
        
        self.enable_rtc = enable_rtc
        if enable_rtc:
            assert launch_config.model.rtc, "rtc is not supported for this model" #type:ignore
            self.rtc_max_delay = launch_config.model.max_delay  # type:ignore
            assert self.Tp - self.Ta <= self.rtc_max_delay, "action_exec_horizon is too big for the given rtc_max_delay and action_chunk_size"
            self.previous_action = None #np.zeros((self.Tp, self.Da), dtype=np.float32)
            overwatch.info(f"RTC enabled with max_delay={self.rtc_max_delay}, \n"
                           f"action_dim={self.Da}, \n"
                           f"action_chunk_size={self.Tp}, \n"
                           f"action_exec_horizon={self.Ta}")
        self.last_serve_time = time.monotonic()


    def predict_action(self, payload: Dict[str, Any]) -> JSONResponse:
        # overwatch.info(f"Received request with payload: {payload}")
        try:
            request = RequestMessage.deserialize(payload)
            image_dict, instruction, history_dict, state_dict, gt_action, dataset_name = \
                request.image, request.instruction, request.history, request.state, request.gt_action, request.dataset_name
            
            overwatch.info(f"Instruction: {instruction}")
            overwatch.info(f"history_dict: {history_dict}")

            transforms = [self.model_transform.resize(), self.model_transform.center_crop()]
            t = v2.Compose(transforms)

            states = torch.from_numpy(state_dict["states"].copy())
            # self.repack_transform.to_psi0_state_format(
            #     torch.from_numpy(state_dict["proprio_joint_positions"].copy()),
            #     torch.from_numpy(state_dict["amo_policy_command"].copy()),
            # )

            if self.maxmin.normalize_state: # type:ignore
                states = torch.from_numpy(
                    self.maxmin.normalize_state_func(
                        pad_to_len(states.numpy(), self.maxmin.pad_state_dim, dim=1)[0]
                    )
                ).to(self.device)

            if not self.enable_rtc:
                raw_pred_actions = self.model.predict_action(
                    observations=[[t(Image.fromarray(img)) for img in image_dict.values()]], 
                    states=states.unsqueeze(0), # B, To, Ds
                    instructions=[instruction], # [Task] * B
                    num_inference_steps=10, 
                    traj2ds=None
                )
            else: # rtc
                current_time = time.monotonic()
                if self.previous_action is None or "reset" in history_dict: #  or (current_time - self.last_serve_time) > 30  #if idle more than 60s, reset previous action
                    overwatch.info("===Reset or first step, without condition===")
                    raw_pred_actions = self.model.predict_action(
                        observations=[[t(Image.fromarray(img)) for img in image_dict.values()]], 
                        states=states.unsqueeze(0), # B, To, Ds
                        instructions=[instruction], # [Task] * B
                        num_inference_steps=10, 
                        traj2ds=None
                    )
                else:
                    overwatch.info("RTC enabled, using RTC inference")
                    overwatch.info("Last chunk execution loop time: {:.2f}s ago".format(current_time - self.last_serve_time))
                    prev_actions = np.concatenate([
                        self.previous_action[None, self.Ta:, :], 
                        np.zeros((1, self.Ta, self.Da), dtype=np.float32)
                    ], axis=1) # (1, Tp, Da)
                    prev_actions = torch.from_numpy(prev_actions).to(self.device)

                    raw_pred_actions = self.model.predict_action_with_training_rtc_flow(
                        observations=[[t(Image.fromarray(img)) for img in image_dict.values()]], 
                        states=states.unsqueeze(0), # B, To, Ds
                        instructions=[instruction], # [Task] * B
                        num_inference_steps=10, 
                        traj2ds=None,
                        prev_actions=prev_actions,
                        inference_delay=(self.Tp - self.Ta), 
                        max_delay=self.rtc_max_delay
                    )

            raw_pred_actions = raw_pred_actions.reshape(-1, self.Da).cpu().numpy() # (Tp, Da)
            pred_actions = self.maxmin.denormalize(raw_pred_actions) # (Ta, Da)
            self.previous_action = raw_pred_actions.copy().astype(np.float32) # for rtc
            pred_actions = pred_actions[:self.Ta] # type:ignore
            overwatch.info(f"Return Action ({pred_actions.shape})") # : {pred_actions}

            self.last_serve_time = time.monotonic()
            response = ResponseMessage(pred_actions, 0.0) # type:ignore
            return JSONResponse(content=response.serialize())

        except Exception as e:
            import traceback
            overwatch.warning(traceback.format_exc())
            return JSONResponse(content=f'{{"status": "{e}"}}')

    
    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self.app = FastAPI()
        self.app.post("/act")(self.predict_action)
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
        cfg.action_exec_horizon
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