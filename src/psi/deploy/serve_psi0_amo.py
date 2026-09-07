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
# import draccus
import uvicorn
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Dict, Any, List
import os.path as osp

from psi.models.psi0 import Psi0Model 

# os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Ensure imports work regardless of current working directory
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from psi.config.config import LaunchConfig, ServerConfig
from psi.deploy.helpers import *

# from pipelines import ActionPipeline
# from misc import move_to_device
from psi.utils import parse_args_to_tyro_config, pad_to_len, seed_everything, apply_legacy_model_config_defaults

from psi.utils.overwatch import initialize_overwatch 
overwatch = initialize_overwatch(__name__)



PREDICT_HORIZON = 30          # == H
MIN_EXEC_HORIZON = 15         # == s_min # TODO: should match D_INIT, ideally s_min >= d_real
DELAY_BUFFER_SIZE = 6        # == delay_buffer_size
D_INIT = 6                   # == d_init # TODO: placeholder, needs calculation
CTRL_PERIOD_SEC = 1. / 30       # 30Hz



class RealTimeChunkController:
    def __init__(self,
                 policy: Psi0Model,
                 prediction_horizon: int = PREDICT_HORIZON,
                 min_exec_horizon: int = MIN_EXEC_HORIZON,
                 delay_buf_size: int = DELAY_BUFFER_SIZE,
                 d_init: int = D_INIT,
                 o_first: np.ndarray | None = None): # type: ignore

        self.policy : Psi0Model = policy
        self.device = self.policy.device
        self.H     = prediction_horizon

        self.s_min = min_exec_horizon

        self.t: int = 0
        assert o_first != None, "please provide o_first"

        A_first = self._predict_action(o_first) # (H, D)

        # warmup the model
        for i in range (2):
            _ = self._predict_action_rtc(copy.deepcopy(o_first), np.concatenate([copy.deepcopy(A_first[self.s_min:, :]), np.zeros((self.s_min, A_first.shape[1]), dtype=A_first.dtype)], axis=0), d_init)
        print("Model warmed up")

        self.A_cur = A_first # (H, D)
        self.o_cur: Dict[str, Any] | None = None 

        self.Q = deque([d_init], maxlen=delay_buf_size)  

        self.M = threading.Lock()
        self.C = threading.Condition(self.M)

        self._infer_th = threading.Thread(target=self._inference_loop, daemon=True)
        self._infer_th.start()

    def replace_prev_actions_to_obs(self, o, previous_rpy, previous_height):
        o['obs'] = np.concatenate([o['obs'][:, :, :28], previous_rpy[np.newaxis, np.newaxis, :], previous_height[np.newaxis, np.newaxis, :], o['obs'][:, :, 32:]], axis=-1) # (1, 1, 28) -> (1, 1, 32)
        return o

        
    def step(self, obs_next: Dict[str, Any]): # consume a_(t-1) and provide o_t
        with self.C:
            self.t += 1
            self.o_cur = obs_next
            self.C.notify()
            if self.t-1 >= len(self.A_cur):
                single_action = self.A_cur[-1]
                print("failed")
            else:
                single_action = self.A_cur[self.t - 1]
            return single_action[np.newaxis, :] # (1, D)

    def _inference_loop(self):
        while True:
            with self.C:
                try:
                    while self.t < self.s_min:
                        self.C.wait() # wait until notified and get the lock
                    s   = self.t

                    # FIXME: 
                    # 1. maybe bug at "s-2"
                    # 2. inputs should be : normalize_states(denormalize_action(A_cur[s-2])) 
                    #    but in our current data, the stats for rpy and height are nearly the same, 
                    #    so "normalize_states(denormalize_action())" equals to doing nothing.

                    assert (s-2) >= 0
                    self.o_cur = self.replace_prev_actions_to_obs(self.o_cur, copy.deepcopy(self.A_cur[s-2, 28:31]), copy.deepcopy(self.A_cur[s-2, 31:32]))
                    #

                    o   = copy.deepcopy(self.o_cur)
                    d   = max(self.Q)
                    # A_prev = copy.deepcopy(torch.cat([self.A_cur[s:, :], torch.zeros((s, self.A_cur.shape[1]), device=self.A_cur.device, dtype=self.A_cur.dtype)], dim=0)) # (H, D)
                    A_prev = np.concatenate([copy.deepcopy(self.A_cur[s:, :]), np.zeros((s, self.A_cur.shape[1]), dtype=self.A_cur.dtype)], axis=0) # (H, D)

                    inference_start = time.perf_counter()
                    self.C.release()
                    A_new = self._predict_action_rtc(o, A_prev, d)
                    self.C.acquire()

                    self.A_cur = A_new
                    self.t = self.t - s
                    self.Q.append(self.t)          
                    # self.C.notify_all()
                    print(f"[inference]  latency={time.perf_counter()-inference_start:.4f}s  s={s}  d={d}  self.t={self.t}")
                except Exception as e:
                    print(f"\n[ERROR] Inference loop crashed!")
                    print(f"Error: {e}")
                    import traceback
                    traceback.print_exc()
                    print("\n[FATAL] Stopping program...")
                    os._exit(1)  # 强制退出整个程序
    
    def _predict_action_rtc(self, o, A_prev, d):
        A_new = self.policy.predict_action_with_training_rtc_flow(
                    observations=o['imgs'], 
                    states=torch.from_numpy(o['obs']).to(self.device),
                    traj2ds=None,
                    instructions=o['text_instructions'],
                    num_inference_steps = 8,
                    prev_actions=torch.from_numpy(A_prev[np.newaxis, :, :]).to(self.device), # (H, D) -> (1, H, D)
                    inference_delay=d,
                    max_delay=8
                )[0].float().detach().cpu().numpy() # (1, H, D) -> (H, D)
        return A_new
    
    def _predict_action(self, o):
        normalized_actions = self.policy.predict_action(
                    observations=o['imgs'], 
                    states=torch.from_numpy(o['obs']).to(self.device),
                    traj2ds=None,
                    instructions=o['text_instructions'],
                    num_inference_steps = 8,
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
        action_exec_horizon: int | None = None
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available. Please check your CUDA installation.")
        
        self.device = torch.device(device)
        overwatch.info(f"Using device: {self.device}")
        overwatch.info(f"Serving {policy}")

        overwatch.info(f"Using device: {self.device}")
        overwatch.info(f"Serving {policy}")

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


        # control - shared state with locks
        self.latest_obs = None
        self.latest_action = None
        self.action_version = 0  # Used by client to check if there's a new action
        
        self.obs_lock = threading.Lock()
        self.action_lock = threading.Lock()

        self.controller = None
        self._control_loop_started = False
        
        # WebSocket: asyncio event to notify when new action is ready
        self.app = FastAPI()
        self._setup_routes()
        
        self._action_ready_event: asyncio.Event = None  # Will be created in async context
        self._active_websocket: WebSocket = None
        self._loop = None  # asyncio event loop reference for thread-safe notification
        self.start_time = time.time()
        self.start_time_obs = time.time()

    def _init_controller(self, o_first):
        controller = RealTimeChunkController(policy=self.model, o_first=o_first)
        return controller

    def _postprocess_action(self, action):
        # return self.launch_cfg.data.data_transforms.denormalize_action(action)
        return self.maxmin.denormalize(action) # denormalization is done in the pipeline



    def preprocess_image(self, image_dict: Dict[str, Any]) -> Dict[str, Any]:
        imgs = {}
        # # FIXME 
        # image_key_to_cam_idx = {'front_stereo_left': 0, 'front_stereo_right': 1, 'left_future_traj_2d': 4, 'right_future_traj_2d': 5, 'side': 3, 'side_future_traj_2d': 7, 'wrist': 2, 'wrist_future_traj_2d': 6}
        # for img_key in self.launch_config.data.transform.repack.image_keys:
        #     cam_idx = image_key_to_cam_idx[img_key] #self.launch_config.data.transform.repack.image_key_to_cam_idx[img_key]
        #     imgs[f"cam{cam_idx}"] = self._process_img(image_dict[f"{img_key}".replace("image_", "")])#[None, ...]

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
        condition_dict = request.condition
        overwatch.info(f"Instruction: {instruction}")
            
        # parts = instruction.split(".")
        # if len(parts) > 1 and parts[-1].isdigit():
        #     instruction = parts[0].lower() + "."
        #     img_id = int(parts[-1])
        #     assert False
        # else:
        instruction = instruction.lower()
        # img_id = -1


        # TODO support image history
        # img dict: {"video": np.array(...).shape(480, 640, 3)}
        imgs = {}
        for cam_idx, img_key in enumerate(self.launch_cfg.data.transform.repack.image_keys):
            imgs[f"cam{cam_idx}"] = Image.fromarray(np.clip(image_dict[img_key], 0, 255).astype(np.uint8))
        
        hand_joints = state_dict["hand_joints"].copy() # shape (14,)
        arm_joints = state_dict["arm_joints"].copy() # shape (14,)
        tmp_torso_rpy = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        tmp_torso_height = np.array([0.75], dtype=np.float32)
        obs = np.concatenate([hand_joints, arm_joints, tmp_torso_rpy, tmp_torso_height], axis=-1) # (32,)

        # normalize states
        assert self.maxmin.normalize_state, "check"
        if self.maxmin.pad_state_dim != len(obs):
            obs = pad_to_len(obs, self.maxmin.pad_state_dim, dim=0)[0]
        obs = self.maxmin.normalize_state_func(obs) # shape (32,)
        obs = obs[np.newaxis, np.newaxis, :] # (32,) -> (1, 1, 32)


        image_input = self.preprocess_image(imgs)
        batch_images = [image_input['cam0']] # batch size == 1


        conditions = {}
        
        text_instructions = [instruction] # len == 1

        return {'imgs': batch_images, 'text_instructions': text_instructions, 'obs': obs, 'conditions': conditions}

    async def websocket_handler(self, websocket: WebSocket):
        """
        WebSocket handler for bidirectional communication:
        - Receive obs from client at high frequency
        - Send action to client immediately when new action is ready
        """
        await websocket.accept()
        self._active_websocket = websocket
        
        # Create asyncio event for action notification
        self._action_ready_event = asyncio.Event()
        
        print("[WebSocket] Client connected")
        async def receive_obs():
            """Continuously receive obs from client"""
            try:
                while True:
                    # Receive obs from client
                    data = await websocket.receive_text()
                    payload = json.loads(data)
                    interval = time.time() - self.start_time_obs
                    self.start_time_obs = time.time()
                    print(f"[WebSocket] receive_obs interval: {interval} seconds")
                    # Parse and update latest_obs
                    this_o = self._parse_obs_payload(payload)
                    with self.obs_lock:
                        self.latest_obs = this_o
                    
                    # If control loop hasn't started, start it
                    if not self._control_loop_started and self.latest_obs is not None:
                        self._start_control_loop()

                    # # 清空缓冲区，只保留最新的
                    # latest_data = None
                    # while True:
                    #     try:
                    #         # 非阻塞地读取所有可用消息
                    #         data = await asyncio.wait_for(
                    #             websocket.receive_text(), 
                    #             timeout=0.001  # 1ms超时
                    #         )
                    #         latest_data = data  # 保留最新的
                    #     except asyncio.TimeoutError:
                    #         break  # 没有更多消息了
                    
                    # if latest_data:
                    #     payload = json.loads(latest_data)
                    #     interval = time.time() - self.start_time_obs
                    #     self.start_time_obs = time.time()
                    #     print(f"[WebSocket] receive_obs interval: {interval} seconds")
                        
                    #     this_o = self._parse_obs_payload(payload)
                    #     with self.obs_lock:
                    #         self.latest_obs = this_o
                        
                    #     if not self._control_loop_started and self.latest_obs is not None:
                    #         self._start_control_loop()
                        
            except WebSocketDisconnect:
                print("[WebSocket] Client disconnected (receive)")
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[WebSocket] Receive error: {e}")
        
        async def send_action():
            """Send action to client when new action is ready"""
            try:
                while True:
                    # Wait for new action to be ready
                    await self._action_ready_event.wait()
                    self._action_ready_event.clear()

                    interval = time.time() - self.start_time
                    self.start_time = time.time()
                    print(f"[WebSocket] send_action interval: {interval} seconds")
                    
                    # Get the action
                    with self.action_lock:
                        action = self.latest_action
                        version = self.action_version
                        self.latest_action = None  # Reset after sending
                    
                    if action is not None:
                        # Send action to client
                        response = ResponseMessage(action, err=0.0)
                        resp_dict = response.serialize()
                        resp_dict["version"] = version
                        await websocket.send_text(json.dumps(resp_dict))
                        print(f"[WebSocket] Sent action, version={version}")
                    else:
                        assert False, "action is None"
                        
            except WebSocketDisconnect:
                print("[WebSocket] Client disconnected (send)")
            except Exception as e:
                print(f"[WebSocket] Send error: {e}")
        
        try:
            # Run both tasks concurrently
            await asyncio.gather(receive_obs(), send_action())
        except Exception as e:
            print(f"[WebSocket] Connection closed: {e}")
        finally:
            self._active_websocket = None
            print("[WebSocket] Handler finished")

    def _start_control_loop(self):
        """Start control loop thread"""
        if self._control_loop_started:
            return
        self._control_loop_started = True
        
        # Initialize controller with first obs
        with self.obs_lock:
            o_first = copy.deepcopy(self.latest_obs)
            
        self.controller = self._init_controller(o_first) # wait for model warm up
        
        # Start control loop thread
        self._control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self._control_thread.start()
        print("[control loop] started")

    def _control_loop(self):
        """
        Control loop: Execute controller.step strictly every CTRL_PERIOD_SEC
        And expect at next time, the obs_next sent from client is the one after executing the action
        """
        next_tick = time.perf_counter()
        prev_tick = time.perf_counter()
        
        while True:
            # loop_start = time.time()
            
            # 1. Get latest obs
            with self.obs_lock:
                obs_next = copy.deepcopy(self.latest_obs)
            
            # 2. Execute step
            action = self.controller.step(obs_next) # (1, D)
            pred_action = self._postprocess_action(action) # (1, D)
            
            # 3. Update latest_action
            with self.action_lock:
                self.latest_action = pred_action
                self.action_version += 1
            
            # 4. Notify WebSocket that new action is ready
            if self._action_ready_event is not None:
                # Thread-safe way to set asyncio event from another thread
                try:
                    self._loop.call_soon_threadsafe(self._action_ready_event.set)
                except Exception as e:
                    print(f"[control loop] Failed to notify WebSocket: {e}")
            
            # elapsed = (time.time() - loop_start) * 1000
            # print(f"[control loop] step took {elapsed:.1f}ms, version={self.action_version}")
            
            # 5. Wait until next ctrl period
            next_tick += CTRL_PERIOD_SEC
            sleep_time = next_tick - time.perf_counter()
            now = time.perf_counter()
            interval = now - prev_tick
            prev_tick = now
            print(f"[control loop] interval: {interval} seconds")
            if sleep_time > 0:
                time.sleep(sleep_time)
                # delay_ms(sleep_time*1000)
            else:
                print(f"[control loop] WARNING: missed tick by {-sleep_time*1000:.1f}ms")
                next_tick = time.perf_counter()
    

    def _setup_routes(self):
        """Set up all the API endpoints including both WebSocket and HTTP routes"""
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            self._loop = asyncio.get_event_loop()
            await self.websocket_handler(websocket)
        
        @self.app.get("/health")
        async def health_check():
            return {"status": "ok"}

    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        print(f"Server listens on {host}:{port}")
        print(f"WebSocket endpoint: ws://{host}:{port}/ws")
        try:
            uvicorn.run(self.app, host=host, port=port)
        except Exception as e:
            print(f"Server crashed, {e}")
        finally:
            print("Server stopped.")
            exit(1)

def serve(cfg: ServerConfig) -> None:
    overwatch.info("Server :: Initializing Policy")
    assert cfg.policy is not None, "which policy to serve?"
    assert cfg.rtc, "this server is for rtc"
    server = Server(
        cfg.policy, 
        Path(cfg.run_dir), 
        cfg.ckpt_step, 
        cfg.device,
        cfg.rtc,
        cfg.action_exec_horizon)
    
    print("Server :: Spinning Up")
    server.run(cfg.host, cfg.port)

def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()  # take environment variables from .env file
    config = tyro.cli(ServerConfig, config=(tyro.conf.ConsolidateSubcommandArgs,))
    serve(config)

if __name__ == "__main__":
    main()