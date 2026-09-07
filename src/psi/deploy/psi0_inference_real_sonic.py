import os
from pathlib import Path

# Set working directory to project root, so that load_dotenv() below finds the
# repo .env and the relative run_dir resolves.
# repo_root/src/psi/deploy/psi0_inference_real_sonic.py -> repo_root
REPO_ROOT = Path(__file__).resolve().parents[3]
os.chdir(REPO_ROOT)
os.environ['PWD'] = str(REPO_ROOT)

import dotenv
dotenv.load_dotenv()

import torch
import numpy as np
import json
from psi.utils import parse_args_to_tyro_config, apply_legacy_model_config_defaults  #, seed_everything, move_to_device, batchify
from psi.config.config import LaunchConfig

ckpt_step = 40000
run_dir = Path(".runs/sonic/sonic.lantent64.nortc.real.flow1000.cosine.lr1.0e-04.b128.gpus4.2604120038")
config_:LaunchConfig = parse_args_to_tyro_config(run_dir / "argv.txt") # type: ignore
conf = apply_legacy_model_config_defaults(
    json.loads((run_dir / "run_config.json").read_text()))
launch_config = config_.model_validate(conf)


from psi.config.data_lerobot import LerobotDataConfig
data_cfg: LerobotDataConfig = launch_config.data # type: ignore

from psi.config.model_psi0 import Psi0ModelConfig
model_cfg: Psi0ModelConfig = launch_config.model # type: ignore

# Use GPU 0
DEVICE = "cuda:0"
print(f"Using device: {DEVICE}")
print(f"GPU name: {torch.cuda.get_device_name(0)}")

from psi.models.psi0 import Psi0Model 
model = Psi0Model.from_pretrained(run_dir, ckpt_step, launch_config, device=DEVICE)
model.to(DEVICE)
model.eval()

maxmin = data_cfg.transform.field

vlm_processor = model.vlm_processor
transform_kwargs=dict(
    vlm_processor=vlm_processor,
)
val_dataset = data_cfg(split="val", transform_kwargs=transform_kwargs)
print(f"Validation dataset size: {len(val_dataset)}")

from PIL import Image
import numpy as np
np.set_printoptions(precision=4, suppress=True)

l2_xyz = []
num_eval=300

dataset = val_dataset
eps_idx = 18
skip = 1

np.random.seed(42)
random_indices = np.random.choice(len(dataset), size=min(num_eval, len(dataset)), replace=False)

start_frame_idx = val_dataset.raw_dataset.base_dataset.episode_data_index["from"][eps_idx].item()
end_frame_idx = val_dataset.raw_dataset.base_dataset.episode_data_index["to"][eps_idx].item()
print("number of frames: ", end_frame_idx - start_frame_idx)

avg_action_errors_denormed_list = []
# for i in random_indices:
from tqdm import tqdm
for i in tqdm(range(start_frame_idx, end_frame_idx, skip)):
    frame = val_dataset[i]
    images = frame["raw_images"] # List[PIL.Image.Image] # (0~255)
    batch_images = [images] # List[List[PIL.Image.Image]] batch size == 1

    instruction = frame['instruction']
    batch_instructions = [instruction] # List[str]

    states = frame['states'] # (1, 32)
    batch_states = torch.from_numpy(states).unsqueeze(0).to(DEVICE) # (B, H, D)

    pred_actions = model.predict_action(
        observations=batch_images, 
        states=batch_states, 
        instructions=batch_instructions, 
        num_inference_steps=10, 
        traj2ds=None)

    gt_action = torch.from_numpy(frame["raw_actions"]).unsqueeze(0).to(DEVICE) # (6, 7)
    denormalized_pred_actions = maxmin.denormalize(pred_actions)
    error = denormalized_pred_actions - gt_action # (B, 6, 7)
    error_l1 = error.detach().abs().cpu().numpy().reshape(-1, 36)

    # action L1 errors
    avg_action_errors_denormed = error_l1.mean(0)  # (36,) NOTE only if the error is L1 (linear)

    labels_denormed = [
        "latent_action",
        "hand_joints"
    ]

    avg_lr_action_err_denormed = np.split(
        avg_action_errors_denormed, [64,], axis=-1
    )
    avg_action_errors_denormed_list.append(avg_action_errors_denormed)

    # log metrics
    for i in range(len(avg_lr_action_err_denormed)):
        tqdm.write(f"denormed_err_l1_{labels_denormed[i]}: {np.linalg.norm(avg_lr_action_err_denormed[i])}")

avg_action_errors_denormed_list = np.stack(avg_action_errors_denormed_list, axis=0)
avg_action_errors_denormed = avg_action_errors_denormed_list.mean(axis=0)

labels_denormed = [
   "latent_action",
    "hand_joints"
]

avg_action_errors_denormed_split = np.split(
    avg_action_errors_denormed, 
    [64,],
    axis=-1
)

print("\n---------------------------\n")
for i in range(len(avg_action_errors_denormed_split)):
    print(f"denormed_err_l1_{labels_denormed[i]}: {np.linalg.norm(avg_action_errors_denormed_split[i])}")