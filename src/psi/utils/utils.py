from __future__ import annotations

import json
import os
import random
import re
import importlib.util
import sys
from typing import Any, Union, Mapping, List, Tuple, TYPE_CHECKING
from pathlib import Path
from typing import Callable, Optional
import importlib.resources as resources
import zipfile
if TYPE_CHECKING:
    from psi.data.gear.schema.lerobot import StateActionMetadata
import numpy as np
import torch
from PIL import Image
from copy import deepcopy
import tyro
import sys

def seed_everything(seed):
    # accelerator.set_seed(cfg.seed, device_specific=True) TODO preferr this way!
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    try:
        import tensorflow as tf
        # unfornately, still not obtaining deterministic results for RLDS
        tf.random.set_seed(seed) 
    except:
        pass

    # torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True) # FIXME 
    # torch.use_deterministic_algorithms(True) # FIXME 
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # for reproducibility in CUBLAS


def flatten(dictionary, parent_key='', separator='/'):
    """ https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys """
    from collections.abc import MutableMapping

    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten(value, new_key, separator=separator).items())
        else:
            if callable(getattr(value, 'item', None)):
                value = value.item()
            items.append((new_key, value))
    return dict(items)

def nice(x):
    if type(x) == float:
        return f"{x:.1e}"
    elif type(x) == list or type(x) == tuple:
        return ",".join([f"{i:.1e}" for i in x])
    return x

# turns long lr scheduler name short. eg: constant_with_warmup -> cww, constant-with-warmup -> cww
def shorten(x):
    if "_" in x or "-" in x:
        return "".join([y[0] for y in re.split(r"[-_]", x)])
    return f"{x[:6]}"

def inspect(x):
    return (x.max(), x.min(), x.mean(), x.std())

def pt_to_pil(x, normalized=True):
    s, b = (0.5, 0.5) if normalized else (1.0, 0.0)
    return Image.fromarray(
        (((x.float() * s + b).clamp(0, 1))*255.0).permute(1,2,0).cpu().numpy().astype(np.uint8)
    )

def rmse(l1_err_list):
    squared_errors = l1_err_list ** 2
    mean_squared_error = np.mean(squared_errors)
    rmse_value = np.sqrt(mean_squared_error)
    return rmse_value

def snake_to_pascal(snake_str: str) -> str:
    parts = re.split(r"[-_]", snake_str)
    return ''.join([parts[0].capitalize()] + [p.capitalize() for p in parts[1:]])

def set_global_seed(seed: int, get_worker_init_fn: bool = False) -> Optional[Callable[[int], None]]:
    """Sets seed for all randomness libraries (mostly random, numpy, torch) and produces a `worker_init_fn`"""
    import numpy as np
    import torch
    
    assert np.iinfo(np.uint32).min < seed < np.iinfo(np.uint32).max, "Seed outside the np.uint32 bounds!"

    # Set Seed as an Environment Variable
    os.environ["EXPERIMENT_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    return worker_init_function if get_worker_init_fn else None

def worker_init_function(worker_id: int) -> None:
    """
    Borrowed directly from PyTorch-Lightning; inspired by this issue comment in the PyTorch repo:
        > Ref: https://github.com/pytorch/pytorch/issues/5059#issuecomment-817392562

    Intuition: You can think of the seed sequence spawn function as a "janky" torch.Generator() or jax.PRNGKey that
    you can run iterative splitting on to get new (predictable) randomness.

    :param worker_id: Identifier for the given worker [0, num_workers) for the Dataloader in question.
    """
    import numpy as np
    import torch
    
    # Get current `rank` (if running distributed) and `process_seed`
    global_rank, process_seed = int(os.environ["LOCAL_RANK"]), torch.initial_seed()

    # Back out the "base" (original) seed - the per-worker seed is set in PyTorch:
    #   > https://pytorch.org/docs/stable/data.html#data-loading-randomness
    base_seed = process_seed - worker_id

    # "Magic" code --> basically creates a seed sequence that mixes different "sources" and seeds every library...
    seed_seq = np.random.SeedSequence([base_seed, worker_id, global_rank])

    # Use 128 bits (4 x 32-bit words) to represent seed --> generate_state(k) produces a `k` element array!
    np.random.seed(seed_seq.generate_state(4))

    # Spawn distinct child sequences for PyTorch (reseed) and stdlib random
    torch_seed_seq, random_seed_seq = seed_seq.spawn(2)

    # Torch Manual seed takes 64 bits (so just specify a dtype of uint64
    torch.manual_seed(torch_seed_seq.generate_state(1, dtype=np.uint64)[0])

    # Use 128 Bits for `random`, but express as integer instead of as an array
    random_seed = (random_seed_seq.generate_state(2, dtype=np.uint64).astype(list) * [1 << 64, 1]).sum()
    random.seed(random_seed)

def get_asset_dir() -> Path:
    return (resources.files(__package__.split('.')[0])  / ".." / ".." ).resolve() / "assets" # type: ignore

def get_cache_dir() -> Path:
    if "HF_HOME" in os.environ:
        return Path(os.environ["HF_HOME"])
    return (resources.files(__package__.split('.')[0]) / ".." / ".." ).resolve() / ".cache" # type: ignore

def get_data_dir():
    if "DATA_HOME" in os.environ:
        return Path(os.environ["DATA_HOME"])
    if "PSI_HOME" in os.environ:
        return Path(os.environ["PSI_HOME"]) / "data"
    return (resources.files(__package__.split('.')[0]) / ".." / ".." ).resolve() / ".data" # type: ignore

def get_we_dir():
    # if "WE_HOME" in os.environ:
    #     return Path(os.environ["WE_HOME"]) / "runs"
    return (resources.files(__package__.split('.')[0]) / ".." / ".." ).resolve()  # type: ignore

def resolve_data_path(path: Union[str, Path], auto_download=False) -> Path:
    return resolve_path(path, subdir="data", auto_download=auto_download)

def resolve_path(path: Union[str, Path], subdir="data", auto_download=False) -> Path:
    if Path(path).absolute().exists():
       return Path(path).absolute()
    
    source_dir = get_asset_dir() / path
    if source_dir.exists():
        return source_dir
    
    if "DATA_HOME" in os.environ and subdir == "data":
        data_dir = Path(os.environ["DATA_HOME"])
        filepath = data_dir / path
        if filepath.exists():
            return filepath
    
    if "PSI_HOME" in os.environ:
        proj_dir = Path(os.environ["PSI_HOME"])
        filepath = proj_dir / subdir / path
        if filepath.exists():
            return filepath
        
    if auto_download:
        # auto download the file from remmote huggingface repo "USC-PSI-Lab/psi-data" and extract to the data dir
        from huggingface_hub import snapshot_download
        repo_id = "USC-PSI-Lab/psi-data"
        filename = path if isinstance(path, str) else path.name
        try:

            def _parse_zip_file_from_rel_path(rel_path: str) -> str:
                pattern = r"^assets/robots/([^/]+)/.+\.urdf$"
                match = re.match(pattern, rel_path)
                if match:
                    robot_name = match.group(1)
                    return f"assets/robots/{robot_name}.zip"
                
                pattern = r"^data/egodex/([^/]+)/.+$"
                match = re.match(pattern, rel_path)
                if match:
                    data_name = match.group(1)
                    return f"data/egodex/{data_name}.zip"

                ... # add more patterns as needed

            zip_file = _parse_zip_file_from_rel_path(filename)
            # print(zip_file);
            # auto download
            snapshot_download(
                repo_id="USC-PSI-Lab/psi-data",
                allow_patterns=[zip_file],
                local_dir=os.getcwd(),
                repo_type="dataset",
                # resume_download=True,
                # token="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",
            )
            # print(data_dir);exit(0)
            # zip_path = os.path.join(data_dir, zip_file)
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(os.path.dirname(zip_file))
        except Exception as e:
            print(f"Failed to download {filename} from HuggingFace repo {repo_id}: {e}")
            raise e 
    return Path(path)

def move_to_device(
    batch,
    device, 
    dtype=None
):
    """
    Recursively moves a batch of data (dictionary, list, or tensor) to the specified device.
    
    Args:
        batch (dict, list, tensor): The batch of data.
        device (torch.device or str): The target device ('cuda' or 'cpu').
        dtype: Optional dtype to cast to. If None, preserves original dtype.
    
    Returns:
        The batch moved to the specified device.
    """
    
    if isinstance(batch, torch.Tensor):
        if dtype is not None:
            return batch.to(device, dtype)
        else:
            return batch.to(device)  # Preserve original dtype
    elif isinstance(batch, dict):
        return {key: move_to_device(value, device, dtype) for key, value in batch.items()}
    elif isinstance(batch, list):
        return [move_to_device(item, device, dtype) for item in batch]
    elif isinstance(batch, tuple):
        return tuple(move_to_device(item, device, dtype) for item in batch)
    elif isinstance(batch, np.ndarray):
        tensor = torch.from_numpy(batch)
        if dtype is not None:
            return tensor.to(device, dtype)
        else:
            return tensor.to(device)  # Preserve numpy's original dtype
    else:
        try:
            import transformers
            if isinstance(batch, transformers.feature_extraction_utils.BatchFeature):
                return batch.to(device)
        except ImportError:
            pass
        return batch  # Return as is if not a recognized type

def overlay(rgb, traj2d, black=0.0, eps=0.0):
    """
    Overlay a 2D trajectory on an RGB image.
    black: Value to consider as black in the trajectory tensor
    eps: Small value to tolerate 
    """
    import torch
    rgb = torch.clone(rgb)
    lower = black + eps
    rgb[traj2d > lower] = traj2d[traj2d > lower]
    return rgb

def count_parameters(model, trainable: bool = False):
    """Return total and trainable parameter counts of a model."""
    total_params = sum(p.numel() for p in model.parameters())
    if trainable:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params, trainable_params
    else:
        return total_params
    # print(f"Total parameters: {total_params:,}")
    # print(f"Trainable parameters: {trainable_params:,}")

def str_to_tensor(s: str, max_length: int | None = None, pad_value: int = 0) -> torch.Tensor:
    data = list(s.encode("utf-8"))
    if max_length is not None:
        data = data[:max_length]
        if len(data) < max_length:
            data = data + [pad_value] * (max_length - len(data))
    return torch.tensor(data, dtype=torch.uint8)

def tensor_to_str(t: torch.Tensor, strip_trailing_pad: bool = True, pad_value: int = 0) -> str:
    values = t.tolist()
    if strip_trailing_pad:
        while len(values) > 0 and values[-1] == pad_value:
            values.pop()
    return bytes(values).decode("utf-8")

def string_compatible_collate(batch):
    
    collated = {}
    keys = batch[0].keys()
    for k in keys:
        values = [d[k] for d in batch]
        if isinstance(values[0], str):
            # Convert list of strings to list of tensors, then pad
            tensors = [str_to_tensor(v) for v in values]
            padded = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True)
            collated[f"{k}_str"] = padded
        elif torch.is_tensor(values[0]):
            collated[k] = torch.stack(values)
        elif isinstance(values[0], np.ndarray):
            collated[k] = torch.from_numpy(np.stack(values))
        else:
            if isinstance(values[0], dict):
                collated[k] = string_compatible_collate(values)
            else:
                collated[k] = torch.tensor(np.array(values))
    return collated

from typing import TypeVar, Dict
T = TypeVar("T")

def batch_str_to_tensor(batch: T) -> T:
    import torch
    
    if isinstance(batch, torch.Tensor):
        return batch_str_to_tensor({"data": batch})["data"]
    ret = {}
    for k, v in batch.items():
        if k.endswith("_str"):
            v = [tensor_to_str(s) for s in (v)]
        ret[k.replace("_str", "")] = v
    return ret # type: ignore

# wrote for pytests
def extract_args_from_shell_script(script_path):
    """Extract arguments from a shell script that uses accelerate launch"""

    with open(script_path, 'r') as f:
        content = f.read()

    # Extract from args variable if it exists; anchor the closing quote to its own
    # line so inner quotes (e.g. $(date +"%y%m%d%H%M")) don't truncate the block.
    args_match = re.search(r'args="(.*?)\n"', content, re.DOTALL) or \
        re.search(r'args="([^"]+)"', content, re.DOTALL)
    if args_match:
        args_content = args_match.group(1)
        # poor-man's shell expansion so tokens stay whitespace-splittable:
        # $(date +FMT) via strftime, then ${VAR:-default} from env or default
        from datetime import datetime
        args_content = re.sub(
            r'\$\(date \+"?([^")]+)"?\)',
            lambda m: datetime.now().strftime(m.group(1)),
            args_content,
        )
        args_content = re.sub(
            r'\$\{(\w+):-([^}]*)\}',
            lambda m: os.environ.get(m.group(1)) or m.group(2),
            args_content,
        )
        # Split by backslash and newline, clean up
        args_list = [
            arg.strip().rstrip('\\').strip()
            for arg in args_content.split('\n')
            if arg.strip() and not arg.strip().startswith('#')
        ]
        
        # Split each argument by space to handle cases like "--train.lr_scheduler_kwargs.betas 0.9 0.999"
        final_args = []
        for arg in args_list:
            if arg:
                final_args.extend(arg.split())
        
        return final_args
    
    print("Could not find args variable in the script.")
    return []

def parse_args_to_tyro_config(args_or_script_path: Path | str, force_rewrite_config_file=False):
    assert Path(args_or_script_path).exists(), f"Path does not exist: {args_or_script_path}"

    if str(args_or_script_path).endswith(".sh"):
        print(f"Parsing configuration {args_or_script_path}")
        argv = [sys.argv[0]] + extract_args_from_shell_script(args_or_script_path)
    else:
        argv = []
        with open(args_or_script_path, "r") as f:
            for line in f:
                # Split by any whitespace (spaces, tabs, newlines)
                argv.extend(line.strip().split())

    assert len(argv) > 1, "Please put all arguments in a spererate variable, eg., args=..."
    sys.argv = argv

    try:
        # By convention, the first argument after the script name is the config module name, 
        # eg., {trainer}_{data}_{model}_config, which corresponds to psi.config.train.pretrain_egodex_qwen3vl_config
        module = importlib.import_module(f"psi.config.train.{sys.argv[1]}")
        DynamicLaunchConfigClass =  getattr(module, "DynamicLaunchConfig")
        config = tyro.cli(DynamicLaunchConfigClass, config=(tyro.conf.ConsolidateSubcommandArgs,), args=sys.argv[2:])
    except Exception as e:
        print(f"Failed to import config module 'psi.config.train.{sys.argv[1]}'")
        raise e

    return config

def batchify(data):
    data = deepcopy(data)
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            data[k] = v[None]
        elif isinstance(v, np.ndarray):
            data[k] = torch.from_numpy(v)[None]
        elif isinstance(v, str):
            data[k] = [v]
    return data


def make_image_grid(images, nrows=None, ncols=None):
    # Determine grid dimensions (e.g., 2x2 for 4 images)
    num_images = len(images)
    if nrows is None and ncols is None:
        if num_images <= 3:
            nrows, ncols = 1, num_images
        else:
            grid_size = int(np.ceil(np.sqrt(num_images)))
            nrows, ncols = grid_size, grid_size
            if (nrows -1 ) * ncols >= num_images:
                nrows -= 1
    elif nrows is None:
        nrows = int(np.ceil(num_images / (ncols or 1)))
    else:
        ncols = int(np.ceil(num_images / nrows))
    
    W,H   = images[0].size
    grid = Image.new("RGB", size=(ncols * W, nrows * H))

    for i, image in enumerate(images):
        row, col = divmod(i, ncols or 1)
        grid.paste(image, box = (col * W, row * H))
    return grid

def pad_to_len(x, target_len, dim=1, pad_value=0.0):
    """Pads a tensor to the target length along the specified dimension.
    Args:
        x: np.ndarray to pad
        target_len: int, target length to pad to
        dim: int, dimension along which to pad
        pad_value: value to use for padding
    Returns:
        padded: np.ndarray, padded array
        mask: np.ndarray of bool, True for original data, False for padded region
    """
    current_len = x.shape[dim]
    if target_len is None or current_len >= target_len:
        mask = np.ones(x.shape, dtype=bool)
        return x, mask
    pad_width = [(0, 0)] * x.ndim
    pad_width[dim] = (0, target_len - current_len)
    # np.pad pads as (before, after) for each axis
    padded = np.pad(x, pad_width, mode='constant', constant_values=pad_value)
    mask_shape = list(x.shape)
    mask_shape[dim] = target_len
    mask = np.ones(mask_shape, dtype=bool)
    # Mark padded region as False (0)
    idx = [slice(None)] * x.ndim
    idx[dim] = slice(current_len, target_len)
    mask[tuple(idx)] = False
    return padded, mask

def split_by_modality(data, modality_metadata: dict[str, StateActionMetadata]):
    # split data dimmension according to modality metadata, return dict of modality to data
    result = {}
    start = 0
    for mod, meta in modality_metadata.items():
        end = start + meta.shape[0]
        result[mod] = data[..., start:end]
        start = end
    return result

def dreamzero_instantiate(config, *args, **kwargs):
    from hydra.utils import instantiate
    from omegaconf import DictConfig, OmegaConf

    def _patch(obj):
        if hasattr(obj, "items"):
            if "_target_" in obj and isinstance(obj["_target_"], str) and obj["_target_"].startswith("groot."):
                obj["_target_"] = "dreamzero." + obj["_target_"]
            for k in obj.keys():
                if isinstance(obj, DictConfig) and OmegaConf.is_missing(obj, k):
                    continue
                _patch(obj[k])
        elif isinstance(obj, (list, tuple)) or (hasattr(obj, "__iter__") and not isinstance(obj, str)):
            for v in obj:
                _patch(v)

    _patch(config)
    return instantiate(config, *args, **kwargs)


def cfg_json_default(o):
    """json.dump default handler for pydantic config objects containing torch/numpy types."""
    from enum import Enum
    if isinstance(o, torch.dtype):
        return str(o)
    if isinstance(o, Enum):
        return o.value
    if isinstance(o, (torch.Tensor, np.ndarray)):
        return o.tolist()
    if isinstance(o, np.dtype):
        return str(o)
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    return repr(o)


# Fields added to the model config after some checkpoints were already trained. A
# run_config.json written before the field existed carries no entry for it, and
# pydantic then fills in today's default -- which silently changes what the loaded
# weights compute (same parameter shapes, different forward). Serving must reproduce
# the head the checkpoint was trained with, so back-fill the pre-flag behaviour
# whenever the key is absent.
#
#   final_layer_norm: the action head. False is the legacy `x * scale + shift`;
#     True (today's default) is `LayerNorm(x) * (1 + scale) + shift`. Both have the
#     same weights, so a legacy checkpoint loads clean and then emits garbage
#     actions -- the policy runs, the robot fails the task.
LEGACY_MODEL_CONFIG_DEFAULTS: dict[str, Any] = {
    "final_layer_norm": False,
}

def apply_legacy_model_config_defaults(conf: dict[str, Any]) -> dict[str, Any]:
    """Fill model-config keys a pre-flag run_config.json does not carry (in place)."""
    model_conf = conf.get("model")
    if not isinstance(model_conf, dict):
        return conf
    for key, legacy_value in LEGACY_MODEL_CONFIG_DEFAULTS.items():
        if key not in model_conf:
            model_conf[key] = legacy_value
            print(f"[legacy ckpt] run_config.json predates `model.{key}`; "
                  f"serving with {key}={legacy_value} (the value it was trained with)")
    return conf

def load_launch_config(run_dir: Path | str):
    """Rebuild the LaunchConfig a checkpoint was trained with, from its run dir.

    Reads argv.txt for the config class and run_config.json for the values, then
    back-fills the defaults of any config field added after the run was written.
    """
    run_dir = Path(run_dir)
    config_ = parse_args_to_tyro_config(run_dir / "argv.txt")
    conf = apply_legacy_model_config_defaults(
        json.loads((run_dir / "run_config.json").read_text()))
    return config_.model_validate(conf)
