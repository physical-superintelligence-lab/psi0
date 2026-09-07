from __future__ import annotations
import os
from typing_extensions import Self, Annotated
from pydantic import BaseModel, Field, model_validator
import os
import tyro
from typing import Union, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from psi.data.dataset import TransformableDataset
    
from tyro.conf import subcommand as cmd
from pathlib import Path
import datetime

from psi.config.transform import DataTransform

class LoggingConfig(BaseModel):
    logging_dir: str = "logs"
    report_to: str | None = None
    log_freq: int = 100


class WandbConfig(BaseModel):
    project: str = "psi"
    entity: str | None = None
    group: str | None = None
    id: str | None = None
    name: str | None = None
    resume: str = "allow"  # allow, must, never

    def model_post_init(self, __context: Any) -> None:
        if self.entity is None:
            self.entity = os.getenv("WANDB_ENTITY", None)


class TrainConfig(BaseModel):
    num_workers: int = 8
    overfit_single_batch: bool = False
    name: str = "human3d"  # "vqvae"
    run_tag: str | None = None  # if set, replaces the shorten(dataset_name) segment of the run folder name
    resume_from_checkpoint: str | None = None
    skip_resumed_steps: bool = False

    # HF Hub Credentials (for any gated models)
    hf_token: str | Path = Path(".hf_token")  # Environment variable or Path to HF Token

    lora: bool = False
    output_dir: str = ".runs"
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "bf16"  # "no" for fp32, "fp16", "bf16"
    max_grad_norm: float | None = None  # 1.0
    optimizer_foreach: bool | None = None

    train_batch_size: int = 16
    val_batch_size: int = 16

    val_num_batches: int = 20

    checkpointing_steps: int = 5000
    max_checkpoints_to_keep: int | None = None
    # Checkpoints on this step cadence are exempt from max_checkpoints_to_keep.
    # Set to None to enforce a strict rolling window (useful for large models).
    preserve_checkpoint_every: int | None = 10000
    validation_steps: int = 50

    learning_rate: float = 1e-5
    # linear, cosine, cosine_with_restarts, polynomial, constant, constant_with_warmup
    lr_scheduler_type: str = "cosine_with_min_lr"
    lr_scheduler_kwargs: dict[str, float | tuple[float, ...]] = Field(
        default_factory=lambda: {
            # "num_warmup_steps": 0,
            "betas": (0.9, 0.95),
            "weight_decay": 1e-8,
            "eps": 1e-8,
        }
    )
    scheduler_specific_kwargs: dict[str, float | tuple[float, ...]] = Field(
        default_factory=lambda: {"min_lr": 5.0e-07}
    )

    ## FSDP or DDP
    data_parallel: str = "ddp"  # "deepspeed", "ddp" or "fsdp"
    sharding_strategy: str = "full-shard"
    # Resolved against the repo root by model_post_init below; keep it relative so
    # the same default works from a container checkout (e.g. /workspace). The old
    # src/InternVLA/... path does not exist in this repo, so any run with
    # data_parallel=deepspeed and no explicit override failed at plugin setup.
    deepspeed_config: str = "scripts/deepspeed/zero3.json"
    enable_gradient_checkpointing: bool = True
    enable_mixed_precision_training: bool = True
    reduce_in_full_precision: bool = True

    max_training_steps: int | None = 100_000
    num_train_epochs: int | None = None
    warmup_steps: int | None = 0  # if > 0, overrides warmup_ratio
    warmup_ratio: float | None = 0.05  # used if warmup_steps is not set or 0

    @model_validator(mode="after")
    def check_warmup(self) -> Self:
        steps, ratio = self.warmup_steps, self.warmup_ratio
        if (steps is not None and steps > 0) and (ratio is not None and ratio > 0):
            raise ValueError("Only one of warmup_steps or warmup_ratio can be set")

        steps, ratio = self.max_training_steps, self.num_train_epochs
        if (steps is None or steps == 0) and (ratio is None or ratio == 0):
            raise ValueError(
                "At least one of max_training_steps or num_train_epochs must be set"
            )
        if steps is not None and ratio is not None and steps > 0 and ratio > 0:
            raise ValueError(
                "Only one of max_training_steps or num_train_epochs can be set"
            )

        return self

    def model_post_init(self, __context: Any) -> None:
        if self.lr_scheduler_type != "cosine_with_min_lr":
            self.scheduler_specific_kwargs = {}

        if not os.path.isabs(self.deepspeed_config):
            from psi.utils import resolve_path
            self.deepspeed_config = os.path.abspath(
                resolve_path(self.deepspeed_config)
            )

class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 21074
    device: str = "cuda:0"
    policy: str | None = None
    # Rows of each predicted chunk that /act returns, i.e. what the client executes
    # before asking again.
    action_exec_horizon: int | None = None
    rtc: bool = False
    run_dir: str
    ckpt_step: int | str | None = "latest"
    
    embodiment_tag: str = "decoupled_wbc_g1"
    dashboard: bool = False  # live htop/nvtop-style terminal dashboard (TTY only)
    ctrl_hz: float = 30.0          # control-loop frequency
    min_exec_horizon: int | None = None
    delay_buf_size: int = 6        # recent-delay window length; max() over it feeds next inference_delay
    initial_delay: int = 6         # bootstrap inference delay (ticks), before any is measured
    cfg_scale: float = 1.0         # goal-image classifier-free guidance; 1.0=off, try 1.5~3.0
    
    rtc_mode: str = "auto"
    rtc_inference_delay: int | None = None
    pig_guidance_weight: float = 5.0    # test_time only: cap on the PiGDM step size
    pig_mask_schedule: str = "exponential"   # test_time only: exponential|linear|hard|simple
    pig_sigma_threshold: float = 0.15
    pig_guidance_alpha: float = 0.9

    # Flow-sampling steps at inference. 10 is what the released psi0 checkpoints
    # were validated with; lowering it trades action quality for latency.
    num_inference_steps: int = 10

    @model_validator(mode="after")
    def set_policy(self):
        if self.policy is None:
            run_dir_path = Path(self.run_dir)
            self.policy = run_dir_path.parts[1]
        return self
        

class DataConfig(BaseModel):
    transform: DataTransform

    def __call__(self, split: str = "train", transform_kwargs={}, **kwargs) -> "TransformableDataset":
        raise NotImplementedError
    
class ModelConfig(BaseModel): 
    ...

class DummyModelConfig(ModelConfig):
    # boilerplate model config
    ...
class LaunchConfig(BaseModel):
    # NOTE: This class is used for type hinting only!
    # The real implmentation is DynamicLaunchConfig .

    exp: str = Field(..., description="Name of the experiment", frozen=True)
    seed: int | None = None
    auto_tag_run: bool = False
    eval: bool = False
    debug: bool = False
    timestamp: str | None = None

    log: LoggingConfig
    wandb: WandbConfig
    train: TrainConfig
    data: DataConfig
    model: ModelConfig
    
    def model_post_init(self, __context: Any) -> None:
        is_multinode = (
            "SLURM_NODELIST" in os.environ and
            len(os.environ["SLURM_NODELIST"].split(",")) > 1
        )
        if is_multinode:
            assert self.timestamp is not None, "Timestamp must be provided for multi-node training, eg., --timestamp=$(date +\"%y%m%d%H%M\")"

        def extract_timestamp(folder_name):
            parts = folder_name.split('.')
            return parts[-1] if len(parts) > 1 else ''

        if self.train.resume_from_checkpoint == "latest":
            """ auto resume by looking up timestamp or latest run folder """
            auto_resume_success = False
            trainer_dir = os.path.join(self.train.output_dir, self.train.name)
            if os.path.exists(trainer_dir):
                # Sort folders by timestamp (assumed to be last part after a dot)
                runs = dict(sorted({
                    extract_timestamp(f): os.path.join(trainer_dir, f) for f in os.listdir(trainer_dir)
                }.items(), reverse=True))

                if self.timestamp is not None and self.timestamp in runs:
                    print(f"Will resume latest run with specified timestamp: {self.timestamp}")
                    self.train.resume_from_checkpoint = runs[self.timestamp]
                    auto_resume_success = True

                """ elif len(runs) > 0:
                    latest_timestamp = next(iter(runs))
                    print(f"Will auto-resume from latest run with timestamp: {latest_timestamp}")
                    if self.timestamp is not None and self.timestamp != latest_timestamp:
                        print(f"Overriding timestamp {self.timestamp} with latest timestamp {latest_timestamp}")
                    self.timestamp = latest_timestamp
                    self.train.resume_from_checkpoint = runs[latest_timestamp]
                    auto_resume_success = True """
            
            if not auto_resume_success:
                self.train.resume_from_checkpoint = None

        if self.timestamp is None:
            self.timestamp = datetime.datetime.now().strftime("%y%m%d%H%M")
