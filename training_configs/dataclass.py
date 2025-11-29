from dataclasses import dataclass

@dataclass
class TrainingArgs:
    # Default values from Lidar training code
    warmup_epochs: int = 5
    grad_clip: float = 1.0
    log_freq: int = 10
    # Other params
    llm: str = None
    freeze_encoder: bool = True
    freeze_lang_model: bool = True
    epochs: int = None
    batch_size: int = None
    run_name: str = None
    save_every: int = None
    wandb_project: str = None
    resume_from_checkpoint: str = None
    lr: float = None
    num_workers: int = None
    penalty_weight: float = None
    version: str = None
    dataroot: str = None
    output_dir: str = None