from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class DataConfig:
    data_dir: Path = Path("data")
    train_subset: str = "train-clean-100"
    eval_subset: str = "dev-clean"
    sample_rate: int = 16000
    max_audio_length_seconds: float = 20.0
    min_audio_length_seconds: float = 0.5


@dataclass
class ModelConfig:
    model_name: str = "facebook/wav2vec2-base"
    vocab_file: str = "vocab.json"
    ctc_loss_reduction: str = "mean"
    ctc_zero_infinity: bool = True
    mask_time_prob: float = 0.05
    mask_time_length: int = 10
    mask_feature_prob: float = 0.0
    mask_feature_length: int = 10


@dataclass
class Stage1Config:
    output_dir: str = "outputs/stage1"
    num_train_epochs: int = 2
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.005
    freeze_feature_extractor: bool = True
    freeze_transformer: bool = False
    use_specaugment: bool = False
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    logging_steps: int = 100
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "wer"
    greater_is_better: bool = False


@dataclass
class Stage2Config:
    output_dir: str = "outputs/stage2"
    num_train_epochs: int = 20
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.005
    freeze_feature_extractor: bool = True
    num_transformer_layers_to_unfreeze: int = 4
    use_specaugment: bool = True
    mask_time_prob: float = 0.05
    mask_feature_prob: float = 0.004
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    logging_steps: int = 100
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "wer"
    greater_is_better: bool = False
    early_stopping_patience: int = 3


@dataclass
class TrainingConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    stage1: Stage1Config = field(default_factory=Stage1Config)
    stage2: Stage2Config = field(default_factory=Stage2Config)
    seed: int = 42
    fp16: bool = True
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    output_dir: Path = Path("outputs")
    vocab_dir: Path = Path("vocab")

    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.vocab_dir.mkdir(parents=True, exist_ok=True)


def get_config() -> TrainingConfig:
    return TrainingConfig()


if __name__ == "__main__":
    config = get_config()
    print(f"Model: {config.model.model_name}")
    print(f"Stage 1 - Batch size: {config.stage1.per_device_train_batch_size}, "
          f"Grad accum: {config.stage1.gradient_accumulation_steps}, "
          f"LR: {config.stage1.learning_rate}")
    print(f"Stage 2 - Batch size: {config.stage2.per_device_train_batch_size}, "
          f"Grad accum: {config.stage2.gradient_accumulation_steps}, "
          f"LR: {config.stage2.learning_rate}")
