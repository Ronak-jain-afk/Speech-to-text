import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import numpy as np
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from transformers.trainer_utils import get_last_checkpoint
import jiwer

from config import get_config, TrainingConfig
from dataset import LibriSpeechDataset, DataCollatorCTCWithPadding, create_datasets
from vocab import build_vocab, load_vocab


def setup_processor(config: TrainingConfig) -> Wav2Vec2Processor:
    vocab_path = config.vocab_dir / config.model.vocab_file
    
    if not vocab_path.exists():
        print("Building vocabulary from train-clean-100...")
        build_vocab(
            data_dir=config.data.data_dir,
            output_path=vocab_path,
            subset=config.data.train_subset,
        )
    
    tokenizer = Wav2Vec2CTCTokenizer(
        str(vocab_path),
        unk_token="[UNK]",
        pad_token="[PAD]",
        word_delimiter_token="|",
    )
    
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1,
        sampling_rate=config.data.sample_rate,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True,
    )
    
    processor = Wav2Vec2Processor(
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
    )
    
    return processor


def setup_model(
    config: TrainingConfig,
    processor: Wav2Vec2Processor,
    stage: int = 1,
) -> Wav2Vec2ForCTC:
    stage_config = config.stage1 if stage == 1 else config.stage2
    
    model = Wav2Vec2ForCTC.from_pretrained(
        config.model.model_name,
        ctc_loss_reduction=config.model.ctc_loss_reduction,
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
        ctc_zero_infinity=config.model.ctc_zero_infinity,
    )
    
    if stage_config.use_specaugment:
        model.config.mask_time_prob = config.stage2.mask_time_prob
        model.config.mask_feature_prob = config.stage2.mask_feature_prob
        model.config.mask_time_length = config.model.mask_time_length
        model.config.mask_feature_length = config.model.mask_feature_length
        print(f"SpecAugment enabled: mask_time_prob={model.config.mask_time_prob}, "
              f"mask_feature_prob={model.config.mask_feature_prob}")
    else:
        model.config.mask_time_prob = 0.0
        model.config.mask_feature_prob = 0.0
        print("SpecAugment disabled")
    
    if stage_config.freeze_feature_extractor:
        model.freeze_feature_encoder()
        print("Feature extractor frozen")
    
    if stage == 1:
        pass
    
    elif stage == 2:
        num_layers_to_unfreeze = config.stage2.num_transformer_layers_to_unfreeze
        total_layers = len(model.wav2vec2.encoder.layers)
        
        for param in model.wav2vec2.encoder.parameters():
            param.requires_grad = False
        
        layers_to_unfreeze = list(range(total_layers - num_layers_to_unfreeze, total_layers))
        for layer_idx in layers_to_unfreeze:
            for param in model.wav2vec2.encoder.layers[layer_idx].parameters():
                param.requires_grad = True
        
        print(f"Unfroze transformer layers: {layers_to_unfreeze} (out of {total_layers})")
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)")
    
    return model


def compute_metrics(pred, processor: Wav2Vec2Processor) -> Dict[str, float]:
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)
    
    valid_pairs = [(p, r) for p, r in zip(pred_str, label_str) if r.strip()]
    if not valid_pairs:
        return {"wer": 1.0}
    
    pred_str_filtered, label_str_filtered = zip(*valid_pairs)
    
    wer = jiwer.wer(list(label_str_filtered), list(pred_str_filtered))
    
    return {"wer": wer}


def get_training_args(config: TrainingConfig, stage: int) -> TrainingArguments:
    stage_config = config.stage1 if stage == 1 else config.stage2
    
    return TrainingArguments(
        output_dir=stage_config.output_dir,
        num_train_epochs=stage_config.num_train_epochs,
        per_device_train_batch_size=stage_config.per_device_train_batch_size,
        per_device_eval_batch_size=stage_config.per_device_eval_batch_size,
        gradient_accumulation_steps=stage_config.gradient_accumulation_steps,
        learning_rate=stage_config.learning_rate,
        warmup_ratio=stage_config.warmup_ratio,
        weight_decay=stage_config.weight_decay,
        eval_strategy=stage_config.evaluation_strategy,
        save_strategy=stage_config.save_strategy,
        logging_steps=stage_config.logging_steps,
        save_total_limit=stage_config.save_total_limit,
        load_best_model_at_end=stage_config.load_best_model_at_end,
        metric_for_best_model=stage_config.metric_for_best_model,
        greater_is_better=stage_config.greater_is_better,
        fp16=config.fp16,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=config.dataloader_pin_memory,
        seed=config.seed,
        logging_dir=f"{stage_config.output_dir}/logs",
        report_to=["tensorboard"],
        push_to_hub=False,
        remove_unused_columns=False,
    )


def train_stage(
    config: TrainingConfig,
    stage: int,
    processor: Wav2Vec2Processor,
    train_dataset: LibriSpeechDataset,
    eval_dataset: LibriSpeechDataset,
    resume_from_checkpoint: Optional[str] = None,
) -> str:
    print(f"\n{'='*60}")
    print(f"STAGE {stage} TRAINING")
    print(f"{'='*60}\n")
    
    if stage == 2 and resume_from_checkpoint:
        print(f"Loading model from Stage 1: {resume_from_checkpoint}")
        model = Wav2Vec2ForCTC.from_pretrained(resume_from_checkpoint)
        
        if not hasattr(model.wav2vec2, 'masked_spec_embed') or model.wav2vec2.masked_spec_embed is None:
            hidden_size = model.config.hidden_size
            model.wav2vec2.masked_spec_embed = torch.nn.Parameter(
                torch.FloatTensor(hidden_size).uniform_()
            )
            print("Reinitialized masked_spec_embed for SpecAugment")
        
        model = setup_model_freezing(config, model, stage=2)
    else:
        model = setup_model(config, processor, stage=stage)
    
    data_collator = DataCollatorCTCWithPadding(processor=processor)
    
    training_args = get_training_args(config, stage)
    
    callbacks = []
    if stage == 2:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=config.stage2.early_stopping_patience
            )
        )
    
    def compute_metrics_fn(pred):
        return compute_metrics(pred, processor)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
        callbacks=callbacks,
    )
    
    stage_config = config.stage1 if stage == 1 else config.stage2
    last_checkpoint = get_last_checkpoint(stage_config.output_dir)
    if last_checkpoint:
        print(f"Resuming from checkpoint: {last_checkpoint}")
    
    trainer.train(resume_from_checkpoint=last_checkpoint)
    
    best_model_path = f"{stage_config.output_dir}/best_model"
    trainer.save_model(best_model_path)
    processor.save_pretrained(best_model_path)
    
    print(f"\nStage {stage} completed. Best model saved to: {best_model_path}")
    
    eval_results = trainer.evaluate()
    print(f"Final evaluation results: {eval_results}")
    
    return best_model_path


def setup_model_freezing(
    config: TrainingConfig,
    model: Wav2Vec2ForCTC,
    stage: int,
) -> Wav2Vec2ForCTC:
    stage_config = config.stage1 if stage == 1 else config.stage2
    
    if stage_config.use_specaugment:
        model.config.mask_time_prob = config.stage2.mask_time_prob
        model.config.mask_feature_prob = config.stage2.mask_feature_prob
        print(f"SpecAugment enabled: mask_time_prob={model.config.mask_time_prob}")
    else:
        model.config.mask_time_prob = 0.0
        model.config.mask_feature_prob = 0.0
        print("SpecAugment disabled")
    
    if stage_config.freeze_feature_extractor:
        model.freeze_feature_encoder()
        print("Feature extractor frozen")
    
    if stage == 2:
        num_layers_to_unfreeze = config.stage2.num_transformer_layers_to_unfreeze
        total_layers = len(model.wav2vec2.encoder.layers)
        
        for param in model.wav2vec2.encoder.parameters():
            param.requires_grad = False
        
        layers_to_unfreeze = list(range(total_layers - num_layers_to_unfreeze, total_layers))
        for layer_idx in layers_to_unfreeze:
            for param in model.wav2vec2.encoder.layers[layer_idx].parameters():
                param.requires_grad = True
        
        print(f"Unfroze transformer layers: {layers_to_unfreeze}")
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
    
    return model


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Wav2Vec2 on LibriSpeech")
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2],
        default=None,
        help="Run only specified stage (default: run both)"
    )
    parser.add_argument(
        "--stage1_checkpoint",
        type=str,
        default=None,
        help="Path to Stage 1 checkpoint for Stage 2 training"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Path to data directory"
    )
    
    args = parser.parse_args()
    
    config = get_config()
    
    if args.data_dir:
        config.data.data_dir = Path(args.data_dir)
    
    print("Setting up processor...")
    processor = setup_processor(config)
    
    print("\nLoading datasets...")
    train_dataset, eval_dataset = create_datasets(
        data_dir=config.data.data_dir,
        processor=processor,
        train_subset=config.data.train_subset,
        eval_subset=config.data.eval_subset,
        max_audio_length_seconds=config.data.max_audio_length_seconds,
        min_audio_length_seconds=config.data.min_audio_length_seconds,
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")
    
    if args.stage == 1 or args.stage is None:
        stage1_best_path = train_stage(
            config=config,
            stage=1,
            processor=processor,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
    else:
        stage1_best_path = args.stage1_checkpoint
    
    if args.stage == 2 or args.stage is None:
        if stage1_best_path is None:
            stage1_best_path = f"{config.stage1.output_dir}/best_model"
            if not Path(stage1_best_path).exists():
                raise ValueError(
                    "Stage 1 checkpoint not found. Run Stage 1 first or provide --stage1_checkpoint"
                )
        
        train_stage(
            config=config,
            stage=2,
            processor=processor,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            resume_from_checkpoint=stage1_best_path,
        )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print("="*60)


if __name__ == "__main__":
    main()
