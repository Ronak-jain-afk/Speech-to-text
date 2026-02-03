import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import torch
import numpy as np
from tqdm import tqdm
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import jiwer

from config import get_config
from dataset import LibriSpeechDataset
from preprocessing import AudioPreprocessor, TextPreprocessor


@dataclass
class EvaluationResult:
    wer: float
    predictions: List[str]
    references: List[str]
    utterance_ids: List[str]


class GreedyCTCDecoder:
    
    def __init__(self, processor: Wav2Vec2Processor):
        self.processor = processor
        self.blank_token_id = processor.tokenizer.pad_token_id
    
    def decode(self, logits: torch.Tensor) -> str:
        pred_ids = torch.argmax(logits, dim=-1)
        decoded_text = self.processor.decode(pred_ids)
        return decoded_text
    
    def decode_batch(self, logits: torch.Tensor) -> List[str]:
        pred_ids = torch.argmax(logits, dim=-1)
        decoded_texts = self.processor.batch_decode(pred_ids)
        return decoded_texts


def load_model_and_processor(
    model_path: str,
    device: str = "cuda"
) -> Tuple[Wav2Vec2ForCTC, Wav2Vec2Processor]:
    print(f"Loading model from: {model_path}")
    
    model = Wav2Vec2ForCTC.from_pretrained(model_path)
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    
    model = model.to(device)
    model.eval()
    
    return model, processor


def evaluate_model(
    model: Wav2Vec2ForCTC,
    processor: Wav2Vec2Processor,
    eval_dataset: LibriSpeechDataset,
    batch_size: int = 8,
    device: str = "cuda",
    num_samples: Optional[int] = None,
) -> EvaluationResult:
    decoder = GreedyCTCDecoder(processor)
    text_preprocessor = TextPreprocessor()
    
    all_predictions = []
    all_references = []
    all_utterance_ids = []
    
    indices = list(range(len(eval_dataset)))
    if num_samples is not None:
        indices = indices[:num_samples]
    
    print(f"Evaluating on {len(indices)} samples...")
    
    with torch.no_grad():
        for idx in tqdm(indices, desc="Evaluating"):
            sample = eval_dataset[idx]
            
            if sample["utterance_id"] == "SKIP":
                continue
            
            input_values = sample["input_values"].unsqueeze(0).to(device)
            
            outputs = model(input_values)
            logits = outputs.logits
            
            prediction = decoder.decode(logits.squeeze(0))
            
            labels = sample["labels"]
            labels[labels == -100] = processor.tokenizer.pad_token_id
            reference = processor.decode(labels, group_tokens=False)
            
            all_predictions.append(prediction)
            all_references.append(reference)
            all_utterance_ids.append(sample["utterance_id"])
    
    valid_pairs = [(p, r) for p, r in zip(all_predictions, all_references) if r.strip()]
    if valid_pairs:
        preds, refs = zip(*valid_pairs)
        wer = jiwer.wer(list(refs), list(preds))
    else:
        wer = 1.0
    
    return EvaluationResult(
        wer=wer,
        predictions=all_predictions,
        references=all_references,
        utterance_ids=all_utterance_ids,
    )


def print_sample_predictions(
    result: EvaluationResult,
    num_samples: int = 10,
    random_samples: bool = True,
) -> None:
    print("\n" + "="*80)
    print("SAMPLE PREDICTIONS")
    print("="*80)
    
    indices = list(range(len(result.predictions)))
    
    if random_samples:
        np.random.shuffle(indices)
    
    indices = indices[:num_samples]
    
    for i, idx in enumerate(indices):
        print(f"\n--- Sample {i+1} (ID: {result.utterance_ids[idx]}) ---")
        print(f"Reference:  {result.references[idx]}")
        print(f"Prediction: {result.predictions[idx]}")
        
        if result.references[idx].strip():
            sample_wer = jiwer.wer(result.references[idx], result.predictions[idx])
        else:
            sample_wer = 1.0
        print(f"WER: {sample_wer:.4f}")


def analyze_errors(result: EvaluationResult) -> Dict[str, any]:
    sample_wers = []
    for pred, ref in zip(result.predictions, result.references):
        if ref.strip():
            wer = jiwer.wer(ref, pred)
        else:
            wer = 1.0
        sample_wers.append(wer)
    
    sample_wers = np.array(sample_wers)
    
    best_indices = np.argsort(sample_wers)[:5]
    worst_indices = np.argsort(sample_wers)[-5:]
    
    analysis = {
        "mean_wer": np.mean(sample_wers),
        "median_wer": np.median(sample_wers),
        "std_wer": np.std(sample_wers),
        "min_wer": np.min(sample_wers),
        "max_wer": np.max(sample_wers),
        "perfect_predictions": np.sum(sample_wers == 0),
        "best_indices": best_indices.tolist(),
        "worst_indices": worst_indices.tolist(),
    }
    
    print("\n" + "="*80)
    print("ERROR ANALYSIS")
    print("="*80)
    print(f"Mean WER: {analysis['mean_wer']:.4f}")
    print(f"Median WER: {analysis['median_wer']:.4f}")
    print(f"Std WER: {analysis['std_wer']:.4f}")
    print(f"Min WER: {analysis['min_wer']:.4f}")
    print(f"Max WER: {analysis['max_wer']:.4f}")
    print(f"Perfect predictions (WER=0): {analysis['perfect_predictions']} / {len(sample_wers)}")
    
    print("\n--- Best Predictions ---")
    for idx in best_indices:
        print(f"ID: {result.utterance_ids[idx]}, WER: {sample_wers[idx]:.4f}")
        print(f"  Ref: {result.references[idx]}")
        print(f"  Pred: {result.predictions[idx]}")
    
    print("\n--- Worst Predictions ---")
    for idx in worst_indices:
        print(f"ID: {result.utterance_ids[idx]}, WER: {sample_wers[idx]:.4f}")
        print(f"  Ref: {result.references[idx]}")
        print(f"  Pred: {result.predictions[idx]}")
    
    return analysis


def main():
    parser = argparse.ArgumentParser(description="Evaluate Wav2Vec2 ASR model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Path to data directory"
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="dev-clean",
        help="Dataset subset to evaluate on"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all)"
    )
    parser.add_argument(
        "--print_samples",
        type=int,
        default=10,
        help="Number of sample predictions to print"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on"
    )
    
    args = parser.parse_args()
    
    model, processor = load_model_and_processor(args.model_path, args.device)
    
    config = get_config()
    
    eval_dataset = LibriSpeechDataset(
        data_dir=Path(args.data_dir),
        subset=args.subset,
        processor=processor,
        max_audio_length_seconds=config.data.max_audio_length_seconds,
        min_audio_length_seconds=config.data.min_audio_length_seconds,
    )
    
    print(f"Evaluation dataset size: {len(eval_dataset)}")
    
    result = evaluate_model(
        model=model,
        processor=processor,
        eval_dataset=eval_dataset,
        batch_size=args.batch_size,
        device=args.device,
        num_samples=args.num_samples,
    )
    
    print("\n" + "="*80)
    print("EVALUATION RESULTS")
    print("="*80)
    print(f"Word Error Rate (WER): {result.wer:.4f} ({result.wer * 100:.2f}%)")
    print(f"Total samples evaluated: {len(result.predictions)}")
    
    if args.print_samples > 0:
        print_sample_predictions(result, num_samples=args.print_samples)
    
    analyze_errors(result)


if __name__ == "__main__":
    main()
