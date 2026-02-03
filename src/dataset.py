import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
import numpy as np

from preprocessing import AudioPreprocessor, TextPreprocessor


@dataclass
class LibriSpeechSample:
    utterance_id: str
    audio_path: Path
    transcript: str
    speaker_id: str
    chapter_id: str


class LibriSpeechDataset(Dataset):
    
    def __init__(
        self,
        data_dir: Path,
        subset: str,
        processor,
        audio_preprocessor: Optional[AudioPreprocessor] = None,
        text_preprocessor: Optional[TextPreprocessor] = None,
        max_audio_length_seconds: float = 20.0,
        min_audio_length_seconds: float = 0.5,
        max_target_length: Optional[int] = None,
    ):
        self.data_dir = Path(data_dir)
        self.subset = subset
        self.processor = processor
        self.max_audio_length_seconds = max_audio_length_seconds
        self.min_audio_length_seconds = min_audio_length_seconds
        self.max_target_length = max_target_length
        
        self.audio_preprocessor = audio_preprocessor or AudioPreprocessor()
        self.text_preprocessor = text_preprocessor or TextPreprocessor()
        
        self.samples = self._load_samples()
        print(f"Loaded {len(self.samples)} samples from {subset}")
    
    def _load_samples(self) -> List[LibriSpeechSample]:
        samples = []
        subset_dir = self.data_dir / self.subset
        
        if not subset_dir.exists():
            raise FileNotFoundError(f"Subset directory not found: {subset_dir}")
        
        transcript_files = list(subset_dir.rglob("*.trans.txt"))
        
        for trans_file in transcript_files:
            transcripts = self._parse_transcript_file(trans_file)
            
            chapter_dir = trans_file.parent
            chapter_id = chapter_dir.name
            speaker_id = chapter_dir.parent.name
            
            for utterance_id, transcript in transcripts.items():
                audio_filename = f"{utterance_id}.flac"
                audio_path = chapter_dir / audio_filename
                
                if audio_path.exists():
                    samples.append(LibriSpeechSample(
                        utterance_id=utterance_id,
                        audio_path=audio_path,
                        transcript=transcript,
                        speaker_id=speaker_id,
                        chapter_id=chapter_id,
                    ))
                else:
                    print(f"Warning: Audio file not found: {audio_path}")
        
        return samples
    
    def _parse_transcript_file(self, transcript_path: Path) -> Dict[str, str]:
        transcripts = {}
        with open(transcript_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(' ', 1)
                    if len(parts) == 2:
                        utterance_id, transcript = parts
                        transcripts[utterance_id] = transcript
        return transcripts
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        
        audio_array = self.audio_preprocessor.process(
            str(sample.audio_path),
            max_length_seconds=self.max_audio_length_seconds,
            min_length_seconds=self.min_audio_length_seconds,
        )
        
        if audio_array is None:
            return self._get_dummy_sample()
        
        processed_transcript = self.text_preprocessor.process(sample.transcript)
        
        inputs = self.processor(
            audio_array,
            sampling_rate=self.audio_preprocessor.target_sample_rate,
            return_tensors="pt",
            padding=False,
        )
        
        labels = self.processor.tokenizer(
            processed_transcript,
            return_tensors="pt",
            padding=False,
        )
        
        input_values = inputs.input_values.squeeze(0)
        label_ids = labels.input_ids.squeeze(0)
        
        if self.max_target_length is not None and len(label_ids) > self.max_target_length:
            label_ids = label_ids[:self.max_target_length]
        
        return {
            "input_values": input_values,
            "labels": label_ids,
            "utterance_id": sample.utterance_id,
        }
    
    def _get_dummy_sample(self) -> Dict[str, Any]:
        return {
            "input_values": torch.zeros(1),
            "labels": torch.zeros(1, dtype=torch.long),
            "utterance_id": "SKIP",
        }


@dataclass
class DataCollatorCTCWithPadding:
    
    processor: Any
    padding: str = "longest"
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        features = [f for f in features if f["utterance_id"] != "SKIP"]
        
        if len(features) == 0:
            raise ValueError("All samples in batch were skipped!")
        
        input_features = [{"input_values": f["input_values"]} for f in features]
        label_features = [{"input_ids": f["labels"]} for f in features]
        
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        
        batch["labels"] = labels
        
        return batch


def create_datasets(
    data_dir: Path,
    processor,
    train_subset: str = "train-clean-100",
    eval_subset: str = "dev-clean",
    max_audio_length_seconds: float = 20.0,
    min_audio_length_seconds: float = 0.5,
) -> Tuple[LibriSpeechDataset, LibriSpeechDataset]:
    audio_preprocessor = AudioPreprocessor()
    text_preprocessor = TextPreprocessor()
    
    train_dataset = LibriSpeechDataset(
        data_dir=data_dir,
        subset=train_subset,
        processor=processor,
        audio_preprocessor=audio_preprocessor,
        text_preprocessor=text_preprocessor,
        max_audio_length_seconds=max_audio_length_seconds,
        min_audio_length_seconds=min_audio_length_seconds,
    )
    
    eval_dataset = LibriSpeechDataset(
        data_dir=data_dir,
        subset=eval_subset,
        processor=processor,
        audio_preprocessor=audio_preprocessor,
        text_preprocessor=text_preprocessor,
        max_audio_length_seconds=max_audio_length_seconds,
        min_audio_length_seconds=min_audio_length_seconds,
    )
    
    return train_dataset, eval_dataset


if __name__ == "__main__":
    from transformers import Wav2Vec2Processor
    
    print("Testing LibriSpeech dataset loading...")
    
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    
    data_dir = Path("data")
    if data_dir.exists():
        train_dataset = LibriSpeechDataset(
            data_dir=data_dir,
            subset="train-clean-100",
            processor=processor,
            max_audio_length_seconds=20.0,
            min_audio_length_seconds=0.5,
        )
        
        print(f"\nDataset size: {len(train_dataset)}")
        
        if len(train_dataset) > 0:
            sample = train_dataset[0]
            print(f"\nSample keys: {sample.keys()}")
            print(f"Input values shape: {sample['input_values'].shape}")
            print(f"Labels shape: {sample['labels'].shape}")
            print(f"Utterance ID: {sample['utterance_id']}")
    else:
        print(f"Data directory not found: {data_dir}")
