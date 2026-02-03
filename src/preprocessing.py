import re
import string
from typing import Tuple, Optional

import torch
import torchaudio
import soundfile as sf
import numpy as np


class AudioPreprocessor:
    
    def __init__(self, target_sample_rate: int = 16000):
        self.target_sample_rate = target_sample_rate
        self._resamplers = {}
    
    def _get_resampler(self, orig_freq: int) -> torchaudio.transforms.Resample:
        if orig_freq not in self._resamplers:
            self._resamplers[orig_freq] = torchaudio.transforms.Resample(
                orig_freq=orig_freq,
                new_freq=self.target_sample_rate
            )
        return self._resamplers[orig_freq]
    
    def load_audio(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        audio_array, sample_rate = sf.read(audio_path, dtype='float32')
        waveform = torch.from_numpy(audio_array)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.T
        return waveform, sample_rate
    
    def resample(self, waveform: torch.Tensor, orig_sample_rate: int) -> torch.Tensor:
        if orig_sample_rate != self.target_sample_rate:
            resampler = self._get_resampler(orig_sample_rate)
            waveform = resampler(waveform)
        return waveform
    
    def to_mono(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform
    
    def normalize(self, waveform: torch.Tensor) -> torch.Tensor:
        max_val = torch.abs(waveform).max()
        if max_val > 0:
            waveform = waveform / max_val
        return waveform
    
    def process(
        self, 
        audio_path: str,
        max_length_seconds: Optional[float] = None,
        min_length_seconds: Optional[float] = None
    ) -> Optional[np.ndarray]:
        waveform, sample_rate = self.load_audio(audio_path)
        waveform = self.to_mono(waveform)
        waveform = self.resample(waveform, sample_rate)
        
        if min_length_seconds is not None:
            min_samples = int(min_length_seconds * self.target_sample_rate)
            if waveform.shape[1] < min_samples:
                return None
        
        if max_length_seconds is not None:
            max_samples = int(max_length_seconds * self.target_sample_rate)
            if waveform.shape[1] > max_samples:
                waveform = waveform[:, :max_samples]
        
        waveform = self.normalize(waveform)
        audio_array = waveform.squeeze(0).numpy()
        return audio_array


class TextPreprocessor:
    
    def __init__(self):
        self.punctuation_pattern = re.compile(f"[{re.escape(string.punctuation)}]")
        self.whitespace_pattern = re.compile(r'\s+')
    
    def lowercase(self, text: str) -> str:
        return text.lower()
    
    def remove_punctuation(self, text: str) -> str:
        return self.punctuation_pattern.sub('', text)
    
    def collapse_whitespace(self, text: str) -> str:
        return self.whitespace_pattern.sub(' ', text).strip()
    
    def process(self, text: str) -> str:
        text = self.lowercase(text)
        text = self.remove_punctuation(text)
        text = self.collapse_whitespace(text)
        return text


def preprocess_audio(
    audio_path: str,
    target_sample_rate: int = 16000,
    max_length_seconds: Optional[float] = None,
    min_length_seconds: Optional[float] = None
) -> Optional[np.ndarray]:
    preprocessor = AudioPreprocessor(target_sample_rate)
    return preprocessor.process(audio_path, max_length_seconds, min_length_seconds)


def preprocess_text(text: str) -> str:
    preprocessor = TextPreprocessor()
    return preprocessor.process(text)


if __name__ == "__main__":
    test_texts = [
        "DON'T WORRY, BE HAPPY!",
        "IT'S A BEAUTIFUL DAY.",
        "HELLO   WORLD",
        "MR. SMITH'S CAR",
    ]
    
    print("Text Preprocessing Examples:")
    print("-" * 50)
    text_proc = TextPreprocessor()
    for text in test_texts:
        processed = text_proc.process(text)
        print(f"Original: {text}")
        print(f"Processed: {processed}")
        print()
