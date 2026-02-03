import argparse
import sys
from pathlib import Path

import torch
import numpy as np
import sounddevice as sd
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

SAMPLE_RATE = 16000


class SpeechToText:
    
    def __init__(self, model_path: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model from: {model_path}")
        print(f"Using device: {self.device}")
        
        self.model = Wav2Vec2ForCTC.from_pretrained(model_path)
        self.processor = Wav2Vec2Processor.from_pretrained(model_path)
        
        self.model.to(self.device)
        self.model.eval()
        
        print("Model loaded successfully!\n")
    
    def transcribe(self, audio: np.ndarray) -> str:
        if audio.max() > 1.0 or audio.min() < -1.0:
            audio = audio / max(abs(audio.max()), abs(audio.min()))
        
        inputs = self.processor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True
        )
        
        input_values = inputs.input_values.to(self.device)
        
        with torch.no_grad():
            logits = self.model(input_values).logits
        
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        
        return transcription


def record_audio(duration: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    print(f"Recording for {duration} seconds... Speak now!")
    
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype='float32'
    )
    sd.wait()
    
    print("Recording complete!")
    
    return audio.flatten()


def continuous_transcription(stt: SpeechToText, chunk_duration: float = 5.0):
    print("\n" + "="*60)
    print("CONTINUOUS TRANSCRIPTION MODE")
    print("="*60)
    print(f"Recording in {chunk_duration}-second chunks.")
    print("Press Ctrl+C to stop.\n")
    
    try:
        while True:
            audio = record_audio(chunk_duration)
            
            if np.abs(audio).max() < 0.01:
                print("[Silence detected, skipping...]\n")
                continue
            
            text = stt.transcribe(audio)
            
            if text.strip():
                print(f">> {text}\n")
            else:
                print("[No speech detected]\n")
                
    except KeyboardInterrupt:
        print("\n\nStopped by user.")


def single_recording(stt: SpeechToText, duration: float = 5.0):
    print("\n" + "="*60)
    print("SINGLE RECORDING MODE")
    print("="*60)
    
    audio = record_audio(duration)
    
    print("\nTranscribing...")
    text = stt.transcribe(audio)
    
    print("\n" + "-"*60)
    print("TRANSCRIPTION:")
    print("-"*60)
    print(f"{text}")
    print("-"*60)


def transcribe_file(stt: SpeechToText, audio_path: str):
    import soundfile as sf
    
    print(f"\nLoading audio file: {audio_path}")
    
    audio, sr = sf.read(audio_path, dtype='float32')
    
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    
    if sr != SAMPLE_RATE:
        import torchaudio
        audio_tensor = torch.from_numpy(audio).unsqueeze(0)
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        audio = resampler(audio_tensor).squeeze().numpy()
        print(f"Resampled from {sr}Hz to {SAMPLE_RATE}Hz")
    
    print("Transcribing...")
    text = stt.transcribe(audio)
    
    print("\n" + "-"*60)
    print("TRANSCRIPTION:")
    print("-"*60)
    print(f"{text}")
    print("-"*60)


def main():
    parser = argparse.ArgumentParser(
        description="Speech-to-Text using Fine-tuned Wav2Vec2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python transcribe.py --model_path outputs/stage2/best_model
  python transcribe.py --model_path outputs/stage2/best_model --duration 10
  python transcribe.py --model_path outputs/stage2/best_model --continuous
  python transcribe.py --model_path outputs/stage2/best_model --file audio.wav
        """
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        default="outputs/stage2/best_model",
        help="Path to fine-tuned model (default: outputs/stage2/best_model)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Recording duration in seconds (default: 5.0)"
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Enable continuous transcription mode"
    )
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to audio file to transcribe (instead of recording)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, default: auto-detect)"
    )
    
    args = parser.parse_args()
    
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Make sure you've trained the model first.")
        sys.exit(1)
    
    stt = SpeechToText(str(model_path), device=args.device)
    
    if args.file:
        transcribe_file(stt, args.file)
    elif args.continuous:
        continuous_transcription(stt, chunk_duration=args.duration)
    else:
        single_recording(stt, duration=args.duration)


if __name__ == "__main__":
    main()
