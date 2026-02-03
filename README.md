# Speech-to-Text: Wav2Vec2 Fine-tuning on LibriSpeech

Fine-tuned Wav2Vec2-base model for automatic speech recognition (ASR) on LibriSpeech clean-100.

## Results

- **WER (Word Error Rate):** 8.75% on dev-clean
- **Perfect Predictions:** 35% of samples

## Project Structure

```
speech-to-text/
├── src/
│   ├── config.py           # Training hyperparameters
│   ├── preprocessing.py    # Audio/text preprocessing
│   ├── vocab.py            # Character-level vocabulary
│   ├── dataset.py          # PyTorch Dataset for LibriSpeech
│   ├── train.py            # Two-stage training script
│   ├── evaluate.py         # WER evaluation
│   ├── transcribe.py       # CLI transcription
│   └── transcribe_gui.py   # GUI transcription app
├── data/                   # LibriSpeech data (not tracked)
├── outputs/                # Model checkpoints (not tracked)
├── vocab/                  # Vocabulary files (not tracked)
└── requirements.txt
```

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download LibriSpeech data:**
   - Place `train-clean-100` and `dev-clean` in the `data/` folder

## Training

Two-stage training strategy:

**Stage 1:** Frozen feature extractor (2 epochs)
- Batch size: 8, Gradient accumulation: 4
- Learning rate: 1e-4
- No SpecAugment

**Stage 2:** Unfreeze last 4 transformer layers (early stopping)
- Batch size: 4, Gradient accumulation: 8
- Learning rate: 1e-5
- SpecAugment enabled
- Early stopping patience: 3

```bash
cd speech-to-text
python src/train.py
```

## Evaluation

```bash
python src/evaluate.py --model_path outputs/stage2/best_model --data_dir data
```

## Inference

**CLI:**
```bash
python src/transcribe.py --model_path outputs/stage2/best_model
python src/transcribe.py --model_path outputs/stage2/best_model --file audio.wav
python src/transcribe.py --model_path outputs/stage2/best_model --continuous
```

**GUI:**
```bash
python src/transcribe_gui.py
```

## Requirements

- Python 3.10+
- PyTorch 2.0+ with CUDA
- transformers
- torchaudio
- soundfile
- sounddevice
- jiwer

## License

MIT
