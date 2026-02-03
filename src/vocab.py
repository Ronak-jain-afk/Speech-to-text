import json
from pathlib import Path
from typing import Set, Dict, List
from collections import Counter

from preprocessing import TextPreprocessor


def find_transcript_files(data_dir: Path, subset: str = "train-clean-100") -> List[Path]:
    subset_dir = data_dir / subset
    if not subset_dir.exists():
        raise FileNotFoundError(f"Subset directory not found: {subset_dir}")
    
    transcript_files = list(subset_dir.rglob("*.trans.txt"))
    print(f"Found {len(transcript_files)} transcript files in {subset}")
    return transcript_files


def parse_transcript_file(transcript_path: Path) -> Dict[str, str]:
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


def extract_characters(
    data_dir: Path,
    subset: str = "train-clean-100",
    text_preprocessor: TextPreprocessor = None
) -> Set[str]:
    if text_preprocessor is None:
        text_preprocessor = TextPreprocessor()
    
    characters = set()
    char_counts = Counter()
    
    transcript_files = find_transcript_files(data_dir, subset)
    
    for trans_file in transcript_files:
        transcripts = parse_transcript_file(trans_file)
        for transcript in transcripts.values():
            processed = text_preprocessor.process(transcript)
            for char in processed:
                char_counts[char] += 1
                characters.add(char)
    
    print(f"\nCharacter Statistics:")
    print(f"Total unique characters: {len(characters)}")
    print(f"\nTop 30 characters by frequency:")
    for char, count in char_counts.most_common(30):
        display_char = repr(char) if char == ' ' else char
        print(f"  {display_char}: {count:,}")
    
    return characters


def build_vocab(
    data_dir: Path,
    output_path: Path,
    subset: str = "train-clean-100"
) -> Dict[str, int]:
    text_preprocessor = TextPreprocessor()
    characters = extract_characters(data_dir, subset, text_preprocessor)
    characters.discard(' ')
    sorted_chars = sorted(list(characters))
    
    vocab = {
        "[PAD]": 0,
        "[UNK]": 1,
        "|": 2,
    }
    
    for idx, char in enumerate(sorted_chars, start=3):
        vocab[char] = idx
    
    print(f"\nFinal Vocabulary:")
    print(f"Total vocab size: {len(vocab)}")
    print(f"Special tokens: [PAD], [UNK], |")
    print(f"Character tokens: {sorted_chars}")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    
    print(f"\nVocabulary saved to: {output_path}")
    
    return vocab


def load_vocab(vocab_path: Path) -> Dict[str, int]:
    with open(vocab_path, 'r', encoding='utf-8') as f:
        return json.load(f)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build vocabulary from LibriSpeech")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Path to data directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="vocab/vocab.json",
        help="Output path for vocab.json"
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="train-clean-100",
        help="Subset to build vocabulary from"
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_path = Path(args.output)
    
    vocab = build_vocab(data_dir, output_path, args.subset)
