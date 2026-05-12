import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import List, Optional

import torch


SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def read_text_files(input_files: List[str], encoding: str = "utf-8") -> str:
    parts: List[str] = []
    for file_str in input_files:
        path = Path(file_str)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        if not path.is_file():
            raise ValueError(f"Input path is not a file: {path}")
        parts.append(path.read_text(encoding=encoding, errors="ignore"))

    text = "\n".join(parts)
    if not text.strip():
        raise ValueError("Input text is empty after reading files")
    return text


def read_hf_dataset(
    dataset_name: str,
    split: str,
    text_column: str,
    max_examples: Optional[int] = None,
) -> str:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required for --hf-dataset mode. Install it with `pip install datasets`."
        ) from exc

    dataset = load_dataset(dataset_name, split=split)
    if text_column not in dataset.column_names:
        available = ", ".join(dataset.column_names)
        raise ValueError(f"Text column '{text_column}' not found. Available columns: {available}")

    texts: List[str] = []
    limit = len(dataset) if max_examples is None else min(max_examples, len(dataset))
    for index in range(limit):
        item = dataset[index]
        text = item[text_column]
        if text is not None:
            texts.append(str(text))

    combined = "\n".join(texts)
    if not combined.strip():
        raise ValueError("HF dataset produced empty text after reading examples")
    return combined


def tokenize_text(text: str, lowercase: bool) -> List[str]:
    if lowercase:
        text = text.lower()
    return TOKEN_PATTERN.findall(text)


def build_vocab(tokens: List[str], vocab_size: int) -> dict:
    if vocab_size < len(SPECIAL_TOKENS):
        raise ValueError(f"vocab_size must be at least {len(SPECIAL_TOKENS)}")

    counter = Counter(tokens)
    num_base_tokens = vocab_size - len(SPECIAL_TOKENS)
    most_common = [tok for tok, _ in counter.most_common(num_base_tokens)]

    vocab = list(SPECIAL_TOKENS)
    for tok in most_common:
        if tok not in vocab:
            vocab.append(tok)

    return {tok: idx for idx, tok in enumerate(vocab)}


def encode_tokens(tokens: List[str], stoi: dict, add_bos_eos: bool) -> torch.Tensor:
    unk_id = stoi["<unk>"]
    bos_id = stoi["<bos>"]
    eos_id = stoi["<eos>"]

    ids = [stoi.get(tok, unk_id) for tok in tokens]
    if add_bos_eos:
        ids = [bos_id] + ids + [eos_id]

    return torch.tensor(ids, dtype=torch.long)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate token IDs from text or HF dataset and save tokens.pt + vocab.json")

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--input-files", nargs="+", help="One or more UTF-8 text files")
    source_group.add_argument("--hf-dataset", type=str, help="Hugging Face dataset name, e.g. roneneldan/TinyStories")

    parser.add_argument("--output-dir", default="data", help="Directory to write outputs")
    parser.add_argument("--vocab-size", type=int, default=16000, help="Maximum vocabulary size including special tokens")
    parser.add_argument("--lowercase", action="store_true", help="Lowercase text before tokenization")
    parser.add_argument("--add-bos-eos", action="store_true", help="Add <bos>/<eos> around the full corpus")

    parser.add_argument("--hf-split", type=str, default="train", help="Split name when using --hf-dataset")
    parser.add_argument("--hf-text-column", type=str, default="text", help="Text column name for the HF dataset")
    parser.add_argument("--hf-max-examples", type=int, default=None, help="Limit number of HF examples to read")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.input_files is not None:
        text = read_text_files(args.input_files)
    else:
        text = read_hf_dataset(
            dataset_name=args.hf_dataset,
            split=args.hf_split,
            text_column=args.hf_text_column,
            max_examples=args.hf_max_examples,
        )

    raw_tokens = tokenize_text(text, lowercase=args.lowercase)
    if len(raw_tokens) < 2:
        raise ValueError("Too few tokens extracted from input text")

    stoi = build_vocab(raw_tokens, vocab_size=args.vocab_size)
    tokens_tensor = encode_tokens(raw_tokens, stoi, add_bos_eos=args.add_bos_eos)

    token_file = output_dir / "tokens.pt"
    vocab_file = output_dir / "vocab.json"
    stats_file = output_dir / "token_stats.json"

    torch.save({"tokens": tokens_tensor}, token_file)

    itos = [None] * len(stoi)
    for tok, idx in stoi.items():
        itos[idx] = tok

    vocab_payload = {
        "stoi": stoi,
        "itos": itos,
        "special_tokens": SPECIAL_TOKENS,
    }
    vocab_file.write_text(json.dumps(vocab_payload, ensure_ascii=True, indent=2), encoding="utf-8")

    stats_payload = {
        "num_tokens": int(tokens_tensor.numel()),
        "vocab_size_actual": int(len(stoi)),
        "vocab_size_requested": int(args.vocab_size),
        "lowercase": bool(args.lowercase),
        "add_bos_eos": bool(args.add_bos_eos),
        "input_files": args.input_files,
        "hf_dataset": args.hf_dataset,
        "hf_split": args.hf_split,
        "hf_text_column": args.hf_text_column,
        "hf_max_examples": args.hf_max_examples,
    }
    stats_file.write_text(json.dumps(stats_payload, ensure_ascii=True, indent=2), encoding="utf-8")

    print("Prepared tokenizer artifacts:")
    print(f"  tokens: {token_file}")
    print(f"  vocab: {vocab_file}")
    print(f"  stats: {stats_file}")
    print()
    print(f"Token count: {tokens_tensor.numel()}")
    print(f"Vocab size (actual): {len(stoi)}")


if __name__ == "__main__":
    main()
