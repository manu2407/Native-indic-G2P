#!/usr/bin/env python3
"""Small character Transformer baseline for full structured Hindi G2P."""

from __future__ import annotations

import argparse
import json
import math
import random
import tempfile
import time
from pathlib import Path

import torch
from torch import nn

from g2p_metrics import decode_target, evaluate
from native_g2p import label_word
from neural_data import load_records, load_training, target, verify_manifest


SPECIALS = ("<pad>", "<bos>", "<eos>", "<unk>")
PAD, BOS, EOS, UNK = range(4)
INPUT_SEPARATOR = "␞"


class Vocabulary:
    def __init__(self, symbols: list[str]):
        self.symbols = symbols
        self.indices = {symbol: index for index, symbol in enumerate(symbols)}

    @classmethod
    def build(cls, values: list[str]) -> "Vocabulary":
        return cls([*SPECIALS, *sorted({char for value in values for char in value})])

    def encode(self, value: str) -> list[int]:
        return [self.indices.get(char, UNK) for char in value]

    def decode(self, values: list[int]) -> str:
        result = []
        for value in values:
            if value == EOS:
                break
            if value >= len(self.symbols) or value < len(SPECIALS):
                if value == UNK:
                    result.append("�")
                continue
            result.append(self.symbols[value])
        return "".join(result)


class CharacterTransformer(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        d_model: int = 128,
        heads: int = 4,
        layers: int = 2,
        feedforward: int = 384,
        dropout: float = 0.1,
        max_length: int = 192,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_length = max_length
        self.input_embedding = nn.Embedding(input_size, d_model, padding_idx=PAD)
        self.output_embedding = nn.Embedding(output_size, d_model, padding_idx=PAD)
        self.position = nn.Embedding(max_length, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, heads, feedforward, dropout, batch_first=True, norm_first=True
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, heads, feedforward, dropout, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, layers, enable_nested_tensor=False)
        self.decoder = nn.TransformerDecoder(decoder_layer, layers)
        self.output = nn.Linear(d_model, output_size)

    def _embed(self, values: torch.Tensor, embedding: nn.Embedding) -> torch.Tensor:
        positions = torch.arange(values.shape[1], device=values.device).unsqueeze(0)
        return embedding(values) * math.sqrt(self.d_model) + self.position(positions)

    def encode(self, source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        padding = source.eq(PAD)
        return self.encoder(self._embed(source, self.input_embedding), src_key_padding_mask=padding), padding

    def decode(self, target_values: torch.Tensor, memory: torch.Tensor, source_padding: torch.Tensor) -> torch.Tensor:
        length = target_values.shape[1]
        causal = torch.triu(torch.ones(length, length, dtype=torch.bool, device=target_values.device), diagonal=1)
        decoded = self.decoder(
            self._embed(target_values, self.output_embedding),
            memory,
            tgt_mask=causal,
            tgt_key_padding_mask=target_values.eq(PAD),
            memory_key_padding_mask=source_padding,
        )
        return self.output(decoded)

    def forward(self, source: torch.Tensor, target_values: torch.Tensor) -> torch.Tensor:
        memory, source_padding = self.encode(source)
        return self.decode(target_values, memory, source_padding)


def padded(sequences: list[list[int]], device: torch.device) -> torch.Tensor:
    result = torch.full((len(sequences), max(map(len, sequences))), PAD, dtype=torch.long, device=device)
    for index, sequence in enumerate(sequences):
        result[index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
    return result


def make_batches(records: list[dict], batch_size: int, seed: int | None = None) -> list[list[dict]]:
    ordered = sorted(records, key=lambda record: max(len(record["_model_input"]), len(target(record))))
    batches = [ordered[index : index + batch_size] for index in range(0, len(ordered), batch_size)]
    if seed is not None:
        random.Random(seed).shuffle(batches)
    return batches


def train_epoch(
    model: CharacterTransformer,
    records: list[dict],
    input_vocab: Vocabulary,
    output_vocab: Vocabulary,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    batch_size: int,
    epoch: int,
) -> float:
    model.train()
    loss_function = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.05)
    total_loss = 0.0
    total_tokens = 0
    for batch in make_batches(records, batch_size, seed=epoch):
        source = padded([input_vocab.encode(record["_model_input"]) + [EOS] for record in batch], device)
        encoded_targets = [output_vocab.encode(target(record)) for record in batch]
        decoder_input = padded([[BOS, *values] for values in encoded_targets], device)
        labels = padded([[*values, EOS] for values in encoded_targets], device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(source, decoder_input)
            loss = loss_function(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        tokens = labels.ne(PAD).sum().item()
        total_loss += loss.item() * tokens
        total_tokens += tokens
    return total_loss / total_tokens


@torch.inference_mode()
def predict_records(
    model: CharacterTransformer,
    records: list[dict],
    input_vocab: Vocabulary,
    output_vocab: Vocabulary,
    device: torch.device,
    batch_size: int,
) -> dict[str, str]:
    model.eval()
    predictions = {}
    for batch in make_batches(records, batch_size):
        source = padded([input_vocab.encode(record["_model_input"]) + [EOS] for record in batch], device)
        memory, source_padding = model.encode(source)
        generated = torch.full((len(batch), 1), BOS, dtype=torch.long, device=device)
        finished = torch.zeros(len(batch), dtype=torch.bool, device=device)
        for _ in range(model.max_length - 1):
            logits = model.decode(generated, memory, source_padding)
            next_values = logits[:, -1].argmax(-1)
            generated = torch.cat((generated, next_values.unsqueeze(1)), dim=1)
            finished |= next_values.eq(EOS)
            if finished.all():
                break
        for record, values in zip(batch, generated[:, 1:].tolist()):
            predictions[record["text"]] = output_vocab.decode(values)
    return predictions


def structured_report(records: list[dict], predictions: dict[str, str]) -> dict:
    return evaluate(records, lambda word: decode_target(predictions[word]))


def model_config(arguments: argparse.Namespace, input_vocab: Vocabulary, output_vocab: Vocabulary) -> dict:
    return {
        "input_size": len(input_vocab.symbols),
        "output_size": len(output_vocab.symbols),
        "d_model": arguments.d_model,
        "heads": arguments.heads,
        "layers": arguments.layers,
        "feedforward": arguments.feedforward,
        "dropout": arguments.dropout,
        "max_length": arguments.max_length,
    }


def prepare_inputs(records: list[dict], mode: str) -> None:
    for record in records:
        if mode == "word":
            record["_model_input"] = record["text"]
            continue
        try:
            candidate = target(label_word(record["text"]))
        except ValueError:
            candidate = "?"
        record["_model_input"] = record["text"] + INPUT_SEPARATOR + candidate


def save_artifact(path: Path, artifact: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    torch.save(artifact, temporary_path)
    temporary_path.replace(path)


def train(arguments: argparse.Namespace) -> None:
    manifest = verify_manifest(arguments.manifest)
    root = arguments.manifest.resolve().parents[3]
    training_paths = [root / source["path"] for source in manifest["training_sources"]]
    records = load_training(training_paths)
    dev = load_records(root / manifest["dev"]["path"])
    device = choose_device(arguments.device)
    torch.manual_seed(arguments.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(arguments.seed)
    start_epoch = 0
    best_score = -1.0
    best_artifact = None
    if arguments.resume:
        model, input_vocab, output_vocab, best_artifact = load_artifact(arguments.resume, device)
        config = best_artifact["config"]
        arguments.input_mode = best_artifact["training"].get("input_mode", "word")
        start_epoch = best_artifact["training"]["best_epoch"]
        best_score = best_artifact["training"]["best_dev"]["strata"]["all"]["rates_percent"]["all"]
    else:
        prepare_inputs(records, arguments.input_mode)
        input_vocab = Vocabulary.build([record["_model_input"] for record in records])
        output_vocab = Vocabulary.build([target(record) for record in records])
        config = model_config(arguments, input_vocab, output_vocab)
        model = CharacterTransformer(**config).to(device)
    prepare_inputs(records, arguments.input_mode)
    prepare_inputs(dev, arguments.input_mode)
    if Vocabulary.build([record["_model_input"] for record in records]).symbols != input_vocab.symbols:
        raise ValueError("resume input vocabulary does not match training data")
    if Vocabulary.build([target(record) for record in records]).symbols != output_vocab.symbols:
        raise ValueError("resume output vocabulary does not match training data")
    if max(max(len(record["_model_input"]), len(target(record)) + 1) for record in records) > config["max_length"]:
        raise ValueError("max_length is smaller than a training sequence")
    if arguments.epochs <= start_epoch:
        raise ValueError("epochs must be greater than the resumed best epoch")
    optimizer = torch.optim.AdamW(model.parameters(), lr=arguments.learning_rate, weight_decay=0.01)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(json.dumps({"device": str(device), "parameters": parameters, "training": len(records), "dev": len(dev)}), flush=True)

    stale = 0
    started = time.perf_counter()
    for epoch in range(start_epoch + 1, arguments.epochs + 1):
        loss = train_epoch(model, records, input_vocab, output_vocab, optimizer, scaler, device, arguments.batch_size, epoch)
        predictions = predict_records(model, dev, input_vocab, output_vocab, device, arguments.eval_batch_size)
        report = structured_report(dev, predictions)
        score = report["strata"]["all"]["rates_percent"]["all"]
        summary = {
            "epoch": epoch,
            "loss": round(loss, 6),
            "dev": report["strata"]["all"]["rates_percent"],
            "minutes": round((time.perf_counter() - started) / 60, 2),
        }
        print(json.dumps(summary), flush=True)
        if score > best_score:
            best_score = score
            stale = 0
            best_artifact = {
                "format": "hindi-neural-g2p-v1",
                "config": config,
                "input_symbols": input_vocab.symbols,
                "output_symbols": output_vocab.symbols,
                "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
                "training": {
                    "manifest": str(arguments.manifest),
                    "seed": arguments.seed,
                    "best_epoch": epoch,
                    "best_dev": report,
                    "optimizer": "AdamW",
                    "learning_rate": arguments.learning_rate,
                    "input_mode": arguments.input_mode,
                },
            }
            save_artifact(arguments.output, best_artifact)
        else:
            stale += 1
            if stale >= arguments.patience:
                break
    if best_artifact is None:
        raise RuntimeError("training did not produce a model")
    print(json.dumps({"saved": str(arguments.output), "best_dev_all_percent": best_score}), flush=True)


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def load_artifact(path: Path, device: torch.device) -> tuple[CharacterTransformer, Vocabulary, Vocabulary, dict]:
    artifact = torch.load(path, map_location=device, weights_only=True)
    if artifact.get("format") != "hindi-neural-g2p-v1":
        raise ValueError("unsupported neural G2P artifact")
    model = CharacterTransformer(**artifact["config"]).to(device)
    model.load_state_dict(artifact["state_dict"])
    return model, Vocabulary(artifact["input_symbols"]), Vocabulary(artifact["output_symbols"]), artifact


def evaluate_artifact(arguments: argparse.Namespace) -> None:
    manifest = verify_manifest(arguments.manifest)
    root = arguments.manifest.resolve().parents[3]
    records = load_records(root / manifest[arguments.split]["path"])
    device = choose_device(arguments.device)
    model, input_vocab, output_vocab, artifact = load_artifact(arguments.model, device)
    prepare_inputs(records, artifact["training"].get("input_mode", "word"))
    predictions = predict_records(model, records, input_vocab, output_vocab, device, arguments.eval_batch_size)
    report = structured_report(records, predictions)
    print(json.dumps({"model_training": artifact["training"], "evaluation": report}, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--manifest", type=Path, default=Path("datasets/validation/neural_g2p_eval_v1/manifest.json"))
    shared.add_argument("--device", default="auto")
    shared.add_argument("--eval-batch-size", type=int, default=128)
    training = subparsers.add_parser("train", parents=[shared])
    training.add_argument("--output", type=Path, default=Path("models/neural_g2p_small_v1.pt"))
    training.add_argument("--resume", type=Path)
    training.add_argument("--epochs", type=int, default=10)
    training.add_argument("--patience", type=int, default=3)
    training.add_argument("--batch-size", type=int, default=96)
    training.add_argument("--learning-rate", type=float, default=3e-4)
    training.add_argument("--seed", type=int, default=23)
    training.add_argument("--d-model", type=int, default=128)
    training.add_argument("--heads", type=int, default=4)
    training.add_argument("--layers", type=int, default=2)
    training.add_argument("--feedforward", type=int, default=384)
    training.add_argument("--dropout", type=float, default=0.1)
    training.add_argument("--max-length", type=int, default=192)
    training.add_argument("--input-mode", choices=("word", "native"), default="word")
    evaluation = subparsers.add_parser("evaluate", parents=[shared])
    evaluation.add_argument("--model", type=Path, default=Path("models/neural_g2p_small_v1.pt"))
    evaluation.add_argument("--split", choices=("dev", "blind"), default="dev")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.command == "train":
        train(args)
    else:
        evaluate_artifact(args)
