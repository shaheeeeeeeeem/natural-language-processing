import argparse
import json
import math
import os
import random
import time
from collections import namedtuple
from pathlib import Path

import tokenizers
from datasets import load_dataset

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parent
CACHE_DIR = ROOT / ".hf_cache"
CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "nmt_attention.pt"
BEST_CHECKPOINT_PATH = CHECKPOINT_DIR / "nmt_attention_best.pt"

os.environ.setdefault("HF_HOME", str(CACHE_DIR))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

FIELDS = ["src_token_ids", "src_mask", "tgt_token_ids", "tgt_mask"]


class NmtPair(namedtuple("NmtPairBase", FIELDS)):
    def to(self, device):
        return NmtPair(
            self.src_token_ids.to(device, non_blocking=True),
            self.src_mask.to(device, non_blocking=True),
            self.tgt_token_ids.to(device, non_blocking=True),
            self.tgt_mask.to(device, non_blocking=True),
        )


def attention(query, key, value, key_mask=None):
    weights = (query @ key.transpose(1, 2)) * (query.size(-1) ** -0.5)
    if key_mask is not None:
        weights = weights.masked_fill(~key_mask[:, None, :].bool(), float("-inf"))
    return torch.softmax(weights, dim=-1) @ value


class NmtModelWithAttention(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size=512,
        embed_dim=512,
        num_layers=2,
        pad_id=0,
    ):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.encoder = nn.GRU(
            embed_dim, hidden_size, num_layers=num_layers, batch_first=True
        )
        self.decoder = nn.GRU(
            embed_dim, hidden_size, num_layers=num_layers, batch_first=True
        )
        self.outputs = nn.Linear(2 * hidden_size, vocab_size)

    def forward(self, pair):
        encoder_embeddings = self.embeddings(pair.src_token_ids)
        decoder_embeddings = self.embeddings(pair.tgt_token_ids)
        lengths = pair.src_mask.sum(dim=1)
        packed = pack_padded_sequence(
            encoder_embeddings,
            lengths=lengths.cpu(),
            enforce_sorted=False,
            batch_first=True,
        )
        encoder_packed, hidden = self.encoder(packed)
        decoder_outputs, _ = self.decoder(decoder_embeddings, hidden)
        encoder_outputs, _ = pad_packed_sequence(encoder_packed, batch_first=True)
        encoder_mask = pair.src_mask[:, : encoder_outputs.size(1)]
        context = attention(
            decoder_outputs,
            encoder_outputs,
            encoder_outputs,
            key_mask=encoder_mask,
        )
        return self.outputs(torch.cat((context, decoder_outputs), dim=-1)).permute(
            0, 2, 1
        )


def load_splits():
    validation, test = load_dataset(
        "ageron/tatoeba_mt_train",
        "eng-spa",
        split=["validation", "test"],
        cache_dir=str(CACHE_DIR / "datasets"),
        keep_in_memory=False,
    )
    split = validation.train_test_split(
        train_size=0.8, seed=42, keep_in_memory=False
    )
    return split["train"], split["test"], test


def train_tokenizer(train_set, max_length=200, vocab_size=10000):
    tokenizer_path = CHECKPOINT_DIR / "nmt_tokenizer.json"
    if tokenizer_path.exists():
        return tokenizers.Tokenizer.from_file(str(tokenizer_path))

    model = tokenizers.models.BPE(unk_token="<unk>")
    tokenizer = tokenizers.Tokenizer(model)
    tokenizer.enable_padding(pad_id=0, pad_token="<pad>")
    tokenizer.enable_truncation(max_length=max_length)
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    trainer = tokenizers.trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<unk>", "<pad>", "<s>", "</s>"],
    )

    def text_iterator():
        for pair in train_set:
            yield pair["source_text"]
            yield pair["target_text"]

    tokenizer.train_from_iterator(text_iterator(), trainer)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(tokenizer_path))
    return tokenizer


def make_collate_fn(tokenizer):
    def collate(batch):
        src_texts = [pair["source_text"] for pair in batch]
        tgt_texts = [f"<s> {pair['target_text']} </s>" for pair in batch]
        src_encodings = tokenizer.encode_batch(src_texts)
        tgt_encodings = tokenizer.encode_batch(tgt_texts)
        src_ids = torch.tensor(
            [encoding.ids for encoding in src_encodings], dtype=torch.long
        )
        tgt_ids = torch.tensor(
            [encoding.ids for encoding in tgt_encodings], dtype=torch.long
        )
        src_mask = torch.tensor(
            [encoding.attention_mask for encoding in src_encodings], dtype=torch.long
        )
        tgt_mask = torch.tensor(
            [encoding.attention_mask for encoding in tgt_encodings], dtype=torch.long
        )
        inputs = NmtPair(src_ids, src_mask, tgt_ids[:, :-1], tgt_mask[:, :-1])
        return inputs, tgt_ids[:, 1:]

    return collate


def token_stats(logits, labels, pad_id=0):
    predictions = logits.argmax(dim=1)
    mask = labels.ne(pad_id)
    correct = predictions.eq(labels).logical_and(mask).sum().item()
    return correct, mask.sum().item()


def evaluate(model, loader, criterion, device, max_batches=None):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    with torch.inference_mode():
        for batch_index, (inputs, labels) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            inputs = inputs.to(device)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(inputs)
                loss = criterion(logits, labels)
            correct, tokens = token_stats(logits, labels)
            total_loss += loss.item()
            total_correct += correct
            total_tokens += tokens
    batches = min(len(loader), max_batches) if max_batches else len(loader)
    return total_loss / max(batches, 1), total_correct / max(total_tokens, 1)


def translate(model, tokenizer, text, device, max_length=40):
    start_id = tokenizer.token_to_id("<s>")
    eos_id = tokenizer.token_to_id("</s>")
    source = tokenizer.encode(text)
    src_ids = torch.tensor([source.ids], dtype=torch.long, device=device)
    src_mask = torch.tensor(
        [source.attention_mask], dtype=torch.long, device=device
    )
    generated = [start_id]
    model.eval()
    with torch.inference_mode():
        for _ in range(max_length):
            tgt_ids = torch.tensor([generated], dtype=torch.long, device=device)
            pair = NmtPair(src_ids, src_mask, tgt_ids, torch.ones_like(tgt_ids))
            next_id = int(model(pair)[0, :, -1].argmax().item())
            if next_id == eos_id:
                break
            generated.append(next_id)
    return tokenizer.decode(generated[1:], skip_special_tokens=True)


def save_checkpoint(path, model, optimizer, tokenizer, epoch, history, best_accuracy):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "tokenizer_json": tokenizer.to_str(),
            "vocab_size": tokenizer.get_vocab_size(),
            "epoch": epoch,
            "history": history,
            "best_accuracy": best_accuracy,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-valid-batches", type=int)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    train_set, valid_set, test_set = load_splits()
    tokenizer = train_tokenizer(train_set)
    collate = make_collate_fn(tokenizer)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "collate_fn": collate,
        "pin_memory": device.type == "cuda",
        "num_workers": 0,
    }
    generator = torch.Generator().manual_seed(42)
    train_loader = DataLoader(
        train_set, shuffle=True, generator=generator, **loader_kwargs
    )
    valid_loader = DataLoader(valid_set, **loader_kwargs)

    model = NmtModelWithAttention(tokenizer.get_vocab_size()).to(device)
    optimizer = torch.optim.NAdam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("<pad>"))
    start_epoch = 0
    history = {"train_loss": [], "train_accuracy": [], "valid_loss": [], "valid_accuracy": []}
    best_accuracy = 0.0

    if CHECKPOINT_PATH.exists() and not args.fresh:
        checkpoint = torch.load(
            CHECKPOINT_PATH, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        history = checkpoint.get("history", history)
        best_accuracy = checkpoint.get("best_accuracy", 0.0)
        print(f"Resuming after epoch {start_epoch}", flush=True)

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    sample_texts = [
        "Hello, I like playing football.",
        "Where is the train station?",
        "This is a beautiful day.",
    ]

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_start = time.time()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        processed_batches = 0

        for batch_index, (inputs, labels) in enumerate(train_loader):
            if (
                args.max_train_batches is not None
                and batch_index >= args.max_train_batches
            ):
                break
            inputs = inputs.to(device)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(inputs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            correct, tokens = token_stats(logits.detach(), labels)
            total_loss += loss.item()
            total_correct += correct
            total_tokens += tokens
            processed_batches += 1
            if (batch_index + 1) % 200 == 0:
                print(
                    f"Epoch {epoch + 1}/{args.epochs} "
                    f"batch {batch_index + 1}/{len(train_loader)} "
                    f"loss={total_loss / processed_batches:.4f} "
                    f"accuracy={total_correct / max(total_tokens, 1):.2%}",
                    flush=True,
                )

        train_loss = total_loss / max(processed_batches, 1)
        train_accuracy = total_correct / max(total_tokens, 1)
        valid_loss, valid_accuracy = evaluate(
            model,
            valid_loader,
            criterion,
            device,
            max_batches=args.max_valid_batches,
        )
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["valid_loss"].append(valid_loss)
        history["valid_accuracy"].append(valid_accuracy)

        best_accuracy = max(best_accuracy, valid_accuracy)
        if not args.no_save:
            save_checkpoint(
                CHECKPOINT_PATH,
                model,
                optimizer,
                tokenizer,
                epoch,
                history,
                best_accuracy,
            )
            if valid_accuracy >= best_accuracy:
                save_checkpoint(
                    BEST_CHECKPOINT_PATH,
                    model,
                    optimizer,
                    tokenizer,
                    epoch,
                    history,
                    best_accuracy,
                )

        elapsed = time.time() - epoch_start
        print(
            f"Epoch {epoch + 1}/{args.epochs}: "
            f"train_loss={train_loss:.4f}, train_accuracy={train_accuracy:.2%}, "
            f"valid_loss={valid_loss:.4f}, valid_accuracy={valid_accuracy:.2%}, "
            f"time={elapsed / 60:.1f} min",
            flush=True,
        )
        for text in sample_texts:
            print(f"  {text} -> {translate(model, tokenizer, text, device)}", flush=True)

    results = {
        "checkpoint": str(CHECKPOINT_PATH),
        "best_checkpoint": str(BEST_CHECKPOINT_PATH),
        "history": history,
        "samples": {
            text: translate(model, tokenizer, text, device) for text in sample_texts
        },
        "rows": {
            "train": len(train_set),
            "validation": len(valid_set),
            "test": len(test_set),
        },
    }
    if not args.no_save:
        (CHECKPOINT_DIR / "training_results.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8"
        )
    print(json.dumps(results["samples"], indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
