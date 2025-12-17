import os
import json
import math
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
from decord import VideoReader, cpu
from transformers import AutoModelForCausalLM, get_linear_schedule_with_warmup
from huggingface_hub import snapshot_download

# Apollo utils from your repo
from utils.mm_utils import ApolloMMLoader
from utils.constants import X_TOKEN, X_START_TOKEN, X_END_TOKEN

# ---------------- CONFIG ---------------- #

MODEL_REPO     = "GoodiesHere/Apollo-LMMs-Apollo-3B-t32"

# Your data folder + JSONs
DATA_DIR       = "data/bees"
TRAIN_JSON     = os.path.join(DATA_DIR, "bee_vqa_train.json")
VAL_JSON       = os.path.join(DATA_DIR, "bee_vqa_val.json")

# Training hyperparams (keep tiny)
N_FRAMES       = 8        # sampled frames per video (ApolloMMLoader also controls this)
BATCH_SIZE     = 1        # keep at 1 to reduce memory
N_EPOCHS       = 3
LR             = 1e-5
WARMUP_RATIO   = 0.1

# ---- FORCE CPU HERE ----
DEVICE         = "cpu"
TARGET_DTYPE   = torch.float32

MAX_TEXT_LEN   = 512


# ---------------- DATASET ---------------- #

class BeeVQADataset(Dataset):
    """
    JSON item format:

    [
      {
        "video": "data/bees/videos/sample1.mp4",
        "question": "What are the bees doing?",
        "answer": "They are foraging and performing waggle dances."
      },
      ...
    ]
    """
    def __init__(self, json_path: str):
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found: {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            self.items = json.load(f)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        video_path = item["video"]
        question   = item["question"]
        answer     = item["answer"]
        return video_path, question, answer


# ---------------- (OPTIONAL) RAW VIDEO LOADER ---------------- #
# We don't actually use this directly because ApolloMMLoader
# handles video processing internally, but keeping it here if needed.

def load_video_clip(path: str, n_frames: int = N_FRAMES) -> torch.Tensor:
    """Load video, sample n_frames uniformly, return (T, C, H, W) float32."""
    vr = VideoReader(path, ctx=cpu(0))
    total = len(vr)
    if total <= 0:
        raise ValueError(f"Empty video: {path}")

    import numpy as np
    if total >= n_frames:
        inds = np.linspace(0, total - 1, n_frames).astype("int64").tolist()
    else:
        inds = list(range(total))
        while len(inds) < n_frames:
            inds.append(total - 1)

    frames = vr.get_batch(inds)      # (T, H, W, 3)
    frames = frames.float() / 255.0
    frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W)
    return frames


# ---------------- TEXT BUILDING ---------------- #

def build_text_and_labels(
    tokenizer,
    question: str,
    answer: str,
    data_type: str = "video",
) -> Dict[str, torch.Tensor]:
    """
    Build full text: [VIDEO_TAG + question] → answer

    We mask the prompt tokens with -100 so loss is only on the answer part.
    """
    video_placeholder = X_TOKEN[data_type]

    prompt = (
        f"{X_START_TOKEN[data_type]}{video_placeholder}{X_END_TOKEN[data_type]}\n\n"
        f"User: {question}\n"
        f"Assistant:"
    )

    full_text = prompt + " " + answer

    full = tokenizer(
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_TEXT_LEN,
        add_special_tokens=True,
    )
    prompt_only = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_TEXT_LEN,
        add_special_tokens=True,
    )

    input_ids = full["input_ids"][0]          # (L,)
    attention_mask = full["attention_mask"][0]
    prompt_len = prompt_only["input_ids"].shape[1]

    labels = input_ids.clone()
    labels[:prompt_len] = -100               # ignore prompt tokens in loss

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


# ---------------- COLLATE ---------------- #

def collate_fn(
    batch: List,
    mm_loader: ApolloMMLoader,
    tokenizer,
) -> Dict[str, Any]:
    """
    batch: list of (video_path, question, answer)
    Returns:
      - input_ids: (B, L)
      - attention_mask: (B, L)
      - labels: (B, L)
      - vision_input: list of video features (for Apollo)
      - data_types: list of "video"
    """
    input_ids_list = []
    attn_list = []
    labels_list = []
    video_feats = []

    max_len = 0

    for video_path, question, answer in batch:
        # Resolve relative → absolute path
        video_path = os.path.abspath(video_path)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        # ---- Text ----
        text_dict = build_text_and_labels(tokenizer, question, answer, data_type="video")
        input_ids_list.append(text_dict["input_ids"])
        attn_list.append(text_dict["attention_mask"])
        labels_list.append(text_dict["labels"])
        max_len = max(max_len, text_dict["input_ids"].shape[0])

        # ---- Video via ApolloMMLoader ----
        # ApolloMMLoader returns whatever the model expects (often nested dict/list)
        video_tensor, _ = mm_loader.load_video(video_path)
        video_feats.append(video_tensor)

    def pad_tensor(t: torch.Tensor, max_len: int, pad_id: int, pad_label: int = -100):
        if t.shape[0] == max_len:
            return t
        pad_size = max_len - t.shape[0]
        val = pad_id if pad_label == -100 else pad_label
        pad = torch.full((pad_size,), val, dtype=t.dtype)
        return torch.cat([t, pad], dim=0)

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    input_ids_batch = []
    attn_batch = []
    labels_batch = []

    for inp, attn, lab in zip(input_ids_list, attn_list, labels_list):
        input_ids_batch.append(pad_tensor(inp,  max_len, pad_token_id))
        attn_batch.append(pad_tensor(attn,      max_len, 0))
        labels_batch.append(pad_tensor(lab,     max_len, -100))

    input_ids_batch = torch.stack(input_ids_batch, dim=0).to(DEVICE)
    attn_batch      = torch.stack(attn_batch, dim=0).to(DEVICE)
    labels_batch    = torch.stack(labels_batch, dim=0).to(DEVICE)

    # Move vision input to DEVICE + TARGET_DTYPE (CPU + float32 now)
    def cast_to(x, device, dtype):
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=dtype)
        if isinstance(x, (list, tuple)):
            return type(x)(cast_to(xx, device, dtype) for xx in x)
        if isinstance(x, dict):
            return {k: cast_to(v, device, dtype) for k, v in x.items()}
        return x

    video_feats = [cast_to(vf, DEVICE, TARGET_DTYPE) for vf in video_feats]

    return {
        "input_ids": input_ids_batch,
        "attention_mask": attn_batch,
        "labels": labels_batch,
        "vision_input": video_feats,
        "data_types": ["video"] * len(video_feats),
    }


# ---------------- TRAINING ---------------- #

def main():
    print("Device:", DEVICE)

    # ---- Load Apollo on CPU ----
    print("Downloading / loading Apollo…")
    model_path = snapshot_download(MODEL_REPO, repo_type="model")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        attn_implementation="eager",  # SDPA not used on CPU
    )
    model = model.to(DEVICE, dtype=TARGET_DTYPE)
    model.train()

    tokenizer = model.tokenizer
    vision_processor = model.vision_tower.vision_processor
    cfg = model.config

    # Apollo video loader (CPU)
    frames_per_clip = 4
    clip_duration = getattr(cfg, "clip_duration")

    mm_loader = ApolloMMLoader(
        vision_processor,
        clip_duration,
        frames_per_clip,
        clip_sampling_ratio=0.5,
        model_max_length=cfg.model_max_length,
        device=DEVICE,  # CPU
        num_repeat_token=cfg.mm_connector_cfg["num_output_tokens"],
    )

    # ---- Datasets ----
    train_ds = BeeVQADataset(TRAIN_JSON)
    val_ds   = BeeVQADataset(VAL_JSON)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, mm_loader, tokenizer),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, mm_loader, tokenizer),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    num_training_steps = N_EPOCHS * len(train_loader)
    num_warmup_steps = int(WARMUP_RATIO * num_training_steps) if num_training_steps > 0 else 0
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max(num_training_steps, 1),
    )

    best_val_loss = math.inf

    for epoch in range(N_EPOCHS):
        print(f"\n===== Epoch {epoch+1}/{N_EPOCHS} =====")
        # ---- Train ----
        model.train()
        total_train_loss = 0.0

        for step, batch in enumerate(train_loader):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                vision_input=batch["vision_input"],
                data_types=batch["data_types"],
                labels=batch["labels"],
            )
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            print(f"  [train] step {step+1}/{len(train_loader)} - loss: {loss.item():.4f}")

        if len(train_loader) > 0:
            avg_train_loss = total_train_loss / len(train_loader)
        else:
            avg_train_loss = float("nan")
        print(f"Epoch {epoch+1} - avg train loss: {avg_train_loss:.4f}")

        # ---- Validation ----
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    vision_input=batch["vision_input"],
                    data_types=batch["data_types"],
                    labels=batch["labels"],
                )
                loss = outputs.loss
                total_val_loss += loss.item()
                print(f"  [val] step {step+1}/{len(val_loader)} - loss: {loss.item():.4f}")

        if len(val_loader) > 0:
            avg_val_loss = total_val_loss / len(val_loader)
        else:
            avg_val_loss = float("nan")
        print(f"Epoch {epoch+1} - avg val loss: {avg_val_loss:.4f}")

        # ---- Save best ----
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs("checkpoints", exist_ok=True)
            save_path = os.path.join("checkpoints", "apollo_bee_vqa_best_cpu.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  ✅ New best val loss {best_val_loss:.4f}, saved: {save_path}")

    print(f"\nBest validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
