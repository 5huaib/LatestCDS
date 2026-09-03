#!/usr/bin/env python3
import os
import glob
import time
import random
import argparse
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Import CrimeDetector3D from pipeline_prototype
from pipeline_prototype import CrimeDetector3D

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class UCFCrimeDataset(Dataset):
    """
    Dataset loader for frame-extracted UCF-Crime dataset.
    Groups frames by video ID, extracts 16-frame sub-sequences,
    and returns (C, T, H, W) tensors with binary labels (0=Normal, 1=Crime).
    """
    def __init__(self, sequences, seq_len=16, is_train=True, img_size=(112, 112)):
        self.sequences = sequences
        self.seq_len = seq_len
        self.is_train = is_train
        self.img_size = img_size

        # ImageNet normalization parameters
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        frame_paths, label = self.sequences[idx]
        
        # Random horizontal flip for data augmentation during training
        do_flip = self.is_train and (random.random() > 0.5)

        frames = []
        for path in frame_paths:
            img = cv2.imread(path)
            if img is None:
                img = np.zeros((self.img_size[1], self.img_size[0], 3), dtype=np.uint8)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, self.img_size)

            if do_flip:
                img = cv2.flip(img, 1)

            # Normalize to [0, 1] then apply ImageNet mean/std
            img_norm = img.astype(np.float32) / 255.0
            img_norm = (img_norm - self.mean) / self.std
            
            # Convert HWC to CHW
            img_tensor = img_norm.transpose(2, 0, 1)
            frames.append(img_tensor)

        # Stack into (T, C, H, W) then permute to (C, T, H, W)
        seq_tensor = np.stack(frames, axis=0)
        seq_tensor = seq_tensor.transpose(1, 0, 2, 3)

        return torch.tensor(seq_tensor, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


def parse_ucf_crime_dataset(dataset_dir, seq_len=16, stride=32):
    """
    Parses the dataset directory, groups images by video ID,
    and returns a list of 16-frame sequences.
    """
    print(f"Parsing dataset directory: {dataset_dir}...")
    categories = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    
    video_frames = defaultdict(list)
    video_labels = {}

    for cat in categories:
        cat_path = os.path.join(dataset_dir, cat)
        label = 0.0 if cat.lower() == 'normalvideos' else 1.0
        
        image_files = glob.glob(os.path.join(cat_path, '*.png')) + glob.glob(os.path.join(cat_path, '*.jpg'))
        
        for filepath in image_files:
            filename = os.path.basename(filepath)
            # Parse video name and frame index (e.g., Fighting003_x264_100.png -> Fighting003_x264, 100)
            parts = filename.rsplit('_', 1)
            if len(parts) == 2:
                video_id = parts[0]
                idx_str = os.path.splitext(parts[1])[0]
                try:
                    frame_idx = int(idx_str)
                    video_frames[video_id].append((frame_idx, filepath))
                    video_labels[video_id] = label
                except ValueError:
                    continue

    print(f"Total unique videos identified: {len(video_frames)}")
    
    # Sort frames for each video and build sequences
    all_sequences = []
    normal_seqs = 0
    crime_seqs = 0

    for video_id, items in video_frames.items():
        items.sort(key=lambda x: x[0])
        paths = [item[1] for item in items]
        label = video_labels[video_id]

        if len(paths) < seq_len:
            padded = paths + [paths[-1]] * (seq_len - len(paths))
            all_sequences.append((padded, label, video_id))
            if label == 0.0: normal_seqs += 1
            else: crime_seqs += 1
        else:
            for i in range(0, len(paths) - seq_len + 1, stride):
                seq = paths[i:i + seq_len]
                all_sequences.append((seq, label, video_id))
                if label == 0.0: normal_seqs += 1
                else: crime_seqs += 1

    print(f"Total sequences extracted: {len(all_sequences)} (Normal: {normal_seqs}, Crime: {crime_seqs})")
    return all_sequences


def train_model():
    parser = argparse.ArgumentParser(description="Train 3D-CNN Crime Detector on UCF-Crime Dataset")
    parser.add_argument("--dataset-dir", type=str, default="/Users/shuaib/Downloads/Test", help="Path to UCF-Crime dataset root folder.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training and validation.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--seq-len", type=int, default=16, help="Sequence length in frames.")
    parser.add_argument("--stride", type=int, default=32, help="Sequence sampling stride across video frames.")
    parser.add_argument("--save-path", type=str, default="best_crime_detector.pth", help="Path to save best model weights.")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation set split ratio.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader num workers.")
    args = parser.parse_args()

    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 
                          ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"Using execution device for training: {device}")

    # 1. Parse dataset and build sequences
    all_sequences = parse_ucf_crime_dataset(args.dataset_dir, seq_len=args.seq_len, stride=args.stride)
    
    # 2. Video-level train/validation split
    video_ids = list(set([seq[2] for seq in all_sequences]))
    random.shuffle(video_ids)
    
    num_val = int(len(video_ids) * args.val_split)
    val_video_ids = set(video_ids[:num_val])
    train_video_ids = set(video_ids[num_val:])

    train_seqs = [(seq[0], seq[1]) for seq in all_sequences if seq[2] in train_video_ids]
    val_seqs = [(seq[0], seq[1]) for seq in all_sequences if seq[2] in val_video_ids]

    print(f"Split Summary: Train Sequences = {len(train_seqs)}, Validation Sequences = {len(val_seqs)}")

    train_dataset = UCFCrimeDataset(train_seqs, seq_len=args.seq_len, is_train=True)
    val_dataset = UCFCrimeDataset(val_seqs, seq_len=args.seq_len, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # 3. Model, Loss, Optimizer
    model = CrimeDetector3D(use_pretrained=True).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float('inf')
    best_val_acc = 0.0

    print("\nStarting Training Loop...")
    for epoch in range(1, args.epochs + 1):
        start_t = time.time()
        
        # Training Phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs = model(inputs).squeeze(1)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            preds = (outputs >= 0.5).float()
            train_correct += (preds == targets).sum().item()
            train_total += targets.size(0)

            if (batch_idx + 1) % 25 == 0 or (batch_idx + 1) == len(train_loader):
                batch_acc = (train_correct / train_total) * 100
                batch_loss = train_loss / train_total
                print(f"  [Epoch {epoch:02d} | Batch {batch_idx+1:03d}/{len(train_loader):03d}] Loss: {batch_loss:.4f} | Acc: {batch_acc:.2f}%", flush=True)

        scheduler.step()
        epoch_train_loss = train_loss / train_total if train_total > 0 else 0
        epoch_train_acc = train_correct / train_total if train_total > 0 else 0

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        tp, fp, fn, tn = 0, 0, 0, 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs).squeeze(1)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)
                preds = (outputs >= 0.5).float()
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)

                tp += ((preds == 1.0) & (targets == 1.0)).sum().item()
                fp += ((preds == 1.0) & (targets == 0.0)).sum().item()
                fn += ((preds == 0.0) & (targets == 1.0)).sum().item()
                tn += ((preds == 0.0) & (targets == 0.0)).sum().item()

        epoch_val_loss = val_loss / val_total if val_total > 0 else 0
        epoch_val_acc = val_correct / val_total if val_total > 0 else 0
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        elapsed = time.time() - start_t
        print(f"[Epoch {epoch:02d}/{args.epochs:02d}] ({elapsed:.1f}s) "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc*100:.2f}% || "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc*100:.2f}% | "
              f"Prec: {precision:.4f} | Rec: {recall:.4f} | F1: {f1:.4f}")

        # Save Best Checkpoint
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_val_acc = epoch_val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'val_acc': best_val_acc,
            }, args.save_path)
            print(f"  --> Saved new best checkpoint to '{args.save_path}' (Val Loss: {best_val_loss:.4f}, Val Acc: {best_val_acc*100:.2f}%)")

    print(f"\nTraining completed! Best Validation Accuracy: {best_val_acc*100:.2f}%, Weights saved to: {args.save_path}")

if __name__ == "__main__":
    train_model()
