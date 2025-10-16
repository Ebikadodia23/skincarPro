# ml/train_finetune.py
import os, json, random
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Config ---
DATA_JSONL = "../dataset/annotations.jsonl"
IMG_ROOT = "../dataset/images"
OUT_DIR = "../models"
IMG_SIZE = 384
BATCH = 8
EPOCHS = 12
LR = 2e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger.info(f"Using device: {DEVICE}")

# Ensure output directory exists
os.makedirs(OUT_DIR, exist_ok=True)

# --- Load dataset records ---
if not os.path.exists(DATA_JSONL):
    logger.error(f"Dataset file not found: {DATA_JSONL}")
    logger.info("Please run the data preparation script first or use the sample dataset")
    exit(1)

records = []
try:
    with open(DATA_JSONL, "r") as f:
        for line in f.read().strip().splitlines():
            if line.strip():
                records.append(json.loads(line))
    logger.info(f"Loaded {len(records)} records from dataset")
except Exception as e:
    logger.error(f"Error loading dataset: {e}")
    exit(1)

# --- Build label mapping for conditions ---
all_conditions = set()
for r in records:
    if 'labels' in r and 'conditions' in r['labels']:
        all_conditions.update(r['labels']['conditions'].keys())

cond_list = sorted(list(all_conditions))
num_conditions = len(cond_list)
logger.info(f"Found {num_conditions} conditions: {cond_list}")

if num_conditions == 0:
    logger.error("No conditions found in dataset")
    exit(1)

# --- Dataset class ---
class SkinDataset(Dataset):
    def __init__(self, recs, transforms=None, img_root=IMG_ROOT):
        self.recs = recs
        self.transforms = transforms
        self.img_root = img_root
        
    def __len__(self): 
        return len(self.recs)
        
    def __getitem__(self, idx):
        r = self.recs[idx]
        
        # Load image
        img_path = os.path.join(self.img_root, r['file_path'])
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.warning(f"Error loading image {img_path}: {e}")
            # Return a blank image as fallback
            img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), color=(128, 128, 128))
            
        if self.transforms: 
            img = self.transforms(img)
            
        # Prepare vector of labels (floats 0..1)
        lab = np.array([r['labels']['conditions'].get(c, 0.0) for c in cond_list], dtype=np.float32)
        
        return img, torch.from_numpy(lab)

# --- Data transforms ---
train_tf = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.15, 0.15, 0.15, 0.05),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Split data ---
train_recs, val_recs = train_test_split(records, test_size=0.15, random_state=42)
logger.info(f"Train samples: {len(train_recs)}, Validation samples: {len(val_recs)}")

train_ds = SkinDataset(train_recs, transforms=train_tf)
val_ds = SkinDataset(val_recs, transforms=val_tf)

train_loader = DataLoader(train_ds, shuffle=True, batch_size=BATCH, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_ds, shuffle=False, batch_size=BATCH, num_workers=2, pin_memory=True)

# --- Model: EfficientNet backbone + regression head ---
logger.info("Creating model...")
backbone = timm.create_model('efficientnet_b3', pretrained=True, num_classes=0, global_pool='avg')
feat_dim = backbone.num_features

head = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(feat_dim, 1024),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(1024, 512),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(512, num_conditions),
    nn.Sigmoid()  # outputs 0..1
)

model = nn.Sequential(backbone, head).to(DEVICE)
logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

# --- Loss & optimizer ---
criterion = nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# --- Training loop ---
best_val_loss = float('inf')
train_losses = []
val_losses = []

logger.info("Starting training...")

for epoch in range(1, EPOCHS + 1):
    # Training phase
    model.train()
    total_loss = 0
    num_batches = len(train_loader)
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]")
    for imgs, labels in pbar:
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)
        
        # Forward pass
        preds = model(imgs)
        loss = criterion(preds, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_train_loss = total_loss / num_batches
    train_losses.append(avg_train_loss)
    
    # Validation phase
    model.eval()
    val_loss = 0
    val_batches = len(val_loader)
    
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]"):
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            preds = model(imgs)
            val_loss += criterion(preds, labels).item()
    
    avg_val_loss = val_loss / val_batches
    val_losses.append(avg_val_loss)
    
    # Learning rate scheduling
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    
    logger.info(f"Epoch {epoch:2d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | LR: {current_lr:.6f}")
    
    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'condition_labels': cond_list
        }, os.path.join(OUT_DIR, 'skin_model_best.pth'))
        logger.info(f"New best model saved (val_loss: {avg_val_loss:.4f})")
    
    # Save checkpoint every 5 epochs
    if epoch % 5 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'condition_labels': cond_list
        }, os.path.join(OUT_DIR, f'skin_model_epoch{epoch}.pth'))

# --- Export TorchScript for serving ---
logger.info("Exporting model to TorchScript...")
model.eval()
dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)

try:
    traced = torch.jit.trace(model, dummy)
    traced_path = os.path.join(OUT_DIR, "skin_model_ts.pt")
    traced.save(traced_path)
    logger.info(f"TorchScript model saved: {traced_path}")
    
    # Save model metadata
    metadata = {
        'model_type': 'efficientnet_b3',
        'input_size': [3, IMG_SIZE, IMG_SIZE],
        'num_conditions': num_conditions,
        'condition_labels': cond_list,
        'normalization': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        }
    }
    
    with open(os.path.join(OUT_DIR, 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
except Exception as e:
    logger.error(f"Error exporting TorchScript: {e}")

# --- Save training history ---
history = {
    'train_losses': train_losses,
    'val_losses': val_losses,
    'best_val_loss': best_val_loss,
    'condition_labels': cond_list
}

with open(os.path.join(OUT_DIR, 'training_history.json'), 'w') as f:
    json.dump(history, f, indent=2)

logger.info("Training completed successfully!")
logger.info(f"Best validation loss: {best_val_loss:.4f}")
logger.info(f"Model files saved in: {OUT_DIR}")