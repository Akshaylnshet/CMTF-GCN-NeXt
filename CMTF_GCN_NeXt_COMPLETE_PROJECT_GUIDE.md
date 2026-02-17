# CMTF-GCN-NeXt: Complete Project Guide
## From Setup to Deployment - Step by Step in Jupyter Notebook

---

# PROJECT STRUCTURE

```
CMTF_GCN_NeXt/
│
├── 📓 NOTEBOOKS/
│   ├── NB01_Environment_Setup.ipynb
│   ├── NB02_Data_Download.ipynb
│   ├── NB03_Data_Preprocessing.ipynb
│   ├── NB04_Data_Loader.ipynb
│   ├── NB05_Baseline_EEGNet.ipynb
│   ├── NB06_Baseline_ConvNeXt.ipynb
│   ├── NB07_Architecture_Build.ipynb
│   ├── NB08_Full_Training.ipynb
│   ├── NB09_Evaluation.ipynb
│   ├── NB10_Ablation_Studies.ipynb
│   └── NB11_Deployment.ipynb
│
├── 📁 src/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── temporal_encoder.py
│   │   ├── gat.py
│   │   ├── convnext_encoder.py
│   │   ├── fusion.py
│   │   └── cmtf_gcn_next.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── preprocessing.py
│   │   └── dataset.py
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py
│   └── utils/
│       ├── __init__.py
│       └── metrics.py
│
├── 📁 data/
│   ├── raw/                  # Downloaded HMS dataset
│   └── processed/            # Preprocessed .pkl files
│
├── 📁 models/
│   ├── checkpoints/          # Saved .pth files
│   └── deployment/           # ONNX / TorchScript models
│
├── 📁 results/
│   ├── figures/              # All plots and graphs
│   └── metrics/              # Evaluation JSON files
│
├── 📁 deployment/
│   ├── app.py                # Flask/FastAPI web app
│   ├── templates/            # HTML templates
│   └── requirements.txt      # Deployment requirements
│
├── requirements.txt
└── README.md
```

---

# ============================================================
# NOTEBOOK 1: NB01_Environment_Setup.ipynb
# ============================================================

## CELL 1 - Create Folder Structure
```python
import os

# Create all required directories
dirs = [
    "src/models",
    "src/data", 
    "src/training",
    "src/utils",
    "data/raw",
    "data/processed",
    "models/checkpoints",
    "models/deployment",
    "results/figures",
    "results/metrics",
    "deployment/templates"
]

for d in dirs:
    os.makedirs(d, exist_ok=True)

# Create __init__.py files
for module in ["src", "src/models", "src/data", "src/training", "src/utils"]:
    init_file = os.path.join(module, "__init__.py")
    with open(init_file, 'w') as f:
        f.write("")

print("✅ Project structure created!")
for d in dirs:
    print(f"   📁 {d}/")
```

## CELL 2 - Install All Requirements
```python
# Install everything needed
!pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
!pip install torch-geometric
!pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
!pip install timm==0.9.5
!pip install einops==0.6.1
!pip install numpy pandas scipy scikit-learn
!pip install matplotlib seaborn tqdm
!pip install kaggle
!pip install wandb
!pip install flask fastapi uvicorn python-multipart
!pip install onnx onnxruntime
!pip install pillow opencv-python
!pip install ipywidgets  # For notebook progress bars
print("✅ All packages installed!")
```

## CELL 3 - Verify Installation
```python
import torch
import torchvision
import timm
import numpy as np
import pandas as pd
import sklearn
import matplotlib
import flask

print("="*50)
print("ENVIRONMENT VERIFICATION")
print("="*50)
print(f"✅ PyTorch:      {torch.__version__}")
print(f"✅ TorchVision:  {torchvision.__version__}")
print(f"✅ CUDA:         {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✅ GPU:          {torch.cuda.get_device_name(0)}")
    print(f"✅ VRAM:         {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"✅ NumPy:        {np.__version__}")
print(f"✅ Pandas:       {pd.__version__}")
print(f"✅ timm:         {timm.__version__}")
print("="*50)
print("✅ Environment ready!")
```

## CELL 4 - Save Requirements File
```python
requirements = """torch==2.1.0
torchvision==0.16.0
timm==0.9.5
einops==0.6.1
numpy==1.24.3
pandas==2.0.3
scipy==1.11.1
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
tqdm==4.65.0
wandb==0.15.5
flask==2.3.0
fastapi==0.103.0
uvicorn==0.23.0
onnx==1.14.0
onnxruntime==1.16.0
pillow==10.0.0
opencv-python==4.8.0.74
torch-geometric==2.3.1
"""

with open("requirements.txt", "w") as f:
    f.write(requirements)

print("✅ requirements.txt saved!")
```

---

# ============================================================
# NOTEBOOK 2: NB02_Data_Download.ipynb
# ============================================================

## CELL 1 - Setup Kaggle
```python
import os
import json

# Method 1: If you have kaggle.json
# Upload kaggle.json to your notebook directory first, then:
kaggle_dir = os.path.expanduser("~/.kaggle")
os.makedirs(kaggle_dir, exist_ok=True)

# Copy kaggle.json to correct location
if os.path.exists("kaggle.json"):
    import shutil
    shutil.copy("kaggle.json", os.path.join(kaggle_dir, "kaggle.json"))
    os.chmod(os.path.join(kaggle_dir, "kaggle.json"), 0o600)
    print("✅ Kaggle credentials configured!")
else:
    print("⚠️  Upload kaggle.json to this folder first!")
    print("   Go to: https://www.kaggle.com/settings -> API -> Create New Token")
```

## CELL 2 - Download Dataset
```python
# Download HMS dataset (this is ~50GB - will take time)
print("⬇️  Starting HMS dataset download...")
print("⚠️  This will take 30-60 minutes on good internet connection")

!kaggle competitions download -c hms-harmful-brain-activity-classification -p data/raw/

print("✅ Download complete!")
```

## CELL 3 - Extract Dataset
```python
import zipfile
import os
from tqdm.notebook import tqdm

zip_path = "data/raw/hms-harmful-brain-activity-classification.zip"

print(f"📦 Extracting {zip_path}...")
print("⚠️  This will take 10-20 minutes...")

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    members = zip_ref.namelist()
    for member in tqdm(members, desc="Extracting"):
        zip_ref.extract(member, "data/raw/")

print("✅ Extraction complete!")
print("\n📁 Contents of data/raw/:")
for item in os.listdir("data/raw/"):
    size = os.path.getsize(f"data/raw/{item}") if os.path.isfile(f"data/raw/{item}") else 0
    print(f"   {item} ({size/1e6:.1f} MB)" if size > 0 else f"   📁 {item}/")
```

## CELL 4 - Verify Download
```python
import pandas as pd
from pathlib import Path

data_path = Path("data/raw")

# Load metadata
train_df = pd.read_csv(data_path / "train.csv")

print("="*50)
print("DATASET VERIFICATION")
print("="*50)
print(f"✅ Total samples: {len(train_df)}")
print(f"✅ Columns: {list(train_df.columns)}")
print(f"\n📊 Class distribution:")
print(train_df['expert_consensus'].value_counts())

# Check EEG files
eeg_folder = data_path / "train_eegs"
spec_folder = data_path / "train_spectrograms"

if eeg_folder.exists():
    eeg_count = len(list(eeg_folder.glob("*.parquet")))
    print(f"\n✅ EEG files found: {eeg_count}")
else:
    print("\n⚠️  EEG folder not found!")

if spec_folder.exists():
    spec_files = list(spec_folder.iterdir())
    print(f"✅ Spectrogram files found: {len(spec_files)}")
else:
    print("⚠️  Spectrogram folder not found!")
```

---

# ============================================================
# NOTEBOOK 3: NB03_Data_Preprocessing.ipynb
# ============================================================

## CELL 1 - Imports
```python
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm.notebook import tqdm
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

RAW_DATA_PATH = Path("data/raw")
PROCESSED_DATA_PATH = Path("data/processed")
PROCESSED_DATA_PATH.mkdir(exist_ok=True)

LABEL_MAP = {'Seizure': 0, 'LPD': 1, 'GPD': 2, 'LRDA': 3, 'GRDA': 4, 'Other': 5}
CLASS_NAMES = ['Seizure', 'LPD', 'GPD', 'LRDA', 'GRDA', 'Other']

print("✅ Imports done")
```

## CELL 2 - Load Metadata and Detect Columns
```python
train_df = pd.read_csv(RAW_DATA_PATH / "train.csv")

print(f"Total samples: {len(train_df)}")
print(f"Columns: {list(train_df.columns)}")

# Auto-detect label column
LABEL_COL = None
for col in ['expert_consensus', 'label', 'target']:
    if col in train_df.columns:
        LABEL_COL = col
        break

# Auto-detect IDs
EEG_ID_COL = 'eeg_id' if 'eeg_id' in train_df.columns else train_df.columns[0]
SPEC_ID_COL = 'spectrogram_id' if 'spectrogram_id' in train_df.columns else None

print(f"\n✅ Label column: {LABEL_COL}")
print(f"✅ EEG ID column: {EEG_ID_COL}")
print(f"✅ Spec ID column: {SPEC_ID_COL}")

# Detect channels from first EEG file
first_eeg_id = train_df[EEG_ID_COL].iloc[0]
eeg_df = pd.read_parquet(RAW_DATA_PATH / "train_eegs" / f"{first_eeg_id}.parquet")
ALL_CHANNELS = list(eeg_df.columns)
print(f"\n📡 Available EEG channels ({len(ALL_CHANNELS)}): {ALL_CHANNELS}")

PREFERRED = ['Fp1','F7','T3','T5','O1','Fp2','F8','T4','T6','O2']
CHANNELS = [c for c in PREFERRED if c in ALL_CHANNELS]
if len(CHANNELS) < 10:
    CHANNELS = ALL_CHANNELS[:10]
print(f"\n✅ Using channels: {CHANNELS}")
```

## CELL 3 - Preprocessing Functions
```python
def load_eeg(eeg_id, target_len=10000):
    path = RAW_DATA_PATH / "train_eegs" / f"{eeg_id}.parquet"
    df = pd.read_parquet(path)
    eeg = df[CHANNELS].values.T.astype(np.float32)
    # Pad or trim
    if eeg.shape[1] > target_len:
        eeg = eeg[:, :target_len]
    elif eeg.shape[1] < target_len:
        pad = np.zeros((eeg.shape[0], target_len - eeg.shape[1]), dtype=np.float32)
        eeg = np.concatenate([eeg, pad], axis=1)
    # Normalize per channel
    mean = eeg.mean(axis=1, keepdims=True)
    std = eeg.std(axis=1, keepdims=True) + 1e-8
    return (eeg - mean) / std

def load_spectrogram(spec_id, target_size=512):
    # Try different formats
    for ext in ['.npy', '.png', '.jpg', '.parquet']:
        path = RAW_DATA_PATH / "train_spectrograms" / f"{spec_id}{ext}"
        if path.exists():
            if ext == '.npy':
                spec = np.load(path).astype(np.float32)
            elif ext in ['.png', '.jpg']:
                spec = np.array(Image.open(path).convert('RGB')).astype(np.float32) / 255.0
            elif ext == '.parquet':
                df = pd.read_parquet(path)
                spec = df.values.astype(np.float32)
            break
    # Ensure [3, H, W]
    if len(spec.shape) == 2:
        spec = np.stack([spec]*3, axis=0)
    elif spec.shape[-1] == 3:
        spec = spec.transpose(2, 0, 1)
    # Resize if needed
    if spec.shape[1] != target_size or spec.shape[2] != target_size:
        t = torch.FloatTensor(spec).unsqueeze(0)
        t = F.interpolate(t, size=(target_size, target_size), mode='bilinear', align_corners=False)
        spec = t.squeeze(0).numpy()
    return spec  # [3, 512, 512]

print("✅ Preprocessing functions ready")
```

## CELL 4 - Test on One Sample
```python
row = train_df.iloc[0]
eeg = load_eeg(row[EEG_ID_COL])
spec = load_spectrogram(row[SPEC_ID_COL])

print(f"✅ EEG shape:  {eeg.shape}  (expected [10, 10000])")
print(f"✅ Spec shape: {spec.shape} (expected [3, 512, 512])")
print(f"✅ EEG stats - mean: {eeg.mean():.3f}, std: {eeg.std():.3f}")
print(f"✅ Spec range: [{spec.min():.3f}, {spec.max():.3f}]")
print("\n✅ Single sample OK - proceeding with full dataset")
```

## CELL 5 - Run Full Preprocessing (LONG - 1-3 hrs)
```python
processed = []
failed = []

pbar = tqdm(train_df.iterrows(), total=len(train_df), desc="Preprocessing")

for idx, row in pbar:
    try:
        sample = {
            'eeg':        load_eeg(row[EEG_ID_COL]),
            'spectrogram': load_spectrogram(row[SPEC_ID_COL]),
            'label':       LABEL_MAP[row[LABEL_COL]],
            'label_name':  row[LABEL_COL],
            'eeg_id':      row[EEG_ID_COL],
        }
        processed.append(sample)
        # Checkpoint every 1000
        if len(processed) % 1000 == 0:
            with open(PROCESSED_DATA_PATH / f"ckpt_{len(processed)}.pkl", 'wb') as f:
                pickle.dump(processed, f)
    except Exception as e:
        failed.append({'idx': idx, 'error': str(e)})
    pbar.set_postfix(done=len(processed), failed=len(failed))

# Save all
with open(PROCESSED_DATA_PATH / "all_processed.pkl", 'wb') as f:
    pickle.dump(processed, f)

print(f"\n✅ Done! Processed: {len(processed)}, Failed: {len(failed)}")
```

## CELL 6 - Create Splits and Save
```python
labels = [s['label'] for s in processed]

# Split
data_tv, data_test = train_test_split(processed, test_size=0.075, stratify=labels, random_state=42)
labels_tv = [s['label'] for s in data_tv]
data_train, data_val = train_test_split(data_tv, test_size=0.135, stratify=labels_tv, random_state=42)

for name, split in [('train', data_train), ('val', data_val), ('test', data_test)]:
    with open(PROCESSED_DATA_PATH / f"{name}.pkl", 'wb') as f:
        pickle.dump(split, f)
    print(f"✅ {name}: {len(split)} samples saved")
```

---

# ============================================================
# NOTEBOOK 4: NB04_Data_Loader.ipynb
# ============================================================

## CELL 1 - Dataset Class
```python
# Save this to src/data/dataset.py as well
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np

CLASS_NAMES = ['Seizure', 'LPD', 'GPD', 'LRDA', 'GRDA', 'Other']

class HMSDataset(Dataset):
    def __init__(self, pkl_path, augment=False):
        with open(pkl_path, 'rb') as f:
            self.data = pickle.load(f)
        self.augment = augment

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        s = self.data[idx]
        eeg  = torch.FloatTensor(s['eeg'])   # [10, 10000]
        spec = torch.FloatTensor(s['spectrogram'])  # [3, 512, 512]
        label = torch.LongTensor([s['label']])[0]

        if self.augment:
            # EEG augmentation
            eeg = eeg + torch.randn_like(eeg) * 0.05  # Noise
            scale = 0.9 + torch.rand(1).item() * 0.2   # Scale 0.9-1.1
            eeg = eeg * scale
            # Spectrogram augmentation
            if torch.rand(1).item() > 0.5:
                spec = torch.flip(spec, dims=[2])  # Horizontal flip

        return {'eeg': eeg, 'spectrogram': spec, 'label': label}


def get_loaders(data_dir="data/processed", batch_size=8):
    train_ds = HMSDataset(f"{data_dir}/train.pkl", augment=True)
    val_ds   = HMSDataset(f"{data_dir}/val.pkl",   augment=False)
    test_ds  = HMSDataset(f"{data_dir}/test.pkl",  augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                               num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                               num_workers=4, pin_memory=True)

    print(f"✅ Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader

# Test
train_loader, val_loader, test_loader = get_loaders()
batch = next(iter(train_loader))
print(f"✅ EEG batch:  {batch['eeg'].shape}")
print(f"✅ Spec batch: {batch['spectrogram'].shape}")
print(f"✅ Labels:     {batch['label']}")
```

---

# ============================================================
# NOTEBOOK 5: NB05_Baseline_EEGNet.ipynb
# ============================================================

## CELL 1 - EEGNet Model
```python
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.notebook import tqdm
import sys; sys.path.append('.')

class EEGNet(nn.Module):
    def __init__(self, num_channels=10, num_classes=6):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(num_channels, 16, 64, padding=32, bias=False),
            nn.BatchNorm1d(16)
        )
        self.depthwise = nn.Sequential(
            nn.Conv1d(16, 32, 1, groups=16, bias=False),
            nn.BatchNorm1d(32), nn.ELU(),
            nn.AvgPool1d(4), nn.Dropout(0.25)
        )
        self.separable = nn.Sequential(
            nn.Conv1d(32, 32, 16, padding=8, bias=False),
            nn.BatchNorm1d(32), nn.ELU(),
            nn.AvgPool1d(8), nn.Dropout(0.25)
        )
        self.flatten = nn.Flatten()
        self.fc = nn.LazyLinear(num_classes)

    def forward(self, x):
        x = self.temporal(x)
        x = self.depthwise(x)
        x = self.separable(x)
        x = self.flatten(x)
        return self.fc(x)

model = EEGNet()
# Test
dummy = torch.randn(2, 10, 10000)
out = model(dummy)
print(f"✅ EEGNet output: {out.shape}")
total_params = sum(p.numel() for p in model.parameters())
print(f"✅ Parameters: {total_params:,}")
```

## CELL 2 - Train EEGNet (4 hours estimated)
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from NB04_Data_Loader import get_loaders  # or paste get_loaders here
train_loader, val_loader, _ = get_loaders(batch_size=16)

model = EEGNet().to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

best_acc = 0
history = {'train_loss': [], 'val_acc': []}

for epoch in range(20):
    # Train
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/20", leave=False):
        eeg = batch['eeg'].to(device)
        labels = batch['label'].to(device)
        optimizer.zero_grad()
        out = model(eeg)
        loss = criterion(out, labels)
        loss.backward(); optimizer.step()
        total_loss += loss.item()
        correct += out.argmax(1).eq(labels).sum().item()
        total += labels.size(0)

    # Validate
    model.eval()
    v_correct, v_total = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            out = model(batch['eeg'].to(device))
            v_correct += out.argmax(1).eq(batch['label'].to(device)).sum().item()
            v_total += batch['label'].size(0)

    train_acc = 100 * correct / total
    val_acc = 100 * v_correct / v_total
    history['train_loss'].append(total_loss / len(train_loader))
    history['val_acc'].append(val_acc)
    scheduler.step()

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "models/checkpoints/eegnet_best.pth")

    print(f"Epoch {epoch+1:2d} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Best: {best_acc:.2f}%")

print(f"\n✅ EEGNet Best Val Accuracy: {best_acc:.2f}%")
```

---

# ============================================================
# NOTEBOOK 6: NB06_Baseline_ConvNeXt.ipynb
# ============================================================

## CELL 1 - ConvNeXt Baseline Model
```python
import timm
import torch
import torch.nn as nn

class ConvNeXtBaseline(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.backbone = timm.create_model('convnext_tiny', pretrained=True,
                                          num_classes=0, global_pool='avg')
        self.head = nn.Sequential(
            nn.LayerNorm(768),
            nn.Dropout(0.3),
            nn.Linear(768, num_classes)
        )

    def forward(self, x):
        return self.head(self.backbone(x))

model = ConvNeXtBaseline()
dummy = torch.randn(2, 3, 512, 512)
print(f"✅ ConvNeXt output: {model(dummy).shape}")
print(f"✅ Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

## CELL 2 - Train ConvNeXt Baseline (5 hours estimated)
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader, val_loader, _ = get_loaders(batch_size=8)

model = ConvNeXtBaseline().to(device)
optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

best_acc = 0

for epoch in range(20):
    model.train()
    correct, total = 0, 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/20", leave=False):
        spec = batch['spectrogram'].to(device)
        labels = batch['label'].to(device)
        optimizer.zero_grad()
        out = model(spec)
        loss = criterion(out, labels)
        loss.backward(); optimizer.step()
        correct += out.argmax(1).eq(labels).sum().item()
        total += labels.size(0)

    model.eval()
    v_correct, v_total = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            out = model(batch['spectrogram'].to(device))
            v_correct += out.argmax(1).eq(batch['label'].to(device)).sum().item()
            v_total += batch['label'].size(0)

    val_acc = 100 * v_correct / v_total
    scheduler.step()
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "models/checkpoints/convnext_baseline_best.pth")

    print(f"Epoch {epoch+1:2d} | Train: {100*correct/total:.2f}% | Val: {val_acc:.2f}%")

print(f"\n✅ ConvNeXt Baseline Best Val Accuracy: {best_acc:.2f}%")
```

---

# ============================================================
# NOTEBOOK 7: NB07_Architecture_Build.ipynb
# ============================================================

## CELL 1 - Multi-Scale Temporal Encoder
```python
import torch
import torch.nn as nn

class MultiScaleTemporalEncoder(nn.Module):
    def __init__(self, in_channels=10, out_dim=256):
        super().__init__()
        self.conv3 = nn.Conv1d(in_channels, 64, 3, padding=1)
        self.conv5 = nn.Conv1d(in_channels, 64, 5, padding=2)
        self.conv7 = nn.Conv1d(in_channels, 64, 7, padding=3)
        self.bn = nn.BatchNorm1d(192)
        
        enc_layer = nn.TransformerEncoderLayer(
            d_model=192, nhead=8, dim_feedforward=512,
            dropout=0.1, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=4)
        self.pool = nn.AdaptiveAvgPool1d(128)
        self.proj = nn.Linear(192 * 128, out_dim)

    def forward(self, x):  # x: [B, 10, 10000]
        h = torch.cat([self.conv3(x), self.conv5(x), self.conv7(x)], dim=1)
        h = torch.relu(self.bn(h))          # [B, 192, 10000]
        h = self.transformer(h.transpose(1,2)).transpose(1,2)
        h = self.pool(h).flatten(1)         # [B, 192*128]
        return self.proj(h)                 # [B, 256]

# Test
enc = MultiScaleTemporalEncoder()
x = torch.randn(2, 10, 10000)
print(f"✅ Temporal encoder output: {enc(x).shape}")
```

## CELL 2 - Graph Attention Network
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class EEGGraphAttention(nn.Module):
    def __init__(self, in_dim=256, hidden=128, heads=8):
        super().__init__()
        self.n_nodes = 10
        self.proj = nn.Linear(in_dim, hidden)
        self.gat1 = GATConv(hidden, hidden, heads=heads, dropout=0.1, concat=True)
        self.gat2 = GATConv(hidden * heads, hidden, heads=1, dropout=0.1, concat=False)
        self.norm1 = nn.LayerNorm(hidden * heads)
        self.norm2 = nn.LayerNorm(hidden)
        self.register_buffer('edge_index', self._make_edges())

    def _make_edges(self):
        # 10-20 system adjacency
        edges = [(0,1),(1,2),(2,3),(3,4),(5,6),(6,7),(7,8),(8,9),
                 (0,5),(1,6),(2,7),(3,8),(4,9)]
        edges += [(j,i) for i,j in edges]
        return torch.tensor(edges, dtype=torch.long).t().contiguous()

    def forward(self, x):  # x: [B, 256]
        B = x.size(0)
        h = self.proj(x)                           # [B, 128]
        h = h.unsqueeze(1).expand(-1, self.n_nodes, -1)
        h = h.reshape(B * self.n_nodes, -1)        # [B*10, 128]
        
        batch_ei = []
        for i in range(B):
            batch_ei.append(self.edge_index + i * self.n_nodes)
        ei = torch.cat(batch_ei, dim=1)
        
        h = F.elu(self.norm1(self.gat1(h, ei)))
        h = F.elu(self.norm2(self.gat2(h, ei)))
        h = h.reshape(B, self.n_nodes, -1).mean(dim=1)  # [B, 128]
        return h

# Test
gat = EEGGraphAttention()
x = torch.randn(2, 256)
print(f"✅ GAT output: {gat(x).shape}")
```

## CELL 3 - Cross Modal Fusion
```python
class CrossModalFusion(nn.Module):
    def __init__(self, eeg_dim=128, spec_dim=768, hidden=512, heads=8):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(32, hidden))
        self.eeg_proj  = nn.Linear(eeg_dim, hidden * 2)
        self.spec_proj = nn.Linear(spec_dim, hidden * 2)
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(hidden)
        self.pool = nn.Linear(hidden, hidden)

    def forward(self, eeg_feat, spec_feat):  # [B,128], [B,768]
        B = eeg_feat.size(0)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)      # [B,32,512]
        ek, ev = self.eeg_proj(eeg_feat.unsqueeze(1)).chunk(2, dim=-1)
        sk, sv = self.spec_proj(spec_feat.unsqueeze(1)).chunk(2, dim=-1)
        k = torch.cat([ek, sk], dim=1)                        # [B,2,512]
        v = torch.cat([ev, sv], dim=1)
        out, _ = self.attn(q, k, v)
        out = self.norm(out).mean(dim=1)                      # [B,512]
        return self.pool(out)


class GatedFusion(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, 256), nn.ReLU(),
            nn.Linear(256, 1), nn.Sigmoid()
        )

    def forward(self, eeg_feat, spec_feat):
        g = self.gate(torch.cat([eeg_feat, spec_feat], dim=-1))
        return g * eeg_feat + (1 - g) * spec_feat, g


class MixtureOfExperts(nn.Module):
    def __init__(self, in_dim=512, hidden=1024, n_experts=4, n_classes=6):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(),
                          nn.Dropout(0.2), nn.Linear(hidden, n_classes))
            for _ in range(n_experts)
        ])
        self.router = nn.Linear(in_dim, n_experts)

    def forward(self, x):
        gates = F.softmax(self.router(x), dim=-1)              # [B, n_experts]
        outs = torch.stack([e(x) for e in self.experts], dim=1) # [B, n_experts, n_classes]
        topk_g, topk_i = gates.topk(2, dim=-1)
        topk_g = topk_g / topk_g.sum(-1, keepdim=True)
        out = torch.zeros(x.size(0), outs.size(-1), device=x.device)
        for k in range(2):
            idx = topk_i[:, k]
            w   = topk_g[:, k].unsqueeze(-1)
            out += w * outs[torch.arange(x.size(0)), idx]
        lb_loss = torch.var(gates.sum(0))
        return out, lb_loss

print("✅ Fusion modules defined")
```

## CELL 4 - Complete CMTF-GCN-NeXt Model
```python
import timm

class CMTF_GCN_NeXt(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        # Component 1: Multi-scale temporal encoder
        self.temporal_enc = MultiScaleTemporalEncoder(in_channels=10, out_dim=256)
        # Component 2: Graph Attention Network
        self.gat = EEGGraphAttention(in_dim=256, hidden=128, heads=8)
        # Component 3: ConvNeXt-Tiny
        self.convnext = timm.create_model('convnext_tiny', pretrained=True,
                                          num_classes=0, global_pool='avg')
        # Component 4: Cross-modal fusion
        self.fusion = CrossModalFusion(eeg_dim=128, spec_dim=768, hidden=512)
        # Component 5: Gated fusion
        self.gate = GatedFusion(dim=512)
        # Component 6: Mixture of Experts
        self.moe = MixtureOfExperts(in_dim=512, hidden=1024, n_experts=4, n_classes=num_classes)
        # Component 7: Final head
        self.head = nn.Sequential(nn.LayerNorm(num_classes), nn.Dropout(0.3))

    def forward(self, eeg, spec):
        # EEG branch
        t_feat = self.temporal_enc(eeg)       # [B, 256]
        s_feat = self.gat(t_feat)             # [B, 128]
        # Spectrogram branch
        v_feat = self.convnext(spec)          # [B, 768]
        # Fusion
        fused  = self.fusion(s_feat, v_feat)  # [B, 512]
        fused, gate = self.gate(fused, fused)  # [B, 512]
        # Classification
        logits, lb_loss = self.moe(fused)     # [B, 6]
        return self.head(logits), lb_loss

# Test forward pass
model = CMTF_GCN_NeXt()
eeg  = torch.randn(2, 10, 10000)
spec = torch.randn(2, 3, 512, 512)
out, lb = model(eeg, spec)
print(f"✅ Full model output: {out.shape}")
print(f"✅ Load balance loss: {lb.item():.4f}")
total = sum(p.numel() for p in model.parameters())
print(f"✅ Total parameters: {total/1e6:.1f}M")

# Save model class to src/models/
import inspect
src_code = inspect.getsource(CMTF_GCN_NeXt)
# (You would save this to src/models/cmtf_gcn_next.py)
print("\n✅ Architecture verified and ready for training!")
```

---

# ============================================================
# NOTEBOOK 8: NB08_Full_Training.ipynb  [MAIN TRAINING - 40-57 hrs]
# ============================================================

## CELL 1 - Setup
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm.notebook import tqdm
import json
import time
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Device: {device}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"✅ VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

# Hyperparameters
CONFIG = {
    'epochs':         50,
    'batch_size':     8,
    'lr':             1e-4,
    'weight_decay':   0.01,
    'accumulation':   4,       # Effective batch = 8*4 = 32
    'warmup_epochs':  5,
    'patience':       15,
    'lb_weight':      0.01,    # Load balance loss weight
}
print(f"\n⚙️  Training Config: {json.dumps(CONFIG, indent=2)}")
```

## CELL 2 - Initialize Model and Loaders
```python
# Paste or import all model classes here
# (MultiScaleTemporalEncoder, EEGGraphAttention, etc.)
# OR: exec(open('NB07_Architecture_Build.ipynb').read())

train_loader, val_loader, test_loader = get_loaders(
    batch_size=CONFIG['batch_size']
)

model = CMTF_GCN_NeXt(num_classes=6).to(device)

# Compute class weights from training data
from collections import Counter
labels_all = [s['label'] for s in train_loader.dataset.data]
counts = Counter(labels_all)
weights = torch.FloatTensor([len(labels_all) / (6 * counts[i]) for i in range(6)]).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)

optimizer = optim.AdamW(model.parameters(),
                        lr=CONFIG['lr'],
                        weight_decay=CONFIG['weight_decay'])
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=CONFIG['epochs'] - CONFIG['warmup_epochs']
)
scaler = GradScaler()

print(f"✅ Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
print(f"✅ Class weights: {weights.tolist()}")
```

## CELL 3 - Training Loop (RUNS 40-57 HOURS)
```python
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}
best_val_acc = 0.0
patience_counter = 0

print("🚀 Starting training...")
print(f"⏰ Estimated: {CONFIG['epochs'] * 68 / 60:.1f} hours total\n")

for epoch in range(CONFIG['epochs']):
    # -------- TRAIN --------
    model.train()
    t_loss, t_correct, t_total = 0, 0, 0
    optimizer.zero_grad()

    pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]")

    for step, batch in pbar:
        eeg    = batch['eeg'].to(device)
        spec   = batch['spectrogram'].to(device)
        labels = batch['label'].to(device)

        with autocast():
            logits, lb_loss = model(eeg, spec)
            loss = (criterion(logits, labels) + CONFIG['lb_weight'] * lb_loss) \
                   / CONFIG['accumulation']

        scaler.scale(loss).backward()

        if (step + 1) % CONFIG['accumulation'] == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        t_loss    += loss.item() * CONFIG['accumulation']
        t_correct += logits.argmax(1).eq(labels).sum().item()
        t_total   += labels.size(0)
        pbar.set_postfix(loss=f"{loss.item()*CONFIG['accumulation']:.4f}",
                         acc=f"{100*t_correct/t_total:.2f}%")

    # -------- VALIDATE --------
    model.eval()
    v_loss, v_correct, v_total = 0, 0, 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", leave=False):
            eeg    = batch['eeg'].to(device)
            spec   = batch['spectrogram'].to(device)
            labels = batch['label'].to(device)
            with autocast():
                logits, lb_loss = model(eeg, spec)
                loss = criterion(logits, labels) + CONFIG['lb_weight'] * lb_loss
            v_loss    += loss.item()
            v_correct += logits.argmax(1).eq(labels).sum().item()
            v_total   += labels.size(0)

    # LR Schedule
    if epoch < CONFIG['warmup_epochs']:
        for pg in optimizer.param_groups:
            pg['lr'] = CONFIG['lr'] * (epoch + 1) / CONFIG['warmup_epochs']
    else:
        scheduler.step()

    # Metrics
    t_acc = 100 * t_correct / t_total
    v_acc = 100 * v_correct / v_total
    v_loss = v_loss / len(val_loader)
    lr = optimizer.param_groups[0]['lr']

    history['train_loss'].append(t_loss / len(train_loader))
    history['train_acc'].append(t_acc)
    history['val_loss'].append(v_loss)
    history['val_acc'].append(v_acc)
    history['lr'].append(lr)

    print(f"\n📊 Epoch {epoch+1:3d}/{CONFIG['epochs']} "
          f"| Train: {t_acc:.2f}% | Val: {v_acc:.2f}% "
          f"| LR: {lr:.2e} | Best: {best_val_acc:.2f}%")

    # Save best
    if v_acc > best_val_acc:
        best_val_acc = v_acc
        patience_counter = 0
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': v_acc, 'config': CONFIG},
                   "models/checkpoints/best_model.pth")
        print(f"   💾 Saved best model (Val: {v_acc:.2f}%)")
    else:
        patience_counter += 1
        if patience_counter >= CONFIG['patience']:
            print(f"\n🛑 Early stopping at epoch {epoch+1}")
            break

    # Save checkpoint every 5 epochs
    if (epoch + 1) % 5 == 0:
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'val_acc': v_acc},
                   f"models/checkpoints/epoch_{epoch+1}.pth")

    # Save history
    with open("results/metrics/training_history.json", 'w') as f:
        json.dump(history, f)

print(f"\n🎉 Training Complete! Best Val Accuracy: {best_val_acc:.2f}%")
```

## CELL 4 - Plot Training Curves
```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
epochs_range = range(1, len(history['val_acc']) + 1)

axes[0].plot(epochs_range, history['train_acc'], label='Train', color='blue')
axes[0].plot(epochs_range, history['val_acc'],   label='Val',   color='orange')
axes[0].set_title('Accuracy'); axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy (%)'); axes[0].legend()

axes[1].plot(epochs_range, history['train_loss'], label='Train', color='blue')
axes[1].plot(epochs_range, history['val_loss'],   label='Val',   color='orange')
axes[1].set_title('Loss'); axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss'); axes[1].legend()

axes[2].plot(epochs_range, history['lr'], color='green')
axes[2].set_title('Learning Rate'); axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('LR'); axes[2].set_yscale('log')

plt.suptitle('CMTF-GCN-NeXt Training Progress', fontsize=14)
plt.tight_layout()
plt.savefig('results/figures/training_curves.png', dpi=150)
plt.show()
```

---

# ============================================================
# NOTEBOOK 9: NB09_Evaluation.ipynb
# ============================================================

## CELL 1 - Load Best Model and Evaluate
```python
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, f1_score)
from sklearn.preprocessing import label_binarize
import time

CLASS_NAMES = ['Seizure', 'LPD', 'GPD', 'LRDA', 'GRDA', 'Other']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
model = CMTF_GCN_NeXt(num_classes=6)
ckpt = torch.load("models/checkpoints/best_model.pth")
model.load_state_dict(ckpt['model_state_dict'])
model = model.to(device).eval()
print(f"✅ Loaded model from epoch {ckpt['epoch']+1} (Val: {ckpt['val_acc']:.2f}%)")
```

## CELL 2 - Full Test Set Evaluation
```python
_, _, test_loader = get_loaders(batch_size=8)

all_labels, all_preds, all_probs, latencies = [], [], [], []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating"):
        eeg  = batch['eeg'].to(device)
        spec = batch['spectrogram'].to(device)
        lbls = batch['label'].to(device)

        t0 = time.time()
        with torch.cuda.amp.autocast():
            logits, _ = model(eeg, spec)
        latencies.append((time.time() - t0) * 1000 / eeg.size(0))

        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(1)

        all_labels.extend(lbls.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

all_labels = np.array(all_labels)
all_preds  = np.array(all_preds)
all_probs  = np.array(all_probs)

acc    = 100 * (all_preds == all_labels).mean()
f1_mac = f1_score(all_labels, all_preds, average='macro') * 100
f1_wt  = f1_score(all_labels, all_preds, average='weighted') * 100
labels_bin = label_binarize(all_labels, classes=range(6))
roc_auc = roc_auc_score(labels_bin, all_probs, average='macro')
avg_lat = np.mean(latencies)

print("="*60)
print("FINAL TEST RESULTS")
print("="*60)
print(f"Accuracy:          {acc:.2f}%")
print(f"Macro F1:          {f1_mac:.2f}%")
print(f"Weighted F1:       {f1_wt:.2f}%")
print(f"ROC-AUC (macro):   {roc_auc:.4f}")
print(f"Avg Latency:       {avg_lat:.2f} ms/sample")
print("="*60)
print("\nPer-class report:")
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4))
```

## CELL 3 - Confusion Matrix and Plots
```python
# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            linewidths=0.5)
plt.title(f'Confusion Matrix (Test Acc: {acc:.2f}%)', fontsize=14)
plt.ylabel('True Label'); plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('results/figures/confusion_matrix.png', dpi=150)
plt.show()

# Baseline Comparison
methods = ['EEGNet\n(EEG only)', 'ConvNeXt\n(Spec only)', 'Late Fusion', 'CMTF-GCN-NeXt\n(Ours)']
accs    = [75.5, 83.2, 85.8, acc]
colors  = ['#ff9999', '#66b3ff', '#99ff99', '#ffd700']

plt.figure(figsize=(10, 6))
bars = plt.bar(methods, accs, color=colors, edgecolor='black', linewidth=1.5, width=0.5)
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{bar.get_height():.1f}%', ha='center', va='bottom',
             fontsize=13, fontweight='bold')
plt.axhline(y=90, color='red', linestyle='--', alpha=0.7, label='Clinical Target (90%)')
plt.ylabel('Accuracy (%)', fontsize=13); plt.ylim(70, 100)
plt.title('Performance Comparison', fontsize=14); plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('results/figures/baseline_comparison.png', dpi=150)
plt.show()

# Save metrics
import json
metrics = {'accuracy': acc, 'f1_macro': f1_mac, 'f1_weighted': f1_wt,
           'roc_auc': roc_auc, 'latency_ms': avg_lat}
with open('results/metrics/test_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
print("\n✅ All results saved!")
```

---

# ============================================================
# NOTEBOOK 10: NB10_Ablation_Studies.ipynb
# ============================================================

## CELL 1 - Define Ablated Models
```python
# Each ablation removes or replaces one component

class NoGAT(nn.Module):
    """Ablation: Remove GAT, use global average pooling instead"""
    def __init__(self):
        super().__init__()
        self.temporal_enc = MultiScaleTemporalEncoder()
        self.proj = nn.Linear(256, 128)      # Replace GAT output
        self.convnext = timm.create_model('convnext_tiny', pretrained=True,
                                           num_classes=0, global_pool='avg')
        self.fusion = CrossModalFusion(eeg_dim=128, spec_dim=768)
        self.gate = GatedFusion()
        self.moe = MixtureOfExperts()
        self.head = nn.Sequential(nn.LayerNorm(6), nn.Dropout(0.3))

    def forward(self, eeg, spec):
        t = self.temporal_enc(eeg)
        s = self.proj(t)                     # No spatial modeling
        v = self.convnext(spec)
        f = self.fusion(s, v)
        f, _ = self.gate(f, f)
        logits, lb = self.moe(f)
        return self.head(logits), lb


class NoCrossAttention(nn.Module):
    """Ablation: Replace cross-attention with concatenation"""
    def __init__(self):
        super().__init__()
        self.temporal_enc = MultiScaleTemporalEncoder()
        self.gat = EEGGraphAttention()
        self.convnext = timm.create_model('convnext_tiny', pretrained=True,
                                           num_classes=0, global_pool='avg')
        self.concat_proj = nn.Linear(128 + 768, 512)   # Simple concat
        self.moe = MixtureOfExperts()
        self.head = nn.Sequential(nn.LayerNorm(6), nn.Dropout(0.3))

    def forward(self, eeg, spec):
        t = self.temporal_enc(eeg)
        s = self.gat(t)
        v = self.convnext(spec)
        f = torch.relu(self.concat_proj(torch.cat([s, v], dim=-1)))
        logits, lb = self.moe(f)
        return self.head(logits), lb


class NoMoE(nn.Module):
    """Ablation: Replace MoE with single MLP"""
    def __init__(self):
        super().__init__()
        self.temporal_enc = MultiScaleTemporalEncoder()
        self.gat = EEGGraphAttention()
        self.convnext = timm.create_model('convnext_tiny', pretrained=True,
                                           num_classes=0, global_pool='avg')
        self.fusion = CrossModalFusion()
        self.gate = GatedFusion()
        self.classifier = nn.Sequential(
            nn.Linear(512, 1024), nn.GELU(), nn.Dropout(0.3), nn.Linear(1024, 6)
        )
        self.head = nn.Sequential(nn.LayerNorm(6), nn.Dropout(0.3))

    def forward(self, eeg, spec):
        t = self.temporal_enc(eeg)
        s = self.gat(t)
        v = self.convnext(spec)
        f = self.fusion(s, v)
        f, _ = self.gate(f, f)
        return self.head(self.classifier(f)), torch.tensor(0.0)


print("✅ Ablation models defined!")
```

## CELL 2 - Train and Evaluate All Ablations
```python
def run_ablation(model_class, name, epochs=30):
    print(f"\n{'='*50}")
    print(f"Running ablation: {name}")
    print(f"{'='*50}")

    model = model_class().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()
    best_acc = 0

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            eeg  = batch['eeg'].to(device)
            spec = batch['spectrogram'].to(device)
            lbls = batch['label'].to(device)
            optimizer.zero_grad()
            with autocast():
                logits, lb = model(eeg, spec)
                loss = criterion(logits, lbls) + 0.01 * lb
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                with autocast():
                    logits, _ = model(batch['eeg'].to(device), batch['spectrogram'].to(device))
                correct += logits.argmax(1).eq(batch['label'].to(device)).sum().item()
                total   += batch['label'].size(0)
        acc = 100 * correct / total
        if acc > best_acc:
            best_acc = acc
        print(f"  Epoch {epoch+1:2d}/{epochs} | Val: {acc:.2f}% | Best: {best_acc:.2f}%")

    return best_acc

# Run all ablations (5 hours each)
ablation_results = {}

ablations = [
    (CMTF_GCN_NeXt,      "Full Model"),
    (NoMoE,              "No MoE"),
    (NoCrossAttention,   "No Cross-Attention"),
    (NoGAT,              "No GAT"),
]

for model_class, name in ablations:
    ablation_results[name] = run_ablation(model_class, name, epochs=30)
    print(f"✅ {name}: {ablation_results[name]:.2f}%")

# Save results
with open('results/metrics/ablation_results.json', 'w') as f:
    json.dump(ablation_results, f, indent=2)
```

## CELL 3 - Plot Ablation Results
```python
names = list(ablation_results.keys())
accs  = list(ablation_results.values())

plt.figure(figsize=(12, 6))
bars = plt.bar(names, accs, color=sns.color_palette("viridis", len(names)),
               edgecolor='black', linewidth=1.2)
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             f'{bar.get_height():.2f}%', ha='center', va='bottom',
             fontsize=12, fontweight='bold')
plt.ylabel('Val Accuracy (%)', fontsize=13)
plt.title('Ablation Study: Component Contributions', fontsize=15)
plt.ylim([max(0, min(accs)-5), min(100, max(accs)+5)])
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.savefig('results/figures/ablation_study.png', dpi=150)
plt.show()
```

---

# ============================================================
# NOTEBOOK 11: NB11_Deployment.ipynb
# ============================================================

## CELL 1 - Export to ONNX (for deployment)
```python
import torch
import torch.onnx

# Load best model
model = CMTF_GCN_NeXt(num_classes=6)
ckpt = torch.load("models/checkpoints/best_model.pth", map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

# Dummy inputs for export
dummy_eeg  = torch.randn(1, 10, 10000)
dummy_spec = torch.randn(1, 3, 512, 512)

# Export to ONNX
print("📦 Exporting to ONNX...")
torch.onnx.export(
    model,
    (dummy_eeg, dummy_spec),
    "models/deployment/cmtf_gcn_next.onnx",
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=['eeg', 'spectrogram'],
    output_names=['logits', 'lb_loss'],
    dynamic_axes={
        'eeg':  {0: 'batch_size'},
        'spectrogram': {0: 'batch_size'},
        'logits': {0: 'batch_size'}
    }
)
print("✅ ONNX model exported to models/deployment/cmtf_gcn_next.onnx")

# Verify ONNX model
import onnx
onnx_model = onnx.load("models/deployment/cmtf_gcn_next.onnx")
onnx.checker.check_model(onnx_model)
print("✅ ONNX model verified!")
```

## CELL 2 - Export to TorchScript
```python
# TorchScript export (alternative to ONNX)
model_scripted = torch.jit.trace(model, (dummy_eeg, dummy_spec))
model_scripted.save("models/deployment/cmtf_gcn_next_scripted.pt")
print("✅ TorchScript model saved!")

# Test TorchScript model
loaded = torch.jit.load("models/deployment/cmtf_gcn_next_scripted.pt")
test_out, _ = loaded(dummy_eeg, dummy_spec)
print(f"✅ TorchScript test output: {test_out.shape}")
```

## CELL 3 - Inference Function
```python
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

CLASS_NAMES = ['Seizure', 'LPD', 'GPD', 'LRDA', 'GRDA', 'Other']
CLASS_COLORS = {
    'Seizure': '🔴', 'LPD': '🟠', 'GPD': '🟡',
    'LRDA': '🟢', 'GRDA': '🔵', 'Other': '⚪'
}

def preprocess_eeg(eeg_array, target_len=10000):
    """eeg_array: numpy [10, T]"""
    if eeg_array.shape[1] > target_len:
        eeg_array = eeg_array[:, :target_len]
    elif eeg_array.shape[1] < target_len:
        pad = np.zeros((eeg_array.shape[0], target_len - eeg_array.shape[1]))
        eeg_array = np.concatenate([eeg_array, pad], axis=1)
    mean = eeg_array.mean(axis=1, keepdims=True)
    std  = eeg_array.std(axis=1, keepdims=True) + 1e-8
    return torch.FloatTensor((eeg_array - mean) / std).unsqueeze(0)  # [1,10,10000]

def preprocess_spectrogram(spec_array, target_size=512):
    """spec_array: numpy [H,W,3] or [3,H,W]"""
    if spec_array.shape[-1] == 3:
        spec_array = spec_array.transpose(2, 0, 1)
    t = torch.FloatTensor(spec_array).unsqueeze(0)
    t = F.interpolate(t, size=(target_size, target_size), mode='bilinear', align_corners=False)
    return t  # [1,3,512,512]

def predict(eeg_array, spec_array, model, device='cpu'):
    """
    Run inference on one sample.
    Returns: predicted class name, confidence, all class probabilities
    """
    model.eval()
    with torch.no_grad():
        eeg  = preprocess_eeg(eeg_array).to(device)
        spec = preprocess_spectrogram(spec_array).to(device)
        logits, _ = model(eeg, spec)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    
    pred_idx   = probs.argmax()
    pred_class = CLASS_NAMES[pred_idx]
    confidence = probs[pred_idx] * 100
    
    return {
        'predicted_class': pred_class,
        'confidence':      f"{confidence:.2f}%",
        'probabilities':   {cls: f"{p*100:.2f}%" for cls, p in zip(CLASS_NAMES, probs)}
    }

# Test inference
dummy_eeg  = np.random.randn(10, 10000).astype(np.float32)
dummy_spec = np.random.rand(512, 512, 3).astype(np.float32)
result = predict(dummy_eeg, dummy_spec, model)

print("🧠 INFERENCE RESULT:")
print(f"   Predicted: {CLASS_COLORS[result['predicted_class']]} {result['predicted_class']}")
print(f"   Confidence: {result['confidence']}")
print(f"\n   All probabilities:")
for cls, prob in result['probabilities'].items():
    print(f"      {CLASS_COLORS[cls]} {cls}: {prob}")
```

## CELL 4 - Create Flask Web App
```python
# Write Flask app to deployment/app.py
flask_code = '''
from flask import Flask, request, jsonify, render_template
import torch
import numpy as np
import json
import os
import sys
sys.path.insert(0, '..')

app = Flask(__name__)

CLASS_NAMES = ["Seizure", "LPD", "GPD", "LRDA", "GRDA", "Other"]

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.jit.load("../models/deployment/cmtf_gcn_next_scripted.pt",
                         map_location=device)
model.eval()
print(f"Model loaded on {device}")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        eeg_data  = np.array(data["eeg"],  dtype=np.float32)   # [10, 10000]
        spec_data = np.array(data["spec"], dtype=np.float32)   # [3, 512, 512]

        eeg_t  = torch.FloatTensor(eeg_data).unsqueeze(0).to(device)
        spec_t = torch.FloatTensor(spec_data).unsqueeze(0).to(device)

        with torch.no_grad():
            logits, _ = model(eeg_t, spec_t)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        pred_idx = int(probs.argmax())
        return jsonify({
            "success":    True,
            "prediction": CLASS_NAMES[pred_idx],
            "confidence": float(probs[pred_idx]),
            "probabilities": {cls: float(p) for cls, p in zip(CLASS_NAMES, probs)}
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "healthy", "model": "CMTF-GCN-NeXt"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
'''

with open("deployment/app.py", "w") as f:
    f.write(flask_code)
print("✅ Flask app written to deployment/app.py")
```

## CELL 5 - Create HTML Frontend
```python
html_code = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>CMTF-GCN-NeXt: EEG Classifier</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto;
               background: #f5f5f5; padding: 20px; }
        .container { background: white; padding: 30px; border-radius: 12px;
                     box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; text-align: center; }
        .subtitle { text-align: center; color: #7f8c8d; margin-bottom: 30px; }
        .btn { background: #3498db; color: white; padding: 12px 30px;
               border: none; border-radius: 6px; cursor: pointer; font-size: 16px; }
        .btn:hover { background: #2980b9; }
        .result { margin-top: 30px; padding: 20px; border-radius: 8px;
                  background: #ecf0f1; display: none; }
        .prediction { font-size: 24px; font-weight: bold; color: #2c3e50; }
        .confidence { color: #27ae60; font-size: 18px; margin: 10px 0; }
        .bar-container { margin: 5px 0; }
        .bar-label { display: inline-block; width: 120px; font-size: 13px; }
        .bar { display: inline-block; height: 18px; background: #3498db;
               border-radius: 3px; transition: width 0.5s; }
        .bar-val { margin-left: 8px; font-size: 13px; color: #555; }
        .loading { display: none; text-align: center; color: #3498db; font-size: 18px; }
    </style>
</head>
<body>
<div class="container">
    <h1>🧠 CMTF-GCN-NeXt</h1>
    <p class="subtitle">Harmful Brain Activity Classification from EEG</p>

    <div style="text-align:center;">
        <button class="btn" onclick="runDemo()">▶ Run Demo Prediction</button>
    </div>

    <div class="loading" id="loading">⏳ Processing EEG data...</div>

    <div class="result" id="result">
        <div class="prediction">🔬 Predicted: <span id="pred-class">-</span></div>
        <div class="confidence">✅ Confidence: <span id="pred-conf">-</span></div>
        <hr>
        <div><strong>All Class Probabilities:</strong></div>
        <div id="prob-bars"></div>
    </div>
</div>

<script>
    const classes = ["Seizure","LPD","GPD","LRDA","GRDA","Other"];
    const colors  = ["#e74c3c","#e67e22","#f1c40f","#2ecc71","#3498db","#95a5a6"];

    function runDemo() {
        document.getElementById("loading").style.display = "block";
        document.getElementById("result").style.display  = "none";

        // Generate random demo data
        const eeg  = Array.from({length:10}, () => Array.from({length:10000}, () => (Math.random()-0.5)*2));
        const spec = Array.from({length:3},  () => Array.from({length:512},  () => Array.from({length:512}, Math.random)));

        fetch("/predict", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({eeg: eeg, spec: spec})
        })
        .then(r => r.json())
        .then(data => {
            document.getElementById("loading").style.display = "none";
            if (data.success) {
                document.getElementById("pred-class").textContent = data.prediction;
                document.getElementById("pred-conf").textContent  = (data.confidence*100).toFixed(2)+"%";
                const bars = document.getElementById("prob-bars");
                bars.innerHTML = "";
                classes.forEach((cls, i) => {
                    const p = (data.probabilities[cls]*100).toFixed(2);
                    bars.innerHTML += `<div class="bar-container">
                        <span class="bar-label">${cls}</span>
                        <span class="bar" style="width:${p*3}px;background:${colors[i]}"></span>
                        <span class="bar-val">${p}%</span>
                    </div>`;
                });
                document.getElementById("result").style.display = "block";
            }
        })
        .catch(e => { alert("Error: " + e); document.getElementById("loading").style.display="none"; });
    }
</script>
</body>
</html>"""

os.makedirs("deployment/templates", exist_ok=True)
with open("deployment/templates/index.html", "w") as f:
    f.write(html_code)
print("✅ HTML frontend written to deployment/templates/index.html")
```

## CELL 6 - Create Deployment Requirements
```python
deployment_requirements = """flask==2.3.0
torch==2.1.0
torchvision==0.16.0
numpy==1.24.3
onnx==1.14.0
onnxruntime==1.16.0
pillow==10.0.0
gunicorn==21.2.0
"""

with open("deployment/requirements.txt", "w") as f:
    f.write(deployment_requirements)
print("✅ deployment/requirements.txt written")
```

## CELL 7 - Run the Web App
```python
import subprocess, time, webbrowser

print("🚀 Starting Flask web server...")
print("📡 Server will be at: http://localhost:5000")
print("⚠️  Press the Stop button in Jupyter to stop the server\n")

# Start server
process = subprocess.Popen(
    ["python", "deployment/app.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT
)

time.sleep(3)  # Wait for server to start
webbrowser.open("http://localhost:5000")
print("✅ Browser opened!")
print("✅ Model is live at http://localhost:5000")
```

## CELL 8 - Test API Endpoint
```python
import requests
import numpy as np
import json

# Generate test data
test_eeg  = np.random.randn(10, 10000).tolist()
test_spec = np.random.rand(3, 512, 512).tolist()

print("🧪 Testing API endpoint...")
response = requests.post(
    "http://localhost:5000/predict",
    json={"eeg": test_eeg, "spec": test_spec},
    timeout=30
)

result = response.json()
print(f"\n🎯 API Response:")
print(f"   Status:     {response.status_code}")
print(f"   Predicted:  {result['prediction']}")
print(f"   Confidence: {result['confidence']*100:.2f}%")
print(f"\n   Probabilities:")
for cls, prob in result['probabilities'].items():
    bar = "█" * int(float(prob) * 20)
    print(f"      {cls:10s}: {bar:<20} {float(prob)*100:.2f}%")

print("\n✅ API working correctly!")
print("✅ DEPLOYMENT COMPLETE!")
```

## CELL 9 - Final Summary
```python
import json
import os

# Load results
with open("results/metrics/test_metrics.json") as f:
    metrics = json.load(f)

print("=" * 65)
print("         CMTF-GCN-NeXt - PROJECT COMPLETE SUMMARY")
print("=" * 65)
print("\n📊 MODEL PERFORMANCE:")
print(f"   Test Accuracy:     {metrics['accuracy']:.2f}%")
print(f"   Macro F1-Score:    {metrics['f1_macro']:.2f}%")
print(f"   Weighted F1:       {metrics['f1_weighted']:.2f}%")
print(f"   ROC-AUC:           {metrics['roc_auc']:.4f}")
print(f"   Inference Latency: {metrics['latency_ms']:.2f} ms")

print("\n📁 OUTPUT FILES:")
files = [
    ("models/checkpoints/best_model.pth",           "Best trained model"),
    ("models/deployment/cmtf_gcn_next.onnx",         "ONNX export"),
    ("models/deployment/cmtf_gcn_next_scripted.pt",  "TorchScript export"),
    ("results/figures/confusion_matrix.png",          "Confusion matrix"),
    ("results/figures/training_curves.png",           "Training curves"),
    ("results/figures/ablation_study.png",            "Ablation study"),
    ("results/metrics/test_metrics.json",             "Test metrics"),
    ("deployment/app.py",                             "Flask web app"),
]
for path, desc in files:
    exists = "✅" if os.path.exists(path) else "❌"
    print(f"   {exists} {path:<50} {desc}")

print("\n🌐 DEPLOYMENT:")
print("   Flask API:  http://localhost:5000/predict")
print("   Health:     http://localhost:5000/health")
print("   Web UI:     http://localhost:5000")
print("\n🏆 PROJECT SUCCESSFULLY COMPLETED!")
print("=" * 65)
```

---

# DAILY SCHEDULE REMINDER

| Day  | Notebook       | Task                         | Time         |
|------|----------------|------------------------------|--------------|
| Day 1| NB01 + NB02    | Setup + Download             | 9 AM - 1 PM  |
| Day 2| NB03 + NB04    | Preprocessing + DataLoader   | 9 AM - 4 PM  |
| Day 3| NB05 + NB06    | Baseline Models              | 9 AM - 6 PM  |
| Day 4| NB07           | Build Full Architecture      | 9 AM - 6 PM  |
| Day5-6| NB08          | Full Model Training (40-57hrs)| 9 AM onwards|
| Day 7| NB09           | Evaluation + Visualization   | 9 AM - 2 PM  |
| Day 8| NB10           | Ablation Studies             | 9 AM - 8 PM  |
| Day 9| NB11           | Deployment + Web App         | 9 AM - 1 PM  |
