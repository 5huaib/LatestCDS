# 🛡️ AI Crime Detection System - Spatio-Temporal Pipeline

A modular, real-time Computer Vision system for **Human Detection**, **Persistent Multi-Object Tracking**, and **Spatio-Temporal Threat / Crime Assessment** from surveillance video feeds.

Built with **YOLOv8**, **DeepSORT**, and a **3D ResNet-18 (3D-CNN)** video classification architecture.

---

## 🏗️ System Architecture

```mermaid
graph LR
    A[Video Stream / CCTV Feed] --> B[YOLOv8 Human Localization]
    B --> C[DeepSORT Multi-Object Tracking]
    C --> D[16-Frame Temporal Crop Buffer]
    D --> E[ResNet3D-18 Spatio-Temporal Classifier]
    E --> F[Visual Overlays & Threat Gauges]
    F --> G[Annotated Video in outputs/]
    F --> H[Incident JSON Alerts Log]
```

### 3-Stage Pipeline:
1. **Human Detection (YOLOv8)**: Detects individuals in each frame (`class 0: person`).
2. **Persistent Tracking (DeepSORT)**: Assigns persistent track IDs across occlusions, camera jitter, and crossings using Kalman filtering and deep cosine Re-ID embeddings.
3. **Spatio-Temporal Anomaly Classification (3D-CNN)**: Evaluates a 16-frame sliding window crop per tracked person using a 3D ResNet backbone (`r3d_18`) with a binary classification head to produce an anomaly/crime probability score.

---

## 📂 Project Organization

```text
crime_detection_system/
├── inputs/                       # 📥 DROP YOUR INPUT VIDEOS HERE (.mp4, .avi, .mov, etc.)
│   └── README.md
├── outputs/                      # 📤 PROCESSED RESULTS SAVED HERE (.mp4 videos & .json alert logs)
│   └── README.md
├── weights/                      # 🧠 MODEL WEIGHTS DIRECTORY
│   ├── best_crime_detector.pth   # Trained 3D-CNN weights (generated after training)
│   ├── yolov8n.pt                # YOLOv8 person detector (auto-downloaded)
│   └── README.md
├── scripts/
│   └── verify_dataset.py         # 🔍 Diagnostic tool to verify your 91 GB UCF-Crime dataset
├── detect.py                     # 🚀 Main end-to-end inference script
├── train.py                      # 🏋️ High-performance training script (NVIDIA RTX A5000 / CUDA)
├── model.py                      # 🧠 CrimeDetector3D neural network architecture
├── pipeline_prototype.py         # 🧪 Legacy pipeline test runner
├── requirements.txt              # 📦 Python dependencies
└── README.md                     # 📖 Complete documentation & guide
```

---

## 🔄 MacBook ➡️ GitHub ➡️ College Workstation Workflow

### Part 1: Push from your MacBook to GitHub

On your MacBook, commit the clean codebase and push to GitHub:

```bash
# 1. Review status (loose test videos and large weights are ignored automatically)
git status

# 2. Stage all clean code files
git add .

# 3. Commit changes
git commit -m "End-to-end crime detection ready: inputs/outputs structure, RTX A5000 training engine, and documentation"

# 4. Push to your GitHub repository
git push origin main
```

> **Note**: Deep learning weight files (`.pth`, `.pt`) and heavy video files are excluded by `.gitignore` so your Git repository remains clean, lightweight, and never hits GitHub's 100 MB file limit.

---

### Part 2: Pull and Set Up on College System (NVIDIA RTX A5000)

On your college computer (equipped with **NVIDIA RTX A5000 24GB GPU**):

#### 1. Clone the repository:
```bash
git clone https://github.com/5huaib/LatestCDS.git
cd LatestCDS
```

#### 2. Create and activate a virtual environment:
```bash
# On Linux:
python3 -m venv venv
source venv/bin/activate

# On Windows:
python -m venv venv
venv\Scripts\activate
```

#### 3. Install PyTorch with CUDA support:
Ensure PyTorch is installed with CUDA acceleration for your RTX A5000:
```bash
# For CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Or for CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Verify that CUDA and your RTX A5000 are recognized:
```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU Device:', torch.cuda.get_device_name(0))"
```
*(Should output `CUDA Available: True` and `GPU Device: NVIDIA RTX A5000`)*

#### 4. Install remaining dependencies:
```bash
pip install -r requirements.txt
```

---

## 🏋️ Training Guide: 91 GB UCF-Crime Dataset (RTX A5000)

The **UCF-Crime dataset** is ~91 GB and contains 1,900 untrimmed surveillance videos across 14 categories (13 crime categories + `NormalVideos`).

### Step 1: Verify the Dataset
Before starting training, run the diagnostic utility to verify your dataset folder structure and video codecs:

```bash
python scripts/verify_dataset.py --dataset-dir /path/to/UCF-Crime
```

This will print:
- Total number of videos and detected categories.
- Label assignments (`Normal (0)` vs `Crime (1)`).
- Frame dimensions, FPS, and test video decoding check.

---

### Step 2: Run Training on the RTX A5000

Run `train.py` with the parameters optimized for your **RTX A5000 (24 GB VRAM)**:

```bash
python train.py \
  --dataset-dir /path/to/UCF-Crime \
  --batch-size 32 \
  --num-workers 8 \
  --epochs 10 \
  --lr 1e-4 \
  --stride 32 \
  --clip-sample-stride 2 \
  --save-dir weights
```

### Why these settings are optimal for RTX A5000:
| Parameter | Value | Reason for RTX A5000 |
| :--- | :--- | :--- |
| `--batch-size` | `32` (or `64`) | The 24 GB VRAM easily accommodates large batches of `(32, 3, 16, 112, 112)` video clips. |
| Mixed Precision (`AMP`) | Automatic | Leverages 3rd-Gen Ampere Tensor Cores for **2x–3x faster training throughput** and cuts memory usage. |
| `--num-workers` | `8` | Multi-threaded CPU clip decoding prevents GPU data starvation. |
| `--stride` | `32` | Samples sequence start frames without redundant overlapping frames across long videos. |
| `--clip-sample-stride` | `2` | Samples every 2nd frame in a 16-frame window to capture larger temporal action context. |

### Checkpoints and Resumption:
- The optimal model is automatically saved to `weights/best_crime_detector.pth`.
- An epoch checkpoint is saved to `weights/last_checkpoint.pth`.
- If training is interrupted, you can resume seamlessly:
  ```bash
  python train.py --dataset-dir /path/to/UCF-Crime --resume weights/last_checkpoint.pth
  ```

---

## 🚀 Running End-to-End Inference

### Where to put your video:
Drop any video clip you want to test into the `inputs/` folder:
```bash
cp /path/to/my_cctv_footage.mp4 inputs/
```

### Run Crime Detection:
```bash
# Option 1: Automatic (detects videos in inputs/ automatically)
python detect.py

# Option 2: Specify input video explicitly
python detect.py --input inputs/my_cctv_footage.mp4

# Option 3: Process live webcam
python detect.py --input 0

# Option 4: Process live RTSP CCTV camera feed
python detect.py --input "rtsp://username:password@192.168.1.100:554/stream1"
```

### Where to get your output:
All results are automatically saved in `outputs/`:
1. **Annotated Video**: `outputs/detected_<video_name>.mp4`
   - Bounding boxes around people with tracking IDs.
   - Dynamic threat score meters.
   - Flashing red banner: `🚨 ALERT: CRIME / SUSPICIOUS BEHAVIOR DETECTED`.
   - Real-time telemetry HUD (FPS, active tracks, device, threat count).
2. **Incident Alert Summary**: `outputs/<video_name>_alerts.json`
   - Formatted JSON log detailing every detected crime event, exact timestamps (`MM:SS`), suspect track IDs, and peak threat scores.

---

## 🎛️ Command-Line Flags Reference

### `detect.py`
| Flag | Default | Description |
| :--- | :--- | :--- |
| `--input` | `""` | Video path, webcam index (`0`), or RTSP stream. If omitted, scans `inputs/`. |
| `--output` | `""` | Explicit output video path. Defaults to `outputs/detected_<name>.mp4`. |
| `--output-dir`| `outputs` | Folder to save output videos and incident reports. |
| `--weights` | `""` | Path to fine-tuned weights file. Defaults to `weights/best_crime_detector.pth`. |
| `--threshold`| `0.20` | Threat score threshold to trigger high crime alert. |
| `--frame-stride`| `2` | Frame sampling stride in the 16-frame buffer. |
| `--eval-interval`| `16` | Frequency of 3D-CNN temporal evaluations in frames. |
| `--no-display` | `False` | Run headlessly without popping up an OpenCV GUI window. |

### `train.py`
| Flag | Default | Description |
| :--- | :--- | :--- |
| `--dataset-dir` | `./dataset` | Path to UCF-Crime dataset directory. |
| `--batch-size` | `32` | Batch size (32 or 64 recommended for 24GB RTX A5000). |
| `--epochs` | `10` | Number of training epochs. |
| `--lr` | `1e-4` | Learning rate for AdamW optimizer. |
| `--num-workers`| `8` | DataLoader worker threads. |
| `--save-dir` | `weights` | Directory where trained checkpoints are saved. |
| `--resume` | `""` | Path to checkpoint file to resume training. |
| `--no-amp` | `False` | Disable Automatic Mixed Precision. |

---

## 📜 License
MIT License
