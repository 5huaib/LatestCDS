# Model Weights Directory

This folder holds the deep learning model weights used for detection and classification.

### Files:
- `best_crime_detector.pth`: The trained 3D-CNN (ResNet3D-18) weights file generated after training on the college workstation.
- `yolov8n.pt`: YOLOv8 object detector weights for human localization (automatically downloaded by Ultralytics on first run if not present).

> **Note on GitHub / File Sizes**:
> Deep learning weights (`.pt`, `.pth`) exceed GitHub's 100 MB file limit and are ignored by `.gitignore`.
> When you run `train.py` on your college machine, the best model will automatically be saved to `weights/best_crime_detector.pth`.
> If you run `detect.py` without fine-tuned weights, it automatically falls back to pre-trained torchvision `R3D_18` feature extractors so the pipeline is always functional out of the box!
