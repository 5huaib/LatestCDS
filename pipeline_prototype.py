#!/usr/bin/env python3
import os
import sys
import math
import time
import argparse
import collections
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.models.video as video_models
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Set seed for reproducibility of the classification head outputs
torch.manual_seed(42)

class CrimeDetector3D(nn.Module):
    """
    3D-CNN temporal feature extractor and binary classification head.
    Uses ResNet3D-18 as a backbone, removes the original fc layer,
    and appends: FC (1024) -> ReLU -> Dropout(0.5) -> FC (512) -> ReLU -> Dropout(0.5) -> FC (1) -> Sigmoid.
    """
    def __init__(self, use_pretrained=True, weights_path=None):
        super().__init__()
        print("Initializing 3D-CNN backbone (ResNet3D-18)...")
        
        # Load backbone
        try:
            if use_pretrained:
                # Modern torchvision versions (0.14+)
                try:
                    weights = video_models.R3D_18_Weights.DEFAULT
                    self.backbone = video_models.r3d_18(weights=weights)
                    print("Successfully loaded pre-trained R3D-18 weights.")
                except AttributeError:
                    # Legacy torchvision versions
                    self.backbone = video_models.r3d_18(pretrained=True)
                    print("Successfully loaded pre-trained R3D-18 weights (legacy API).")
            else:
                self.backbone = video_models.r3d_18(pretrained=False)
                print("Initialized R3D-18 backbone with random weights.")
        except Exception as e:
            print(f"Warning: Failed to load pre-trained weights ({e}). Initializing backbone with random weights.")
            self.backbone = video_models.r3d_18(pretrained=False)

        # Extract features and replace original linear classifier with Identity
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Binary classification head: FC layers 1024 -> 512 -> 1
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

        if weights_path and os.path.exists(weights_path):
            print(f"Loading custom fine-tuned weights from '{weights_path}'...")
            checkpoint = torch.load(weights_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.load_state_dict(checkpoint)
            print("Successfully loaded fine-tuned model weights.")

    def forward(self, x):
        # x shape: (batch_size, channels, time, height, width) -> e.g. (1, 3, 16, 112, 112)
        features = self.backbone(x) # Shape: (batch_size, 512)
        output = self.classifier(features) # Shape: (batch_size, 1)
        return output


def generate_demo_video(filename="demo_video.mp4", num_frames=150, width=640, height=480):
    """
    Generates a synthetic demo video with moving textured blocks to simulate people.
    This ensures the pipeline can run out of the box.
    """
    print(f"Generating synthetic video: '{filename}'...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, 30.0, (width, height))
    
    for i in range(num_frames):
        # Dark grey background
        frame = np.ones((height, width, 3), dtype=np.uint8) * 20
        
        # Draw background grid lines
        for y in range(0, height, 40):
            cv2.line(frame, (0, y), (width, y), (40, 40, 40), 1)
        for x in range(0, width, 40):
            cv2.line(frame, (x, 0), (x, height), (40, 40, 40), 1)
            
        # Draw simulated Subject 1 (Person 1 - Green rectangle representation)
        w1, h1 = 80, 180
        px1 = int(50 + (i * 3) % (width - w1 - 100))
        py1 = int(150 + int(15 * math.sin(i * 0.15)))
        
        # Subject 1 graphics
        cv2.rectangle(frame, (px1, py1), (px1 + w1, py1 + h1), (60, 160, 60), -1) # Body
        cv2.circle(frame, (px1 + w1//2, py1 + 30), 20, (180, 220, 180), -1) # Head
        cv2.rectangle(frame, (px1 + 10, py1 + 60), (px1 + w1 - 10, py1 + h1 - 10), (120, 120, 240), -1) # Shirt
        cv2.putText(frame, "SIMULATED OBJ_0", (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 255, 100), 1)

        # Draw simulated Subject 2 (Person 2 - Blue rectangle representation, moving in opposite direction)
        w2, h2 = 70, 160
        px2 = int(width - 150 - (i * 2.5) % (width - w2 - 100))
        py2 = int(180 + int(10 * math.cos(i * 0.2)))
        
        # Subject 2 graphics
        cv2.rectangle(frame, (px2, py2), (px2 + w2, py2 + h2), (160, 60, 60), -1) # Body
        cv2.circle(frame, (px2 + w2//2, py2 + 25), 18, (220, 180, 180), -1) # Head
        cv2.rectangle(frame, (px2 + 10, py2 + 55), (px2 + w2 - 10, py2 + h2 - 10), (240, 120, 120), -1) # Shirt
        cv2.putText(frame, "SIMULATED OBJ_1", (px2, py2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 255), 1)

        # Add overall watermark
        cv2.putText(frame, "AI CRIME DETECTION SYSTEM - PIPELINE PROTOTYPE", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.putText(frame, f"Frame {i+1}/{num_frames}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        out.write(frame)
        
    out.release()
    print("Synthetic video generation completed.")


def main():
    parser = argparse.ArgumentParser(description="CV-Based AI Crime Detection Pipeline Prototype")
    parser.add_argument("--input", type=str, default="", help="Path to input video file or RTSP stream URL. If empty, runs in demo mode.")
    parser.add_argument("--demo", action="store_true", help="Generate and run a synthetic demo video automatically.")
    parser.add_argument("--output", type=str, default="output.mp4", help="Path to save output video (default: output.mp4).")
    parser.add_argument("--no-display", action="store_true", help="Run headlessly without showing cv2.imshow GUI.")
    parser.add_argument("--yolo-model", type=str, default="yolov8n.pt", help="YOLOv8 weights file (default: yolov8n.pt).")
    parser.add_argument("--eval-interval", type=int, default=16, help="3D-CNN evaluation interval in frames (default: 16).")
    parser.add_argument("--max-frames", type=int, default=0, help="Maximum number of frames to process. 0 means process the whole video.")
    parser.add_argument("--weights", type=str, default="", help="Path to fine-tuned weights file (e.g., best_crime_detector.pth).")
    parser.add_argument("--frame-stride", type=int, default=2, help="Sampling stride between frames stored in temporal buffer (default: 2).")
    parser.add_argument("--threshold", type=float, default=0.2, help="Threat score threshold to trigger SUSPICIOUS alert (default: 0.2).")
    args = parser.parse_args()

    # Determine hardware device
    device = torch.device('cuda' if torch.cuda.is_available() else 
                          ('mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"Using execution device: {device}")

    # Set up input video
    input_source = args.input
    is_demo_mode = args.demo or (not args.input)
    
    if is_demo_mode:
        input_source = "demo_video.mp4"
        if not os.path.exists(input_source):
            generate_demo_video(input_source)
        print("Running in DEMO/SIMULATION mode.")
    else:
        print(f"Running in Standard Mode. Video Source: {input_source}")

    # Initialize video capture
    cap = cv2.VideoCapture(input_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source '{input_source}'.")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video Properties: Resolution={width}x{height}, FPS={fps:.2f}, Total Frames={total_frames}")

    # Initialize Video Writer
    out_writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
        print(f"Writing annotated video to: {args.output}")

    # Initialize YOLOv8 Model (Person class ID is 0)
    print(f"Loading YOLOv8 Model: {args.yolo_model}...")
    yolo_model = YOLO(args.yolo_model)
    
    # Initialize DeepSORT Tracker
    print("Loading DeepSORT Tracker...")
    tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0, max_cosine_distance=0.3)

    # Initialize custom 3D-CNN Anomaly Detector
    weights_path = args.weights if args.weights else None
    classifier = CrimeDetector3D(use_pretrained=True, weights_path=weights_path).to(device)
    classifier.eval()

    # Sliding window queue for temporal features per track ID
    # key: track_id, value: collections.deque(maxlen=16) containing 3x112x112 image tensors
    track_buffers = {}
    track_frame_counts = {}
    track_threat_scores = {}
    
    # Image preprocessing statistics (ImageNet)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)

    frame_idx = 0
    start_time = time.time()

    try:
        while cap.isOpened():
            if args.max_frames > 0 and frame_idx >= args.max_frames:
                print(f"\nReached max-frames limit of {args.max_frames}. Stopping.")
                break
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            
            # 1. Run YOLOv8 detection
            results = yolo_model(frame, device=device, verbose=False)
            
            # 2. Filter for "person" class (ID 0) and confidence >= 0.5
            detections = []
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    if cls == 0 and conf >= 0.5:
                        xmin, ymin, xmax, ymax = box.xyxy[0].tolist()
                        w = xmax - xmin
                        h = ymax - ymin
                        detections.append(([xmin, ymin, w, h], conf, cls))

            # 3. Demo Mode Fallback: Inject simulated detections if YOLO doesn't detect the synthetic body shapes
            if is_demo_mode and len(detections) == 0:
                # Simulating two people moving across the frame matching the generated video graphics
                # Subject 1
                w1, h1 = 80, 180
                px1 = float(50 + (frame_idx * 3) % (width - w1 - 100))
                py1 = float(150 + int(15 * math.sin(frame_idx * 0.15)))
                detections.append(([px1, py1, float(w1), float(h1)], 0.95, 0))
                
                # Subject 2
                w2, h2 = 70, 160
                px2 = float(width - 150 - (frame_idx * 2.5) % (width - w2 - 100))
                py2 = float(180 + int(10 * math.cos(frame_idx * 0.2)))
                detections.append(([px2, py2, float(w2), float(h2)], 0.92, 0))

            # 4. Update DeepSORT tracks
            tracks = tracker.update_tracks(detections, frame=frame)
            
            active_track_ids = set()
            
            # 5. Process confirmed tracks
            for track in tracks:
                if not track.is_confirmed():
                    continue
                
                track_id = track.track_id
                active_track_ids.add(track_id)
                
                # Extract left-top-right-bottom coordinates
                ltrb = track.to_ltrb()
                x1, y1, x2, y2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
                
                # Apply boundaries
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                
                crop_w = x2 - x1
                crop_h = y2 - y1
                
                threat_score = None
                buffer_status = "0/16"
                
                # Only extract sequence if crop is valid
                if crop_w > 0 and crop_h > 0:
                    crop = frame[y1:y2, x1:x2]
                    
                    # Convert to RGB, resize to 112x112
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    crop_resized = cv2.resize(crop_rgb, (112, 112))
                    
                    # Convert to Tensor (C, H, W) normalized to [0, 1]
                    crop_tensor = torch.tensor(crop_resized, dtype=torch.float32).permute(2, 0, 1).to(device) / 255.0
                    # Apply ImageNet normalization
                    crop_tensor = (crop_tensor - mean) / std
                    
                    # Store crop in track buffer with frame-stride sampling
                    if track_id not in track_buffers:
                        track_buffers[track_id] = collections.deque(maxlen=16)
                        track_frame_counts[track_id] = 0
                    
                    track_frame_counts[track_id] += 1
                    if track_frame_counts[track_id] % args.frame_stride == 0:
                        track_buffers[track_id].append(crop_tensor)
                    
                    buffer_len = len(track_buffers[track_id])
                    buffer_status = f"{buffer_len}/16"
                    
                    # 6. Run 3D-CNN evaluation once 16 frames are accumulated, and then every eval-interval
                    if buffer_len == 16:
                        if ((track_frame_counts[track_id] // args.frame_stride) >= 16) and (((track_frame_counts[track_id] // args.frame_stride) - 16) % args.eval_interval == 0):
                            # Stack elements along time dimension (dim=1) to create shape (C, T, H, W)
                            sequence = torch.stack(list(track_buffers[track_id]), dim=1)
                            # Add Batch dimension -> (1, C, T, H, W)
                            sequence = sequence.unsqueeze(0)
                            
                            # Inference
                            with torch.no_grad():
                                output = classifier(sequence)
                                threat_score = output.item()
                                track_threat_scores[track_id] = threat_score
                        else:
                            threat_score = track_threat_scores.get(track_id, None)
                
                # 7. Visualization on frame
                # Green for normal, red for suspicious (threat_score >= threshold)
                color = (0, 255, 0)
                label = f"ID: {track_id} | Buffering: {buffer_status}"
                
                if threat_score is not None:
                    is_suspicious = threat_score >= args.threshold
                    color = (0, 0, 255) if is_suspicious else (0, 255, 0)
                    status_text = "SUSPICIOUS" if is_suspicious else "Normal"
                    label = f"ID: {track_id} | {status_text}: {threat_score:.4f}"
                    
                    # Only print to console when threat score is updated to avoid spamming
                    if (track_frame_counts[track_id] - 16) % args.eval_interval == 0:
                        print(f"[Frame {frame_idx:03d}] Track {track_id}: Threat Score = {threat_score:.4f} -> {status_text}")
                else:
                    if frame_idx % 30 == 0:
                        print(f"[Frame {frame_idx:03d}] Track {track_id}: Buffering ({buffer_status})")
                
                # Draw bounding box and tracking info
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, max(15, y1 - 8)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Cleanup track buffers for tracking IDs that are no longer active
            for tid in list(track_buffers.keys()):
                if tid not in active_track_ids:
                    del track_buffers[tid]
                    if tid in track_frame_counts:
                        del track_frame_counts[tid]
                    if tid in track_threat_scores:
                        del track_threat_scores[tid]

            # Write annotated frame
            if out_writer is not None:
                out_writer.write(frame)

            # Show window GUI if display is enabled
            if not args.no_display:
                cv2.imshow("CV-Based AI Crime Detection System Prototype", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Execution interrupted by user.")
                    break
                    
    except KeyboardInterrupt:
        print("Execution interrupted by keyboard.")
    finally:
        cap.release()
        if out_writer is not None:
            out_writer.release()
        cv2.destroyAllWindows()
        
        elapsed = time.time() - start_time
        fps_actual = frame_idx / elapsed if elapsed > 0 else 0
        print(f"\nProcessing summary:")
        print(f"- Total processed frames: {frame_idx}")
        print(f"- Execution time: {elapsed:.2f} seconds")
        print(f"- Average Processing Speed: {fps_actual:.2f} FPS")
        print("Pipeline execution completed successfully.")

if __name__ == "__main__":
    main()
