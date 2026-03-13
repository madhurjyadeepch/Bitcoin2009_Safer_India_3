# Fight Detection — Training & Improvement Guide

## Why Is The Current Model Inaccurate?

The current fight detection is **rule-based** (heuristic), not trained on fight data. It uses:
- Proximity between two persons
- Wrist velocity (arm movement speed)
- Arm extension angles

This approach gives many **false positives** (waving, dancing, stretching) and **false negatives** (subtle fights at a distance). To improve accuracy, you need a **trained ML model**.

---

## Recommended Approach: LSTM or Transformer on Pose Sequences

### Architecture

```
Video Frames
    ↓
YOLOv8 (person detection) ← already built
    ↓
MediaPipe (pose landmarks) ← already built
    ↓
Feature extraction (joint angles, velocities, distances)
    ↓
Sequence model (LSTM / Transformer) ← TRAIN THIS
    ↓
Fight / No-Fight classification
```

The idea: instead of hand-crafted thresholds, you feed **sequences of pose features** into a trained classifier.

---

## Step-by-Step Training Plan

### 1. Collect Training Data

**Option A: Public Datasets**
| Dataset | Description | Link |
|---------|-------------|------|
| **RWF-2000** | 2000 fight/non-fight clips from surveillance | [GitHub](https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection) |
| **Hockey Fight** | 1000 hockey game clips (fight/non-fight) | Search "Hockey Fight Detection Dataset" |
| **UCF Crime** | Real-world anomaly detection dataset | [UCF CRCV](https://www.crcv.ucf.edu/projects/real-world/) |

**Option B: Record Your Own**
- Use this system to record clips from your CCTV
- Label clips as `fight` or `normal`
- Aim for at least 200 clips per class

### 2. Extract Pose Features

Create a script that processes each video clip:

```python
# scripts/extract_features.py (create this)
import cv2
import numpy as np
from detector import PersonDetector
from pose_estimator import PoseEstimator

def extract_clip_features(video_path, max_frames=90):
    """Extract pose features from a video clip."""
    cap = cv2.VideoCapture(video_path)
    detector = PersonDetector()
    estimator = PoseEstimator()
    
    features_sequence = []
    
    while cap.isOpened() and len(features_sequence) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        tracked, _ = detector.detect(frame)
        poses = estimator.estimate(frame, tracked)
        
        # Extract per-frame feature vector
        frame_features = []
        for pid, pose in poses.items():
            angles = pose["angles"]
            velocities = pose["velocities"]
            tilt = pose["body_tilt"]
            
            vec = [
                angles.get("left_elbow", 0),
                angles.get("right_elbow", 0),
                angles.get("left_shoulder", 0),
                angles.get("right_shoulder", 0),
                angles.get("left_knee", 0),
                angles.get("right_knee", 0),
                velocities.get("left_wrist", 0),
                velocities.get("right_wrist", 0),
                velocities.get("left_ankle", 0),
                velocities.get("right_ankle", 0),
                tilt,
            ]
            frame_features.append(vec)
        
        # Average across persons or pad
        if frame_features:
            features_sequence.append(np.mean(frame_features, axis=0).tolist())
    
    cap.release()
    
    # Pad/truncate to fixed length
    while len(features_sequence) < max_frames:
        features_sequence.append([0.0] * 11)
    
    return np.array(features_sequence[:max_frames])
```

### 3. Train an LSTM Classifier

```python
# scripts/train_fight_model.py (create this)
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class FightDataset(Dataset):
    def __init__(self, features_dir, label_file):
        """
        features_dir: directory with .npy feature files
        label_file: CSV with filename,label (1=fight, 0=normal)
        """
        self.samples = []
        import csv
        with open(label_file) as f:
            for row in csv.DictReader(f):
                feat_path = os.path.join(features_dir, row["filename"] + ".npy")
                if os.path.exists(feat_path):
                    self.samples.append((feat_path, int(row["label"])))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        features = np.load(path).astype(np.float32)
        return torch.tensor(features), torch.tensor(label, dtype=torch.long)


class FightLSTM(nn.Module):
    def __init__(self, input_size=11, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2),
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, (h_n, _) = self.lstm(x)
        # Use last hidden state
        out = self.classifier(h_n[-1])
        return out


def train():
    dataset = FightDataset("data/features", "data/labels.csv")
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    model = FightLSTM()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(50):
        total_loss = 0
        correct = 0
        total = 0
        
        for features, labels in loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        acc = correct / total * 100
        print(f"Epoch {epoch+1}/50 — Loss: {total_loss:.4f} — Acc: {acc:.1f}%")
    
    # Save model
    torch.save(model.state_dict(), "fight_lstm.pth")
    print("Model saved to fight_lstm.pth")


if __name__ == "__main__":
    train()
```

### 4. Replace the Rule-Based Detector

Once trained, update `analyzers/fight_detector.py` to load and use the LSTM model:

```python
# In fight_detector.py, replace the analyze() method:
class FightDetector:
    def __init__(self):
        self.model = FightLSTM()
        self.model.load_state_dict(torch.load("fight_lstm.pth"))
        self.model.eval()
        self._feature_buffer = defaultdict(lambda: deque(maxlen=90))
    
    def analyze(self, tracked_persons, poses):
        # Collect features from poses
        # Feed through LSTM
        # Return events if classification > threshold
        ...
```

---

## Quick Improvements Without Training

If you can't train a model yet, tune these values in `config.py`:

| Parameter | Current | Suggested | Why |
|-----------|---------|-----------|-----|
| `FIGHT_PROXIMITY_THRESHOLD` | 150 | 100 | Reduces false positives from people just walking near each other |
| `FIGHT_WRIST_VELOCITY_THRESHOLD` | 30 | 45 | Filters out normal arm movement |
| `FIGHT_CONFIDENCE_THRESHOLD` | 0.6 | 0.75 | Requires stronger evidence |
| `FIGHT_FRAME_WINDOW` | 15 | 20 | More temporal smoothing |

---

## Resources

- [MediaPipe Pose Landmarks](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) — landmark indices reference
- [PyTorch LSTM Tutorial](https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html)
- [RWF-2000 Paper](https://arxiv.org/abs/1911.05913) — violence detection benchmark
