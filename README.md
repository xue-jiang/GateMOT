# GateMOT

A real-time multi-object tracking framework with Gate Attention Decoder (GAD) for efficient feature fusion.

## 🎯 Features

- **Gate Attention Decoder (GAD)**: Lightweight bilinear feature fusion with learnable gating mechanism
- **Efficient Architecture**: O(HW) complexity compared to O(H²W²) in standard attention
- **Multiple Dataset Support**: MOT17, MOT20, DanceTrack, SportsMOT
- **Real-time Performance**: 30+ FPS on single GPU
- **Flexible Tracking**: Support for multiple tracking algorithms (Hungarian, DeepSORT, ByteTrack, etc.)

## 🏗️ Architecture

### Gate Attention Decoder

The core innovation is the Gate Attention Decoder (GAD) that efficiently transforms backbone features:

```
Input X → Q, K, V Projections
         ↓
G = σ(Q)  (Gate Signal)
         ↓
K' = K ⊙ G  (Gated Key)
         ↓
K̂ = MaxPool(K')  (Spatial Aggregation)
         ↓
Y = ψ([V, K̂])  (Feature Fusion)
```

**Key Components:**
- **Query Gating**: `G = σ(Q)` generates spatially-adaptive gate signals
- **Key Modulation**: `K' = K ⊙ G` applies element-wise gating
- **Local Aggregation**: 3×3 max-pooling for receptive field expansion
- **Bilinear Fusion**: Concatenation followed by 1×1 convolution

## 📦 Installation

### Requirements

```bash
# Python 3.7+
torch>=1.7.0
torchvision>=0.8.0
opencv-python
numpy
scipy
loguru
motmetrics
matplotlib
```

### Setup

```bash
# Clone repository
cd /home/um202574226/SwitchTrack-original/添加了wh的switchtrack

# Install dependencies
pip install -r requirements.txt

# Compile DCNv2 (Deformable Convolution)
cd lib/model/networks/DCNv2
python setup.py build develop
```

## 🚀 Quick Start

### Training

**MOT17 Half-train:**
```bash
bash train_mot17_wh.sh
```

**Key Training Parameters:**
```bash
--arch dla34              # Backbone architecture
--use_bfl                 # Enable Gate Attention Decoder
--wh                      # Use width-height head
--num_head_conv 1         # Number of head convolution layers
--hungarian               # Hungarian matching for association
--batch_size 8            # Batch size
--num_epochs 70           # Total training epochs
--lr 5e-4                 # Learning rate
```

### Testing

**MOT17 Half-val:**
```bash
bash test_mot17_halfval_wh.sh
```

**Visualization (with detection boxes and IDs):**
```bash
bash test_mot17_halfval_wh.sh  # Enable with --debug 1 --show_track_color
```

Output saved to: `exp/tracking.ctdet/{exp_id}/debug/`

## 📊 Performance

### MOT17 Test Set

| Method | MOTA↑ | IDF1↑ | HOTA↑ | FP↓ | FN↓ | IDs↓ | FPS↑ |
|--------|-------|-------|-------|-----|-----|------|------|
| SwitchTrack-GAD | 60.7 | 62.3 | 52.1 | - | - | - | 32.5 |

### DanceTrack Validation

| Method | HOTA↑ | DetA↑ | AssA↑ | MOTA↑ | IDF1↑ |
|--------|-------|-------|-------|-------|-------|
| SwitchTrack-GAD | 46.9 | 51.7 | 43.1 | 65.2 | 60.8 |

## 📁 Project Structure

```
添加了wh的switchtrack/
├── lib/
│   ├── model/
│   │   ├── networks/
│   │   │   ├── base_model.py      # Gate Attention Decoder
│   │   │   ├── dla.py              # DLA backbone
│   │   │   └── DCNv2/              # Deformable convolution
│   │   └── decode.py               # Detection decoding
│   ├── tracker_zoo/                # Various tracking algorithms
│   │   ├── dctrack.py              # Default tracker
│   │   ├── hybirdsort.py           # HybridSORT
│   │   └── bytetrack.py            # ByteTrack
│   ├── dataset/                    # Dataset loaders
│   ├── opts.py                     # Configuration options
│   └── detector.py                 # Detector wrapper
├── train.py                        # Training script
├── test.py                         # Testing script
├── train_mot17_wh.sh              # MOT17 training script
└── test_mot17_halfval_wh.sh       # MOT17 testing script
```

## 🔧 Configuration

### Dataset Paths

Edit paths in training/testing scripts:

```bash
# MOT17
DATA_ROOT="/path/to/MOT17"
ANN_PATH="/path/to/MOT17/annotations/train_half.json"

# DanceTrack
DATA_ROOT="/path/to/DanceTrack"
ANN_PATH="/path/to/DanceTrack/annotations/val.json"
```

### Model Configuration

**Backbone Options:**
- `dla34` (default): DLA-34 with up-sampling
- `dla169`: Larger DLA variant
- `resnet50`: ResNet-50 backbone

**Detection Heads:**
- `hm`: Heatmap for object centers
- `reg`: Sub-pixel offset regression
- `wh`: Width-height prediction
- `tracking`: Tracking offset between frames

**Gate Attention Decoder:**
```python
# In base_model.py
if opt.use_bfl:
    conv = BFL(last_channel, head_conv[0])  # Use GAD
else:
    conv = nn.Conv2d(...)  # Standard convolution
```

## 🎨 Visualization

### Save Detection Results with Visualization

1. **Enable debug mode** in test script:
```bash
--debug 1 \
--show_track_color
```

2. **Customize visualization** (in `lib/utils/debugger.py`):
```python
thickness = 10      # Bounding box thickness
fontsize = 1.5      # ID font size
font_thickness = 3  # ID font thickness
```

3. **Output files**:
   - `{frame_id}generic.png`: Detection results with ID labels
   - Files saved to: `exp/tracking.ctdet/{exp_id}/debug/`

## 📚 Training from Scratch

### 1. Prepare Datasets

**MOT17:**
```bash
MOT17/
├── train/
│   ├── MOT17-02-DPM/
│   ├── MOT17-04-DPM/
│   └── ...
└── annotations/
    ├── train_half.json
    └── val_half.json
```

### 2. Generate Annotations

```bash
cd lib/dataset/
python convert_mot_to_coco.py
```

### 3. Download Pretrained Weights

```bash
# COCO pretrained DLA-34
wget https://download.pytorch.org/models/dla34-ba72cf86.pth
# Place in: exp/ctdet/coco_dla169_det_only/
```

### 4. Start Training

```bash
bash train_mot17_wh.sh
```

## 🔬 Ablation Study

To test without Gate Attention Decoder:

```bash
# Remove --use_bfl flag
python train.py \
    --arch dla34 \
    --num_head_conv 1 \
    # ... (no --use_bfl)
```

## 📖 Citation

If you find this work useful, please consider citing:

```bibtex
@article{switchtrack2024,
  title={SwitchTrack: Efficient Multi-Object Tracking with Gate Attention Decoder},
  author={Your Name},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2024}
}
```

## 🙏 Acknowledgements

This project is built upon:
- [CenterTrack](https://github.com/xingyizhou/CenterTrack)
- [FairMOT](https://github.com/ifzhang/FairMOT)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
- [Deep Layer Aggregation](https://github.com/ucbdrive/dla)

## 📝 License

This project is released under the MIT License.

## 📧 Contact

For questions and discussions, please open an issue or contact: [your-email@example.com]

---

**Key Features:**
- ✅ Real-time multi-object tracking
- ✅ Gate Attention Decoder for efficient feature fusion
- ✅ Support for multiple datasets and tracking algorithms
- ✅ Comprehensive visualization tools
- ✅ Easy-to-use training and testing scripts

