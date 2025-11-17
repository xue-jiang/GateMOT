# Multiple Object Tracking by Switching Clues of the Association in Motion-complex Scene

## Abstract

This project presents an efficient multi-object tracking framework designed for motion-complex scenarios. We introduce a Gate Attention Decoder (GAD) that employs learnable gating mechanisms to selectively emphasize discriminative features while maintaining computational efficiency. The decoder achieves O(HW) complexity through element-wise operations and local pooling, significantly more efficient than standard self-attention's O(H²W²) complexity. Our method demonstrates strong performance across multiple MOT benchmarks including MOT17, MOT20, DanceTrack, and SportsMOT.

**Key Features:**
- Gate Attention Decoder for efficient feature fusion
- Real-time performance (30+ FPS on single GPU)
- Support for multiple tracking algorithms (Hungarian, ByteTrack, DeepSORT, etc.)
- The model can be trained on still **image datasets** if videos are not available.

## Main Results

### MOT17 Test Set

| Tracker | MOTA↑ | IDF1↑ | HOTA↑ | FP↓ | FN↓ | IDs↓ | FPS |
|---------|-------|-------|-------|-----|-----|------|-----|
| SwitchTrack + GAD | 60.7 | 62.3 | 52.1 | - | - | - | 32.5 |
| SwitchTrack (baseline) | 59.8 | 61.5 | 51.4 | - | - | - | 34.2 |

### DanceTrack Validation

| Tracker | HOTA↑ | DetA↑ | AssA↑ | MOTA↑ | IDF1↑ |
|---------|-------|-------|-------|-------|-------|
| SwitchTrack + GAD | 46.9 | 51.7 | 43.1 | 65.2 | 60.8 |

### MOT20 Test Set

| Tracker | MOTA↑ | IDF1↑ | HOTA↑ | FPS |
|---------|-------|-------|-------|-----|
| SwitchTrack + GAD | 58.4 | 60.1 | 49.8 | 28.3 |

## Installation

### Requirements
* Python 3.7+
* PyTorch 1.7.0+
* CUDA 10.2+ (for GPU training)

### Step-by-step Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/SwitchTrack.git
cd SwitchTrack-original/添加了wh的switchtrack
```

2. **Create conda environment:**
```bash
conda create -n switchtrack python=3.8
conda activate switchtrack
```

3. **Install PyTorch:**
```bash
# For CUDA 11.3
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

# For CUDA 10.2
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.2 -c pytorch
```

4. **Install dependencies:**
```bash
pip install -r requirements.txt
```

5. **Compile DCNv2 (Deformable Convolution):**
```bash
cd lib/model/networks/DCNv2
python setup.py build develop
cd ../../../..
```

### Dataset Preparation

**MOT17:**
```bash
MOT17/
├── train/
│   ├── MOT17-02-DPM/
│   │   ├── img1/
│   │   └── gt/
│   ├── MOT17-04-DPM/
│   └── ...
└── test/
    └── ...
```

**Generate COCO-format annotations:**
```bash
cd lib/dataset/
python convert_mot_to_coco.py
```

This will generate:
- `annotations/train_half.json` (for training)
- `annotations/val_half.json` (for validation)

**Download pretrained models:**
```bash
# COCO pretrained DLA-34
cd exp/ctdet/coco_dla169_det_only/
wget https://download.pytorch.org/models/dla34-ba72cf86.pth
```

## Training

### MOT17 Half-train

**Basic training:**
```bash
bash train_mot17_wh.sh
```

**Custom training:**
```bash
python train.py \
    --exp_id mot17_half_wh_bfl \
    --arch dla34 \
    --dataset mot \
    --num_epochs 70 \
    --lr 5e-4 \
    --lr_step 60 \
    --batch_size 8 \
    --num_workers 8 \
    --gpus 0,1 \
    --num_classes 1 \
    --input_h 608 \
    --input_w 1088 \
    --num_head_conv 1 \
    --pre_hm \
    --wh \
    --use_bfl \
    --hungarian \
    --custom_dataset_img_path /path/to/MOT17/train \
    --custom_dataset_ann_path /path/to/MOT17/annotations/train_half.json \
    --load_model exp/ctdet/coco_dla169_det_only/dla34-ba72cf86.pth
```

**Key Parameters:**
- `--arch dla34`: Backbone architecture (dla34, dla169, resnet50)
- `--use_bfl`: Enable Gate Attention Decoder
- `--wh`: Use width-height head for bbox prediction
- `--pre_hm`: Use previous heatmap for tracking
- `--hungarian`: Use Hungarian algorithm for data association
- `--num_head_conv 1`: Number of convolutional layers in detection heads

**Training on image datasets:**
For datasets without video sequences, the model can be trained using still images by disabling tracking heads:
```bash
python train.py \
    --dataset coco \
    --no_pre_hm \
    # ... other parameters
```

**Monitor training:**
```bash
tensorboard --logdir=exp/tracking.ctdet/mot17_half_wh_bfl/logs
```

## Tracking

### MOT17 Validation

**Run tracking:**
```bash
bash test_mot17_halfval_wh.sh
```

**Custom tracking:**
```bash
python test.py \
    --exp_id mot17_half_wh_bfl \
    --arch dla34 \
    --load_model exp/tracking.ctdet/mot17_half_wh_bfl/model_60.pth \
    --test_device 0 \
    --num_classes 1 \
    --input_h 608 \
    --input_w 1088 \
    --K 256 \
    --num_head_conv 1 \
    --pre_hm \
    --wh \
    --use_bfl \
    --track_thresh 0.4 \
    --pre_thresh 0.5 \
    --new_thresh 0.4 \
    --hungarian \
    --custom_dataset_img_path /path/to/MOT17/train \
    --custom_dataset_ann_path /path/to/MOT17/annotations/val_half.json
```

**Key Tracking Parameters:**
- `--track_thresh 0.4`: Threshold for tracklet association
- `--pre_thresh 0.5`: Threshold for using previous frame features
- `--new_thresh 0.4`: Threshold for creating new tracklets
- `--K 256`: Maximum number of objects per frame

**Output:**
Results will be saved to: `results/trackval_dc_model_60/`

**Evaluation:**
```bash
python lib/tracking_utils/eval_mot.py \
    --gt_path /path/to/MOT17/train \
    --result_path results/trackval_dc_model_60/
```

### DanceTrack / SportsMOT

Similar to MOT17, adjust dataset paths and thresholds:
```bash
# DanceTrack
bash test_dance.sh

# SportsMOT  
bash test_sports.sh
```

## Demo

### Visualize Tracking Results

**Enable visualization during testing:**
```bash
python test.py \
    --debug 1 \
    --show_track_color \
    # ... other parameters
```

**Output:** Visualization images will be saved to `exp/tracking.ctdet/{exp_id}/debug/`
- `{frame_id}generic.png`: Detection results with color-coded tracking IDs

**Customize visualization** (edit `lib/utils/debugger.py`):
```python
thickness = 10      # Bounding box line width
fontsize = 1.5      # ID label font size
font_thickness = 3  # ID label font thickness
```

### Video Demo

**Track objects in a video:**
```bash
python demo.py \
    --demo /path/to/video.mp4 \
    --load_model exp/tracking.ctdet/mot17_half_wh_bfl/model_60.pth \
    --arch dla34 \
    --use_bfl \
    --track_thresh 0.4 \
    --debug 1 \
    --save_video
```

**Track from webcam:**
```bash
python demo.py \
    --demo webcam \
    --load_model exp/tracking.ctdet/mot17_half_wh_bfl/model_60.pth \
    --debug 1
```

### Visualize BFL Features

**Visualize Gate Attention Decoder outputs:**
```bash
cd srcwh
python visualize_bfl_features.py
```

This will save feature visualizations to `test_bfl_output/`:
- `hm_bfl_feature.png`: Heatmap head features
- `reg_bfl_feature.png`: Offset regression features
- `wh_bfl_feature.png`: Width-height prediction features
- `tracking_bfl_feature.png`: Tracking offset features

## Citation

```bibtex
@article{switchtrack2024,
  title={Multiple Object Tracking by Switching Clues of the Association in Motion-complex Scene},
  author={Your Name},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2024}
}
```

## Acknowledgements

This work is built upon:
- [CenterTrack](https://github.com/xingyizhou/CenterTrack)
- [FairMOT](https://github.com/ifzhang/FairMOT)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
- [DLA](https://github.com/ucbdrive/dla)

## License

This project is released under the MIT License.
