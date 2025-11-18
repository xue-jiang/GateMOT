# Mutiple Object Tracking by Switching Clues of the Association in Motion-complex Scene

---

## Abstract

This repository implements an online multiple object tracking (MOT) framework that uses a **Q-Gated Linear Attention (Q-Attention)** decoder to jointly perform detection, motion estimation, and ReID on a shared feature map.

Instead of standard self-attention with quadratic complexity, **Q-Attention turns the Query into a spatial gate** that directly filters Key features and fuses them with the original Value features in a **linear-complexity** way. Different Q-Attention heads are used for different tasks (detection / motion / ReID), which allows:

- Task-specific feature selection while keeping the backbone **fully shared**.
- Robust association by **switching the effective clues** between motion and appearance depending on confidence and occlusion.
- High efficiency in dense, motion-complex scenes where classical self-attention is too expensive.

### Key features:

- **Q-Gated Linear Attention decoder** for dense detection / motion / ReID heads.
- Multi-head Q-Attention: **one head per task**, with task-adaptive spatial gates.
- Association strategy that implicitly **switches clues** between motion and appearance via confidence-aware fusion.
- The model can be trained on still **image datasets** if videos are not available.
- Ready for standard MOT benchmarks (e.g. MOT17, DanceTrack, SportsMOT) and custom datasets (BEE24).

---

## Main Results

| Dataset     | HOTA | MOTA | IDF1 |  IDs | AssA | DetA |
|:------------|:----:|:----:|:----:|:----:|:----:|:----:|
| BEE24       | 48.4 | 67.8 | 64.5 | 1058 |  --  | 44.7 |
| MOT17       | 62.7 | 78.7 | 77.0 |  --  | 62.2 | 63.6 |
| DanceTrack  | 61.1 | 89.8 | 64.3 |  --  | 46.7 | 80.2 |
| SportsMOT   | 75.3 | 96.2 | 78.7 |  --  | 64.6 | 87.8 |

---

## Installation

### Requirements
- Python 3.7+
- PyTorch 1.7.0+
- CUDA 10.2+ (for GPU training)

### Step-by-step Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/SwitchTrack.git
    cd SwitchTrack-original/添加了wh的switchtrack
    ```

2.  **Create conda environment:**
    ```bash
    conda create -n switchtrack python=3.8
    conda activate switchtrack
    ```

3.  **Install PyTorch:**
    ```bash
    # For CUDA 11.3
    conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch

    # For CUDA 10.2
    conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.2 -c pytorch
    ```

4.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Compile DCNv2 (Deformable Convolution):**
    ```bash
    cd DCNv2
    python setup.py build develop
    cd ..
    ```

### Dataset Preparation

#### BEE24
```plaintext
{Data ROOT}
|-- bee24
|   |-- train
|   |   |-- BEE24-01
|   |   |   |-- img1
|   |   |   |   |-- 000001.jpg
|   |   |   |   |-- ...
|   |   |   |-- gt
|   |   |   |   |-- gt.txt            
|   |   |   |-- seqinfo.ini
|   |   |-- ...
|   |   |-- BEE24-35
|   |   |   |-- img1
|   |   |   |   |-- 000001.jpg
|   |   |   |   |-- ...
|   |   |   |-- gt
|   |   |   |   |-- gt.txt            
|   |   |   |-- seqinfo.ini
|   |   |-- ...
|   |-- test
|   |   |-- ...
```

#### MOT17
```plaintext
{Data ROOT}
|-- mot
|   |-- train
|   |   |-- MOT17-02
|   |   |   |-- img1
|   |   |   |   |-- 000001.jpg
|   |   |   |   |-- ...
|   |   |   |-- gt
|   |   |   |   |-- gt.txt            
|   |   |   |-- seqinfo.ini
|   |   |-- ...
|   |   |-- MOT20-01
|   |   |   |-- img1
|   |   |   |   |-- 000001.jpg
|   |   |   |   |-- ...
|   |   |   |-- gt
|   |   |   |   |-- gt.txt            
|   |   |   |-- seqinfo.ini
|   |   |-- ...
|   |-- test
|   |   |-- ...
```

#### DanceTrack
```plaintext
{Data ROOT}
|-- dancetrack
|   |-- train
|   |   |-- dancetrack0001
|   |   |   |-- img1
|   |   |   |   |-- 00000001.jpg
|   |   |   |   |-- ...
|   |   |   |-- gt
|   |   |   |   |-- gt.txt            
|   |   |   |-- seqinfo.ini
|   |   |-- ...
|   |-- val
|   |   |-- ...
|   |-- test
|   |   |-- ...
```

#### SportsMOT
```plaintext
{Data ROOT}
|-- sportsmot
|   |-- splits_txt
|   |-- scripts
|   |-- dataset
|   |   |-- train
|   |   |   |-- v_1LwtoLPw2TU_c006
|   |   |   |   |-- img1
|   |   |   |   |   |-- 000001.jpg
|   |   |   |   |   |-- ...
|   |   |   |   |-- gt
|   |   |   |   |   |-- gt.txt
|   |   |   |   |-- seqinfo.ini         
|   |   |   |-- ...
|   |   |-- val
|   |   |   |-- ...
|   |   |-- test
|   |   |   |-- ...
```

#### Generate COCO-format Annotations
```bash
cd lib/tools/
python convert_bee_to_coco.py
python convert_mot17_to_coco.py
python convert_dance_to_coco.py
python convert_sportsmot_to_coco.py
```

#### Download Pretrained Models
```bash
# ImageNet pretrained DLA-34
cd pretrain/
wget http://dl.yf.io/dla/models/imagenet/dla34-ba72cf86.pth
```

---

## Training

### BEE24

**Basic training:**
```bash
bash train_bee.sh
```

**Custom training:**
```bash
python train.py \
    --exp_id bee \
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
    --use_bfl \
    --hungarian \
    --custom_dataset_img_path /path/to/bee/train \
    --custom_dataset_ann_path /path/to/bee/annotations/train.json \
    --load_model pretrain/dla34-ba72cf86.pth
```

### MOT17

**Basic training:**
```bash
bash train_mot17.sh
```

**Custom training:**
```bash
python train.py \
    --exp_id mot17 \
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
    --custom_dataset_ann_path /path/to/MOT17/annotations/train.json \
    --load_model pretrain/dla34-ba72cf86.pth
```

### DanceTrack

**Basic training:**
```bash
bash train_dance.sh
```

**Custom training:**
```bash
python train.py \
    --exp_id dance \
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
    --custom_dataset_img_path /path/to/dancetrack/train \
    --custom_dataset_ann_path /path/to/dancetrack/annotations/train.json \
    --load_model pretrain/dla34-ba72cf86.pth
```

### SportsMOT

**Basic training:**
```bash
bash train_sports.sh
```

**Custom training:**
```bash
python train.py \
    --exp_id sports \
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
    --custom_dataset_img_path /path/to/sportsmot/train \
    --custom_dataset_ann_path /path/to/sportsmot/annotations/train.json \
    --load_model pretrain/dla34-ba72cf86.pth
```

### Key Parameters:

- `--arch dla34`: Backbone architecture (dla34, dla169, resnet50)
- `--use_bfl`: Enable Gate Attention Decoder
- `--wh`: Use width-height head for bbox prediction
- `--pre_hm`: Use previous heatmap for tracking
- `--hungarian`: Use Hungarian algorithm for data association
- `--num_head_conv 1`: Number of convolutional layers in detection heads

### Monitor training:
```bash
tensorboard --logdir=exp/tracking.ctdet/mot17_half_wh_bfl/logs
```

---

## Tracking

### MOT17 Validation

**Run tracking:**
```bash
bash test_mot17.sh
```

**Custom tracking:**
```bash
python test.py \
    --exp_id mot17 \
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

### DanceTrack / SportsMOT / BEE24

Similar to MOT17, adjust dataset paths and thresholds:

```bash
# BEE24
bash test_bee.sh

# DanceTrack
bash test_dance.sh

# SportsMOT  
bash test_sports.sh
```

---

## Demo

### Visualize Tracking Results

**Enable visualization during testing:**
```bash
python test.py \
    --debug 1 \
    --show_track_color \
    # ... other parameters
```

**Output:** 

Visualization images will be saved to `exp/tracking.ctdet/{exp_id}/debug/`
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

---

## Citation

```bibtex
@article{switchtrack2024,
  title={Multiple Object Tracking by Switching Clues of the Association in Motion-complex Scene},
  author={Your Name},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2024}
}
```

---

## Acknowledgements

This work is built upon:
- [CenterTrack](https://github.com/xingyizhou/CenterTrack)
- [FairMOT](https://github.com/ifzhang/FairMOT)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
- [DLA](https://github.com/ucbdrive/dla)

---

## License

This project is released under the MIT License.
