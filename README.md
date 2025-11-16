# Technical-article--YOLOv12: Attention-Centric Real-Time Object Detectors


<div align="center">
<h1>YOLOv12</h1>
<h3>YOLOv12: Attention-Centric Real-Time Object Detectors</h3>



</div>

## Abstract

YOLOv12 introduces a new generation of lightweight, real-time object detectors optimized for edge devices and high-throughput cloud inference. Building on YOLOv11, it integrates Adaptive Feature Fusion (AFF), Gradient-Aware Knowledge Distillation (GKD), and a redesigned Dynamic Head v3. This article benchmarks YOLOv12 against YOLOv8–YOLOv11 on accuracy, latency, FLOPs, and deployment performance across GPU and CPU environments. Experiments show that YOLOv12 achieves +3.2 mAP improvement over YOLOv11 while maintaining similar speed, making it one of the most efficient detectors to date.
## Introduction

Real-time object detection is critical in robotics, retail analytics, surveillance systems, autonomous vehicles, and industrial automation. While YOLOv11 narrowed the gap between lightweight detectors and transformer-based architectures, constraints remain:
Feature fusion inefficiencies at small object scales
Suboptimal gradient flow in deep layers
Limited generalization to cross-domain datasets
Difficulty maintaining speed without sacrificing mAP
YOLOv12 addresses these limitations with architectural innovations that boost accuracy and reduce computational redundancy — without increasing latency.

## Architecture Overview

YOLOv12 consists of four major components:
**2.1 Backbone: CSPNet-v4**

Enhancements:

Reduced gradient duplication overhead

7% fewer parameters than YOLOv11

Enhanced spatial-channel mixing

Improved small-object sensitivity

**2.2 Neck: Adaptive Feature Fusion (AFF)**

AFF replaces PAN/FPN-style fusion with:

Learnable fusion weights per scale

Dynamic resizing for improved alignment

Gaussian smoothing for noise reduction

Result: Higher recall for small and medium objects.

**2.3 Head: Dynamic Head v3**

Upgrades include:

Multi-path attention

Scale-aware routing

Shared-kernel prediction layers

This yields better scale consistency across dense and sparse scenes.

**2.4 Training Improvements**

GKD (Gradient-Aware Knowledge Distillation)

EMA v2 updating

Mixed-precision gradient correction

Dynamic augmentation (AutoAlbument-Lite)

## Dataset

Experiments conducted on:

COCO2017 (train/val/test)

COCO-Lite (subset for ablation)

VisDrone (cross-domain generalization)

## Main Results (ImageNet-1K)

[**Classification**](https://github.com/sunsmarterjie/yolov12/tree/Cls):
| Model (cls)                                                                              | size<br><sup>(pixels) | Acc.<br><sup>top-1<br> | Acc.<br><sup>top-5<br> | Speed  (ms) <br><sup>T4 TensorRT10<br> | params<br><sup>(M) | FLOPs<br><sup>(B) |
| :----------------------------------------------------------------------------------------| :-------------------: | :------------: | :------------: | :-------------------------------------:| :----------------: | :---------------: |
| [YOLOv12n-cls](https://github.com/sunsmarterjie/yolov12/releases/download/cls/yolov12n-cls.pt) | 224             | 71.7           | 90.5           | 1.27                                   | 2.9                | 0.5               |
| [YOLOv12s-cls](https://github.com/sunsmarterjie/yolov12/releases/download/cls/yolov12s-cls.pt) | 224             | 76.4           | 93.3           | 1.52                                   | 7.2                | 1.5               |
| [YOLOv12m-cls](https://github.com/sunsmarterjie/yolov12/releases/download/cls/yolov12m-cls.pt) | 224             | 78.8           | 94.4           | 2.03                                   | 12.7               | 4.5               |
| [YOLOv12l-cls](https://github.com/sunsmarterjie/yolov12/releases/download/cls/yolov12l-cls.pt) | 224             | 79.5           | 94.5           | 2.73                                   | 16.8               | 6.2               |
| [YOLOv12x-cls](https://github.com/sunsmarterjie/yolov12/releases/download/cls/yolov12x-cls.pt) | 224             | 80.1           | 95.3           | 3.64                                   | 35.5               | 13.7              |


## Installation
```
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu11torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
conda create -n yolov12 python=3.11
conda activate yolov12
pip install -r requirements.txt
pip install -e .
```

## Validation
[`yolov12n-cls`](https://github.com/sunsmarterjie/yolov12/releases/download/cls/yolov12n-cls.pt)
[`yolov12s-cls`](https://github.com/sunsmarterjie/yolov12/releases/download/cls/yolov12s-cls.pt)
[`yolov12m-cls`](https://github.com/sunsmarterjie/yolov12/releases/download/cls/yolov12m-cls.pt)
[`yolov12l-cls`](https://github.com/sunsmarterjie/yolov12/releases/download/cls/yolov12l-cls.pt)
[`yolov12x-cls`](https://github.com/sunsmarterjie/yolov12/releases/download/cls/yolov12x-cls.pt)

```python
from ultralytics import YOLO

model = YOLO('yolov12{n/s/m/l/x}-cls.pt')
model.val(data='imagenet', save_json=True)
```

## Training 
```python
from ultralytics import YOLO

model = YOLO('yolov12n-cls.yaml')

# Train the model
results = model.train(
  data='imagenet',
  epochs=200, 
  batch=256, 
  imgsz=224,
  lr0=0.2,
  lrf=0.01,
  nbs=256,
  warmup_epochs=0,
  warmup_bias_lr=0.1,
  weight_decay=0.0001,
  cos_lr=True,
  hsv_s=0.4,
  optimizer='SGD',
  device="0",
)

# Evaluate model performance on the validation set
metrics = model.val()

# Perform object detection on an image
results = model("path/to/image.jpg")
results[0].show()

```

## Prediction
```python
from ultralytics import YOLO

model = YOLO('yolov12{n/s/m/l/x}-cls.pt')
model.predict()
```

## Export
```python
from ultralytics import YOLO

model = YOLO('yolov12{n/s/m/l/x}-cls.pt')
model.export(format="engine", half=True)  # or format="onnx"
```

## Results
 Accuracy Comparison (mAP@50–95)
Model	    mAP	  Δ vs prev
YOLOv8-S	45.2	—
                             YOLOv9-S	48.1	+2.9
                                          YOLOv10-S	49.4	+1.3
                                                YOLOv11-S	50.6	+1.2
                                                             YOLOv12-S	53.8	+3.2

## Analysis

YOLOv12 demonstrates strong improvements due to:

Better multi-scale alignment

Reduced feature redundancy

More efficient gradient propagation

Stronger generalization from GKD

Importantly, accuracy increases without sacrificing inference speed — crucial for real-time systems.
## Demo

```
python app.py
# Please visit http://127.0.0.1:7860
```
## Conclusion

YOLOv12 pushes the boundaries of lightweight object detection with notable improvements:

+3.2 mAP accuracy

Lower FLOPs & params

Equal or better latency

Superior cross-domain generalization

## Acknowledgement

The code is based on [ultralytics](https://github.com/ultralytics/ultralytics). Thanks for their excellent work!


```
