# Technical-article--YOLOv12: Attention-Centric Real-Time Object Detectors


<div align="center">
<h1>YOLOv12</h1>
<h3>YOLOv12: Attention-Centric Real-Time Object Detectors</h3>



</div>
YOLO12 introduces an attention-centric architecture that departs from the traditional CNN-based approaches used in previous YOLO models, yet retains the real-time inference speed essential for many applications. This model achieves state-of-the-art object detection accuracy through novel methodological innovations in attention mechanisms and overall network architecture, while maintaining real-time performance. Despite those advantages, YOLO12 remains a community-driven release that may exhibit training instability, elevated memory consumption, and slower CPU throughput due to its heavy attention blocks, so Ultralytics still recommends YOLO11 for most production workloads.

Key Features 
Area Attention Mechanism: A new self-attention approach that processes large receptive fields efficiently. It divides feature maps into l equal-sized regions (defaulting to 4), either horizontally or vertically, avoiding complex operations and maintaining a large effective receptive field. This significantly reduces computational cost compared to standard self-attention.
Residual Efficient Layer Aggregation Networks (R-ELAN): An improved feature aggregation module based on ELAN, designed to address optimization challenges, especially in larger-scale attention-centric models. R-ELAN introduces:
Block-level residual connections with scaling (similar to layer scaling).
A redesigned feature aggregation method creating a bottleneck-like structure.
Optimized Attention Architecture: YOLO12 streamlines the standard attention mechanism for greater efficiency and compatibility with the YOLO framework. This includes:
Using FlashAttention to minimize memory access overhead.
Removing positional encoding for a cleaner and faster model.
Adjusting the MLP ratio (from the typical 4 to 1.2 or 2) to better balance computation between attention and feed-forward layers.
Reducing the depth of stacked blocks for improved optimization.
Leveraging convolution operations (where appropriate) for their computational efficiency.
Adding a 7x7 separable convolution (the "position perceiver") to the attention mechanism to implicitly encode positional information.
Comprehensive Task Support: YOLO12 supports a range of core computer vision tasks: object detection, instance segmentation, image classification, pose estimation, and oriented object detection (OBB).
Enhanced Efficiency: Achieves higher accuracy with fewer parameters compared to many prior models, demonstrating an improved balance between speed and accuracy.
Flexible Deployment: Designed for deployment across diverse platforms, from edge devices to cloud infrastructure.
YOLO12 comparison visualization







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


## Demo

```
python app.py
# Please visit http://127.0.0.1:7860
```


## Acknowledgement

The code is based on [ultralytics](https://github.com/ultralytics/ultralytics). Thanks for their excellent work!


```
