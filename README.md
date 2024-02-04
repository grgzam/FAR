# FAR
This is the official implementation of "Feature Aware Re-weighting (FAR) in Bird’s Eye View for LiDAR-based 3D object detection in autonomous driving applications" paper, that you can download [here](). 
This project is built on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet).

## Requirements

All the codes are tested in the following environment:

- Linux (tested on Ubuntu 18.04)
- Python 3.8.8
- PyTorch 1.10
- CUDA 11.1

- ## Install
1. Install the spconv library from [spconv](https://github.com/traveller59/spconv).
2. Install pytorch 1.10 `conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.1 -c pytorch -c conda-forge`
3. Install requirements `pip install -r requirements.txt`
4. Install pcdet library `python setup.py develop`

### KITTI 3D Object Detection Baselines
Selected supported methods are shown in the below table. The results are the 3D detection performance of moderate difficulty on the *val* set of KITTI dataset.


|                                             | Car@R11 | Pedestrian@R11 | Cyclist@R11  | weights |
|---------------------------------------------|:-------:|:-------:|:-------:|:---------:|
| [PointPillar_FAR](tools/cfgs/kitti_models/pointpillar_FAR.yaml) | 77.28 | 52.29 | 62.68 | [model](https://vc.ee.duth.gr:6960/index.php/s/0krLEwFNkHrN4Wz) | 
| [CenterPoint_FAR](tools/cfgs/kitti_models/centerpoint_dyn_pillar_1x_FAR.yaml) | 78.70 | 54.41 | 72.11 | [model](https://vc.ee.duth.gr:6960/index.php/s/j2r54j77MsTgyVu)| 
| [SECOND_FAR](tools/cfgs/kitti_models/second_FAR.yaml)       | 78.62 | 52.98 | 67.15 | [model](https://vc.ee.duth.gr:6960/index.php/s/g95yytjRRdSwAwG) |
| [PV-RCNN_FAR](tools/cfgs/kitti_models/pv_rcnn_FAR.yaml) | 83.61 | 57.90 | 70.47 | [model](https://vc.ee.duth.gr:6960/index.php/s/g95yytjRRdSwAwG) |
| [Voxel R-CNN_FAR (all classes)](tools/cfgs/kitti_models/voxel_rcnn_all_classes_FAR.yaml) |84.54 | - | - | [model](https://vc.ee.duth.gr:6960/index.php/s/JK3KAIC2Ze3SLG2) |
||




* All LiDAR-based models are trained with 8 GTX 1080Ti GPUs and are available for download. 
* The training time is measured with 8 TITAN XP GPUs and PyTorch 1.5.

|                                             | training time | Car@R11 | Pedestrian@R11 | Cyclist@R11  | download | 
|---------------------------------------------|----------:|:-------:|:-------:|:-------:|:---------:|
| [PointPillar](tools/cfgs/kitti_models/pointpillar.yaml) |~1.2 hours| 77.28 | 52.29 | 62.68 | [model-18M](https://drive.google.com/file/d/1wMxWTpU1qUoY3DsCH31WJmvJxcjFXKlm/view?usp=sharing) | 
| [SECOND](tools/cfgs/kitti_models/second.yaml)       |  ~1.7 hours  | 78.62 | 52.98 | 67.15 | [model-20M](https://drive.google.com/file/d/1-01zsPOsqanZQqIIyy7FpNXStL3y4jdR/view?usp=sharing) |
| [SECOND-IoU](tools/cfgs/kitti_models/second_iou.yaml)       | -  | 79.09 | 55.74 | 71.31 | [model-46M](https://drive.google.com/file/d/1AQkeNs4bxhvhDQ-5sEo_yvQUlfo73lsW/view?usp=sharing) |
| [PointRCNN](tools/cfgs/kitti_models/pointrcnn.yaml) | ~3 hours | 78.70 | 54.41 | 72.11 | [model-16M](https://drive.google.com/file/d/1BCX9wMn-GYAfSOPpyxf6Iv6fc0qKLSiU/view?usp=sharing)| 
| [PointRCNN-IoU](tools/cfgs/kitti_models/pointrcnn_iou.yaml) | ~3 hours | 78.75 | 58.32 | 71.34 | [model-16M](https://drive.google.com/file/d/1V0vNZ3lAHpEEt0MlT80eL2f41K2tHm_D/view?usp=sharing)|
| [Part-A2-Free](tools/cfgs/kitti_models/PartA2_free.yaml)   | ~3.8 hours| 78.72 | 65.99 | 74.29 | [model-226M](https://drive.google.com/file/d/1lcUUxF8mJgZ_e-tZhP1XNQtTBuC-R0zr/view?usp=sharing) |
| [Part-A2-Anchor](tools/cfgs/kitti_models/PartA2.yaml)    | ~4.3 hours| 79.40 | 60.05 | 69.90 | [model-244M](https://drive.google.com/file/d/10GK1aCkLqxGNeX3lVu8cLZyE0G8002hY/view?usp=sharing) |
| [PV-RCNN](tools/cfgs/kitti_models/pv_rcnn.yaml) | ~5 hours| 83.61 | 57.90 | 70.47 | [model-50M](https://drive.google.com/file/d/1lIOq4Hxr0W3qsX83ilQv0nk1Cls6KAr-/view?usp=sharing) |
| [Voxel R-CNN (Car)](tools/cfgs/kitti_models/voxel_rcnn_car.yaml) | ~2.2 hours| 84.54 | - | - | [model-28M](https://drive.google.com/file/d/19_jiAeGLz7V0wNjSJw4cKmMjdm5EW5By/view?usp=sharing) |
| [Focals Conv - F](tools/cfgs/kitti_models/voxel_rcnn_car_focal_multimodal.yaml) | ~4 hours| 85.66 | - | - | [model-30M](https://drive.google.com/file/d/1u2Vcg7gZPOI-EqrHy7_6fqaibvRt2IjQ/view?usp=sharing) |
||
| [CaDDN (Mono)](tools/cfgs/kitti_models/CaDDN.yaml) |~15 hours| 21.38 | 13.02 | 9.76 | [model-774M](https://drive.google.com/file/d/1OQTO2PtXT8GGr35W9m2GZGuqgb6fyU1V/view?usp=sharing) |
