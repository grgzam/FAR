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
| [PointPillar_FAR](tools/cfgs/kitti_models/pointpillar_FAR.yaml) | 76.87 | 52.05 | 63.63 | [model](https://vc.ee.duth.gr:6960/index.php/s/0krLEwFNkHrN4Wz) | 
| [CenterPoint_FAR](tools/cfgs/kitti_models/centerpoint_dyn_pillar_1x_FAR.yaml) | 76.73 | 50.72 | 65.10 | [model](https://vc.ee.duth.gr:6960/index.php/s/j2r54j77MsTgyVu)| 
| [SECOND_FAR](tools/cfgs/kitti_models/second_FAR.yaml)       | 78.30 | 53.92 | 67.27 | [model](https://vc.ee.duth.gr:6960/index.php/s/g95yytjRRdSwAwG) |
| [PV-RCNN_FAR](tools/cfgs/kitti_models/pv_rcnn_FAR.yaml) | 83.89 | 60.76 | 72.18 | [model](https://vc.ee.duth.gr:6960/index.php/s/g95yytjRRdSwAwG) |
| [Voxel R-CNN_FAR (all classes)](tools/cfgs/kitti_models/voxel_rcnn_all_classes_FAR.yaml) |83.38 | 60.43 | 72.47 | [model](https://vc.ee.duth.gr:6960/index.php/s/JK3KAIC2Ze3SLG2) |
||



