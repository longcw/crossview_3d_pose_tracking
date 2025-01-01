# Dataset of "Cross-View Tracking for Multi-Human 3D Pose Estimation at over 100 FPS"

>**Note**: The repo contains the dataset used in the paper, 
including Campus, Shelf, StoreLayout1, StoreLayout2.
Along with the data,
we provide some scripts to visualize the data, in both 2D and 3D,
and also to evaluate with the results.
The source code is not included 
as this is a commercial project,
find more in http://aifi.io if you are interested.


## Dataset
Here we provide four datasets, including
+ Campus: https://www.epfl.ch/labs/cvlab/data/data-pom-index-php/
+ Shelf: http://campar.in.tum.de/Chair/MultiHumanPose
+ StoreLayout1: proposed by AiFi Inc.
+ StoreLayout2: proposed by AiFi Inc.


For convenient, you can find and download them by one click from [GoogleDrive](https://drive.google.com/drive/folders/1LJGcP2v0aQDmetnCzO2PiRP1v4jU6sFC?usp=drive_link).

### Data Structure
For each dataset, the structure of the directory is organized as follow
```
Campus_Seq1
├── annotation_2d.json
├── annotation_3d.json
├── calibration.json
├── detection.json
├── frames
│   ├── Camera0
│   ├── Camera1
│   └── Camera2
│       ├── 0060.720.jpg
│       ├── 0060.760.jpg
│       ├── 0060.800.jpg
│       └── xxxxxxxx.jpg
└── result_3d.json
```
The `annotations` were only provided in Campus and Shelf datasets
and the `detection` is generated using 
Cascaded Pyramid Network (CPN) in https://github.com/zju3dv/mvpose.
The `frames` are renamed using timestamps, i.e. the name
of each file is the tiemstamp in second of that frame.



### Data Format
2D (2D annotation and detection) and 3D (3D annotation and tracking result) 
data have their own unified data format as follows.
#### 2D Data Format
The 2D data is organized by frames:
```json
{
  "image_wh": [360, 288],
  "frames": {
    "Camera0/0002.320.jpg": {
      "camera": "Camera0",
      "timestamp": 2.32,
      "poses": []
    },
    "Camera0/0002.360.jpg": {
      "camera": "Camera0",
      "timestamp": 2.36,
      "poses": [
        {
          "id": -1,
          "points_2d": Nx2 Array,
          "scores": N Array
        },
        ...
      ]
    },
    ...
  }
}
```

#### 3D Data Format
The 3D data is organized by timestamps:
```json
[
  {
    "timestamp": 6.08,
    "poses": [
      {
        "id": 10159970873491820000,
        "points_3d": Nx3 Array,
        "scores": N Array
      },
      ...
    ]
  },
  ...
]
```

### Human Pose Format
In the annotation
the human pose has 14 keypoints:
```json
0: 'r-ankle',
1: 'r-knee',
2: 'r-hip',
3: 'l-hip',
4: 'l-knee',
5: 'l-ankle',
6: 'r-wrist',
7: 'r-elbow',
8: 'r-shoulder',
9: 'l-shoulder',
10: 'l-elbow',
11: 'l-wrist',
12: 'bottom-head',
13: 'top-head'
```
In detection and result, the human pose has 17 keypoints:
```json
0: 'nose',
1: 'l-eye',
2: 'r-eye',
3: 'l-ear',
4: 'r-ear',
5: 'l-shoulder',
6: 'r-shoulder',
7: 'l-elbow',
8: 'r-elbowr',
9: 'l-wrist',
10: 'r-wrist',
11: 'l-hip',
12: 'r-hip',
13: 'l-knee'
14: 'r-knee'
15: 'l-ankle'
16: 'r-ankle'
```

## Demo
Along with the data, here we provide some tools 
to load the data and calibration, visualize and evaluate the result.
### Visualize Annotation
```bash
DATA_ROOT=/data/3DPose_pub/Campus_Seq1

# 2D
python display.py --frame-root ${DATA_ROOT}/frames --calibration ${DATA_ROOT}/calibration.json --pose-file ${DATA_ROOT}/annotation_2d.json --pose-type 2d

# 3D (only tested on Linux)
python display.py --frame-root ${DATA_ROOT}/frames --calibration ${DATA_ROOT}/calibration.json --pose-file ${DATA_ROOT}/annotation_3d.json --pose-type 3d
```

### Visualize Detection and Result
```bash
DATA_ROOT=/data/3DPose_pub/Campus_Seq1

# 2D detection
python display.py --frame-root ${DATA_ROOT}/frames --calibration ${DATA_ROOT}/calibration.json --pose-file ${DATA_ROOT}/detection.json --pose-type 2d

# 3D result
python display.py --frame-root ${DATA_ROOT}/frames --calibration ${DATA_ROOT}/calibration.json --pose-file ${DATA_ROOT}/result_3d.json --pose-type 3d
```

### 3D visualization with Docker
Sometimes it's hard to setup the environment for vispy. 
Here we provide a dockerfile supports OpenGL and CUDA applications (from https://medium.com/@benjamin.botto/opengl-and-cuda-applications-in-docker-af0eece000f1).

1. To use it you will need `nvidia-container-runtime`: https://github.com/NVIDIA/nvidia-container-runtime#installation

2. Build the docker image 
   ```bash
   docker build -t glvnd-x-vispy:latest .
   ```
3. Start the container
   ```bash
   # Connecting to the Host’s X Server
   xhost +local:root

   docker run \
   --rm \
   -it \
   --gpus all \
   -v /tmp/.X11-unix:/tmp/.X11-unix \
   -e DISPLAY=$DISPLAY \
   -e QT_X11_NO_MITSHM=1 \
   -v /PATH-TO-DATA/3DPose_pub:/data/3DPose_pub \
   -v /PATH-TO-CODE/crossview_3d_pose_tracking:/app \
   glvnd-x-vispy bash
   ```
4. Run the demo in a docker container
   ```bash
   cd /app
   pip3 install -r requirements.txt

   DATA_ROOT=/data/3DPose_pub/Campus_Seq1

   # 2D detection
   python3 display.py --frame-root ${DATA_ROOT}/frames --calibration ${DATA_ROOT}/calibration.json --pose-file ${DATA_ROOT}/detection.json --pose-type 2d

   # 3D result
   python3 display.py --frame-root ${DATA_ROOT}/frames --calibration ${DATA_ROOT}/calibration.json --pose-file ${DATA_ROOT}/result_3d.json --pose-type 3d
   ```

### Evaluate

```bash
DATA_ROOT=/data/3DPose_pub/Campus_Seq1

python evaluate.py --annotation ${DATA_ROOT}/annotation_3d.json --result ${DATA_ROOT}/result_3d.json 
```
Then you will get the the output like
```
+------------+---------+---------+---------+---------+
| Bone Group | Actor 0 | Actor 1 | Actor 2 | Average |
+------------+---------+---------+---------+---------+
|    Head    |  1.0000 |  1.0000 |  0.9928 |  0.9976 |
|   Torso    |  1.0000 |  1.0000 |  1.0000 |  1.0000 |
| Upper arms |  0.9592 |  1.0000 |  1.0000 |  0.9864 |
| Lower arms |  0.8980 |  0.7063 |  0.9348 |  0.8464 |
| Upper legs |  1.0000 |  1.0000 |  1.0000 |  1.0000 |
| Lower legs |  1.0000 |  1.0000 |  1.0000 |  1.0000 |
|   Total    |  0.9714 |  0.9413 |  0.9862 |  0.9663 |
+------------+---------+---------+---------+---------+
```

## Citation
```
@InProceedings{Chen_2020_CVPR,
author = {Chen, Long and Ai, Haizhou and Chen, Rui and Zhuang, Zijie and Liu, Shuang},
title = {Cross-View Tracking for Multi-Human 3D Pose Estimation at Over 100 FPS},
booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```
