# NextStop

# Installation
## System Requirements
This code has only been tested on the following combination of major pre-requisites. Please check beforehand.

* Ubuntu 22.04
* Python 3.8

## Dependencies:
To install required dependencies on the system python, please run the following command at the root of this code:
```
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

pip3 install -r requirements.txt
```

# Data
Download the SemanticKITTI dataset with labels from [here](http://semantic-kitti.org/dataset.html#download/).  
Add the [semantic-kitti.yaml](https://github.com/PRBonn/semantic-kitti-api/blob/master/config/semantic-kitti.yaml) file to the folder.

Folder structure:
```
SemanticKitti/  
└── semantic-kitti.yaml  
└── sequences/  
    └── 00/  
        └── calib.txt  
        └── poses.txt  
        └── times.txt  
        └── labels  
            ├── 000000.label  
            ├── 000000.center.npy  
            ...  
         └── velodyne  
            ├── 000000.bin  
            ...
```
# Tracking
We provided 4D-stop sample result to this repository, so there is no need to run 4DSTOP network for our tracker.
just run the two files: 

1. bounding box tracker :
```
tracking/box_tracker.py
```
2. from bounding box to labels tracker:
```
tracking/box_tracker_to_labels.py
```

if you dont want to run this sample you can run from your own path by :
1. bounding box tracker :
```
tracking/box_tracker.py --dataset [path to the SemanticKitti point cloud] --data_cfg [path to SemanticKitti config file] -- --predictions [path to prediction] --sequences [sequence number] --split [valid or not]
```
2. from bounding box to labels tracker:
```
tracking/box_tracker_to_labels.py --dataset [path to the SemanticKitti point cloud] --data_cfg [path to SemanticKitti config file] -- --predictions [path to prediction] --sequences [sequence number] --split [valid or not]
```

# Evaluation

## eval_lstq
```
utils/evaluate_4dpanoptic.py --dataset=/media/nirit/mugiwara/datasets/SemanticKitti/ --predictions=/media/nirit/mugiwara/code/4D-StOP/4D-StOP-main/nirit_test_net/Log_2022-06-13_17-33-24_importance_None_str1_bigpug_2_current_chkp/AB3DMOT_tracker/to_labels/ --data_cfg=/media/nirit/mugiwara/datasets/SemanticKitti/semantic-kitti.yaml --split valid
```

## eval_motp
```
--dataset=/media/nirit/mugiwara/datasets/SemanticKitti/ --predictions=./predictions_data/NextStop_tracker/to_labels/ --data_cfg=/media/nirit/mugiwara/datasets/SemanticKitti/semantic-kitti.yaml --split valid
```

## eval_lD
```
utils/evaluate_identity.py --dataset=/media/nirit/mugiwara/datasets/SemanticKitti/ --predictions=./predictions_data/NextStop_tracker/to_labels/ --data_cfg=/media/nirit/mugiwara/datasets/SemanticKitti/semantic-kitti.yaml --split valid
```

# Acknowledgments
The code is based on [4D-StOP](https://github.com/LarsKreuzberg/4D-StOP), [semantic-kitti-api](https://github.com/PRBonn/semantic-kitti-api) and [AB3DMOT](https://github.com/xinshuoweng/AB3DMOT)
