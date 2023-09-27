# NextStop

# Installation
## System Requirements
This code has only been tested on the following combination of major pre-requisites. Please check beforehand.

* Ubuntu 22.04
* Python 3.8

## Dependencies:
To install required dependencies on the system python, please run the following command at the root of this code:
```
pip3 install -r requirements.txt
```

# Data
Download the SemanticKITTI dataset with labels from [here](http://semantic-kitti.org/dataset.html#download/).  
Add the semantic-kitti.yaml file to the folder.

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

# Acknowledgments
The code is based on [4D-StOP](https://github.com/LarsKreuzberg/4D-StOP) and [AB3DMOT](https://github.com/xinshuoweng/AB3DMOT)
