# GL-Fusion: Global-Local Fusion Network for Multi-view Echocardiogram Video Segmentation


## Contents
1. [setup](#setup)
2. [Training](#training)
3. [Evaluation](#evaluation)
4. [Visualize](#Visualize)

## Setup

Build the environment

```Shell
conda create -n GLFusion python=3.7
conda activate GLFusion
pip3 install -r requirements.txt
```

## Training
The training code are provided below. We train our network with 4 RTX 3090Ti.
```Shell
python main.py --mode train
```
We provide our pretrained model [here](https://hkustconnect-my.sharepoint.com/:f:/g/personal/jyangcu_connect_ust_hk/EuAO44c-s1pBgOvNqAzCgTYBuHLcgjjG_a4E1aPjotAJ1w?e=yRo0l3).

## Evaluation and Test
The evaluation and test code are provided below.
```Shell
python main.py --mode val 
```
## Visualize
To get visualization result.
```Shell
python main.py --mode visual
```

