# Untrained deep network (UDN) reconstructions for compressive lensless photography
## [Project Page](https://waller-lab.github.io/UDN/index.html) | [Paper](https://doi.org/10.1364/OE.424075)


Please cite the following paper when using this code or data:


```
@article{Monakhova:21,
author = {Kristina Monakhova and Vi Tran and Grace Kuo and Laura Waller},
journal = {Opt. Express},
number = {13},
pages = {20913--20929},
publisher = {OSA},
title = {Untrained networks for compressive lensless photography},
volume = {29},
month = {Jun},
year = {2021},
url = {http://www.opticsexpress.org/abstract.cfm?URI=oe-29-13-20913},
doi = {10.1364/OE.424075}
}

```


## Contents

1. [Description](#Description)
2. [Data](#Data)
3. [Setup](#Setup)

## Description
This repository contains python code to run untrained deep reconstructions (UDN) for compressive lensless photography. It contains examples of how to recover videos from stills and hyperspectral volumes from a single image. Our unsupervised method doesn't require any training data, so feel free to try this out on any raw measurement from your compressive lensless camera. 

This codebase draws heavily upon the Deep Image Prior work, which can be found [here](https://github.com/DmitryUlyanov/deep-image-prior). 
In addition, we use raw measurements, camera hardware, and models from the following works:
 * [Spectral DiffuserCam](https://waller-lab.github.io/SpectralDiffuserCam/)
 * [Video from Stills](https://arxiv.org/abs/1905.13221)
 * [DiffuserCam](https://waller-lab.github.io/DiffuserCam/)

## Data
Sample data (needed to run the code) can be found [here](https://drive.google.com/drive/folders/11vksgAZY0pPK1lNiWAl9MvMug_sFPhl5?usp=sharing). Please place the sample data in the data folder.


## Setup
Clone this project using: 
```
git clone https://github.com/Waller-Lab/UDN.git
```

The dependencies can be installed by using:
```
conda env create -f environment.yml
source activate UDN
```

Please place the downloaded data in the data folder, as specified above.

The following jupyter notebook demos are provided:

* Demo_2D_erasures.ipynb - contains example UDN reconstructions for 2D imaging with added erasures using simulated or experimental data
* Demo_single_shot_video.ipynb - contains example reconstructions for recovering video from a single image using UDN, both in simulation and experiment
* Demo_single_shot_hyperspectral.ipynb - contains example reconstructions for recovering hyperspectral volumes from a single image using UDN, both in simulation and experiment

This code requires a GPU to run. 