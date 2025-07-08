
<h1 align="center">Toward Robust Neural Reconstruction from Sparse Point Sets</h1>
<p align="center"><a href="https://arxiv.org/pdf/2412.16361"><img src='https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white' alt='arXiv'></a>
<a href='https://ouasfi.github.io/sdro/'><img src='https://img.shields.io/badge/Project_Page-Website-green?logo=googlechrome&logoColor=white' alt='Project Page'></a>
<p align="center"><img src="assets/teaser.png" width="100%"></p>



This repository contains the implementation of the CVPR 2025 paper [Toward Robust Neural Reconstruction from Sparse Point Sets](https://arxiv.org/pdf/2412.16361)  by Amine Ouasfi, Shubhendu Jena, Eric Marchand, Adnane Boukhayma.

## Overview
------------

This paper proposes a novel approach for unsupervised signed distance learning from sparse and noisy point clouds. The method learns to predict the signed distance function of a 3D shape from a sparse set of points, without requiring any supervision or prior knowledge of the scene. 

## Repository Structure
------------------------

The repository is organized as follows:

* `models`: contains the implementation of the neural network architecture used in the paper.
* `traniers`: contains the implementation of our proposed method.
* `train.py`: contains the training script.

## Environment setup
------------

The code is written in Python and requires the following dependencies:

````
 # Create conda environment
conda env create

# Activate it
conda activate sdf_dro

# Install pytorch 
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
````


## Data
-----

### ShapeNet
We use a subset of the [ShapeNet](https://shapenet.org/) data as chosen by [Neural Splines](https://github.com/fwilliams/neural-splines). This data is first preprocessed to be watertight as per the pipeline in the [Occupancy Networks repository](https://github.com/autonomousvision/occupancy_networks), who provide both the pipleline and the entire preprocessed dataset (73.4GB). 

The Neural Spline split uses the first 20 shapes from the test set of 13 shape classes from ShapeNet.You can download the dataset (73.4 GB) by running the [script](https://github.com/autonomousvision/occupancy_networks#preprocessed-data) from Occupancy Networks. After, you should have the dataset in `data/ShapeNet` folder.


### Faust

The Faust Dataset can be downloaded from the [official website ](https://faust-leaderboard.is.tuebingen.mpg.de). We followed the preprocessing steps outlined in [Occupancy Networks repository](https://github.com/autonomousvision/occupancy_networks). Specifically, we normalized the meshes to the unit cube and uniformly sampled 100,000 points with their corresponding normals for evaluation.

### Surface Reconstruction Benchamark data
The Surface Reconstruction Benchmark (SRB) data is provided in the [Deep Geometric Prior repository](https://github.com/fwilliams/deep-geometric-prior).

If you use this data in your research, make sure to cite the Deep Geometric Prior paper.

## Training
------------

To train the SDF network, run the following command:
```bash
python train.py sn_config.json
```
This will train the network using the configuration specified in `config.json` and store the trained model in the `results` directory. 

## Evaluation
-------------

To evaluate the trained model, run the following command:
```bash
python eval.py  sn_config.json
```
This will evaluate the model on the test set and store the results in the `results` directory.

## Configuration
-------------

The configuration file `configs/conf.conf` contains the following parameters:

* `n_points`: the number of points to sample from the point cloud.
* `sigma`: the standard deviation of the noise added to the point cloud.
* `rho`: controls the the strength of the entropic regularization.
* `lambda_wasserstain`: controls how close the worst-case distribution Q′ is to the nominal distribution.
* `m_dro`: The number of queries used to estimate the worst-case distribution.
 
 
## Citation
------------

If you use this code in your research, please cite the following paper:
```
@inproceedings{ouasfi2025toward,
  title={Toward robust neural reconstruction from sparse point sets},
  author={Ouasfi, Amine and Jena, Shubhendu and Marchand, Eric and Boukhayma, Adnane},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={6552--6562},
  year={2025}
}
```


