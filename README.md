# Operator-Transformer
Code for reproducing *"Transformer for Partial Differential Equations' Operator Learning"*  ([paper](https://openreview.net/forum?id=EPPqt3uERT&referrer=%5BTMLR%5D(%2Fgroup%3Fid%3DTMLR))).

For instruction on different cases, please go the corresponding subfolder. These codes are tested under PyTorch 1.8.1 on Ubuntu 18.

### Datasets for 1D Burgers/2D Darcy flow/2D Navier-Stokes (uniform equidist grid)

The dataset for 1D Burgers (Burgers_R10.zip), 2D Darcy flow (Darcy_421.zip) can be downloaded from [dataset link](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-) .</br>
We provide our processed dataset for 2D Navier-Stokes (in .npy format) at [dataset link](https://drive.google.com/drive/folders/1z-0V6NSl2STzrSA6QkzYWOGHSTgiOSYq?usp=sharing) .</br>
The dataset for these problems are under the courtesy of [FNO](https://github.com/zongyi-li/fourier_neural_operator).

### Datasets for BVP problem on non-uniform grid

Dataset courtesy under [GNN-BVP](https://github.com/merantix-momentum/gnn-bvp-solver), please check the original repo for data downloading.

### Datasets for IVP problem on Airfoil

Dataset courtesy under [MeshGraphNet](https://github.com/deepmind/deepmind-research/tree/master/meshgraphnets), we provide our processed dataset at [train](https://drive.google.com/file/d/1z88dPaJixOo6KYjjZ7EsBfNX42EwBQ4B/view?usp=share_link)/[test](https://drive.google.com/file/d/1mEZ6gVYJ5UvLupWz1cft4XXlVuwWktgs/view?usp=sharing).

### Pretrained model checkpoint and log

| Problem       | link   |
|---------------|---------------------------------------------------------------------------|
| NS2D-Re200  |  [link](https://drive.google.com/drive/folders/1hkmyAzO0glTLfnI8k5x84MTADUafNU4A?usp=sharing) |
| NS2D-mixRe |  [link](https://drive.google.com/file/d/1_-aAd_GHX8StyKWZLpvSWeGQ3vyytf7L) |
| NS2D-Re20| [link](https://drive.google.com/drive/folders/1KYSYsmB0XAi90g39x3qsbRCy8xSDOk1r?usp=sharing) |
| Burgers   |  [link](https://drive.google.com/file/d/1eDFJD-wiTxzDzywSvXLgzffI25su1S1q) |
| Darcy | [link](https://drive.google.com/drive/folders/1-56szGnQxZhv-uUQyw9RjT54WiPg9CFa?usp=sharing) |
| Airfoil | [link](https://drive.google.com/drive/folders/1teVWGi-hPST-aY914OVrLMLPREQ-JKxz?usp=sharing) |
| Electrostatics | [link](https://drive.google.com/drive/folders/1sfihAxiPdLAQEF6mlR0n5ZpjK_Ctpfee?usp=sharing) |
| Magnetostatics | [link](https://drive.google.com/drive/folders/1atAGnt_CRZhaPVVgxbpwRtAbMODxa5nZ?usp=sharing) |


## Relevant works
Alongside the aforementioned projects that have generously shared their valuable datasets, the following repositories have also been helpful for this project.

* Galerkin Transformer: https://github.com/scaomath/galerkin-transformer
* A lot useful modules for building Transformer: https://github.com/lucidrains/x-transformers

## Citations
