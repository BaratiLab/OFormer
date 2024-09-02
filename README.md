# Operator-Transformer
Code for reproducing *"Transformer for Partial Differential Equations' Operator Learning"*  ([paper](https://openreview.net/forum?id=EPPqt3uERT&referrer=%5BTMLR%5D(%2Fgroup%3Fid%3DTMLR))).

(Updated Aug31/2024) OFormer code has been re-organized will become part of the [neuraloperator library](https://github.com/neuraloperator/neuraloperator/pull/293) in the future release. Check it out!

<div style style=”line-height: 20%” align="center">
<h3> Models prediction on 2D Incompressible flow </h3>
<img src="https://github.com/BaratiLab/OFormer/blob/main/oformer_ns2d_re200.gif" width="600">
</div>

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
If you find this project useful, please consider citing our work:
```
@article{
li2023transformer,
title={Transformer for Partial Differential Equations{\textquoteright} Operator Learning},
author={Zijie Li and Kazem Meidani and Amir Barati Farimani},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2023},
url={https://openreview.net/forum?id=EPPqt3uERT},
note={}
}
```
