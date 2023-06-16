<div align="center">
  <h1> Distilling Self-Supervised Vision Transformers for Weakly-Supervised Few-Shot Classification & Segmentation </h1>
</div>

<div align="center">
  <h3><a href=http://dahyun-kang.github.io>Dahyun Kang<sup>1,2</sup></a> &nbsp;&nbsp; Piotr Koniusz<sup>3,4</sup> &nbsp;&nbsp; Minsu Cho<sup>2</sup> &nbsp;&nbsp; Naila Murray<sup>1</sup> </h3>
  <h4> <sup>1</sup>Meta AI &nbsp; <sup>2</sup>POSTECH &nbsp; <sup>3</sup>Data61🖤CSIRO &nbsp; <sup>4</sup>Australian National University</h4>
</div>
<br />

  
</div>
<br />

<div align="center">
  <img src="data/assets/teaser.png" alt="result" width="500"/>
</div>

This repo is the official implementation of the CVPR 2023 paper: [Distilling Self-Supervised Vision Transformers for Weakly-Supervised Few-Shot Classification & Segmentation](https://openaccess.thecvf.com/content/CVPR2023/papers/Kang_Distilling_Self-Supervised_Vision_Transformers_for_Weakly-Supervised_Few-Shot_Classification__Segmentation_CVPR_2023_paper.pdf).



## Environmnet installation
This project is built upon the following environment:
* [Ubuntu 18.04](https://ubuntu.com/download)
* [Python 3.10](https://pytorch.org)
* [CUDA 11.0](https://developer.nvidia.com/cuda-toolkit)
* [PyTorch 1.12.0](https://pytorch.org)

The package requirements can be installed via `environment.yml`, which includes
* [`pytorch`](https://pytorch.org)==1.12.0
* [`torchvision`](https://pytorch.org/vision/stable/index.html)==0.13.0
* [`cudatoolkit`](https://developer.nvidia.com/cuda-toolkit)==11.3
* [`pytorch-lightning`](https://www.pytorchlightning.ai/)==1.6.5
* [`einops`](https://einops.rocks/pytorch-examples.html)==0.6.0
```bash
conda env create --name pytorch1.12 --file environment.yml -p YOURCONDADIR/envs/pytorch1.12
conda activate pytorch1.12
```
Make sure to replace `YOURCONDADIR` in the installation path with your conda dir, e.g., `~/anaconda3`

## Datasets
* [PASCAL VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) and [SBD](http://home.bharathh.info/pubs/codes/SBD/download.html)
* [Microsoft COCO 2014](https://cocodataset.org/#download)

Download the datasets by following the file structure below and set `args.datapath=YOUR_DATASET_DIR`:

```
    YOUR_DATASET_DIR/
    ├── VOC2012/
    │   ├── Annotations/
    │   ├── JPEGImages/
    │   ├── ...
    ├── COCO2014/
    │   ├── annotations/
    │   ├── train2014/
    │   ├── val2014/
    │   ├── ...
    ├── ...
```



## Training with pixel-level supervision
```bash
python main.py --datapath YOUR_DATASET_DIR \
               --benchmark {pascal, coco} \
               --logpath YOUR_DIR_TO_SAVE_CKPT \
               --fold {0, 1, 2, 3} \
               --sup mask
```

## Training with image-level supervision
```bash
python main.py --datapath YOUR_DATASET_DIR \
               --benchmark {pascal, coco} \
               --logpath YOUR_DIR_TO_SAVE_CKPT \
               --fold {0, 1, 2, 3} \
               --sup pseudo
```


## Model checkpoints 
Currently, trained checkpoints are unavailable, but they are expected to be available by the end of June 2023.



## :scroll: BibTex source
If you find our code or paper useful, please consider citing our paper:

```BibTeX
@inproceedings{kang2023distilling,
  title={Distilling Self-Supervised Vision Transformers for Weakly-Supervised Few-Shot Classification \& Segmentation},
  author={Kang, Dahyun and Koniusz, Piotr and Cho, Minsu and Murray, Naila},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023}
}
```


