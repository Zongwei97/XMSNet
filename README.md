# Object Segmentation by Mining Cross-Modal Semantics, ACMMM'23

Official PyTorch implementaton of ACMMM'23 paper [Object Segmentation by Mining Cross-Modal Semantics](https://arxiv.org/pdf/2305.10469.pdf)


# Abstract

Multi-sensor clues have shown promise for object segmentation, but inherent noise in each sensor, as well as the calibration error in practice, may bias the segmentation accuracy. In this paper, we propose a novel approach by mining the Cross-Modal Semantics to guide the fusion and decoding of multimodal features, with the aim of controlling the modal contribution based on relative entropy. We explore semantics among the multimodal inputs in two aspects: the modality-shared consistency and the modality-specific variation. Specifically, we propose a novel network, termed XMSNet, consisting of (1) all-round attentive fusion (AF), (2) coarse-to-fine decoder (CFD), and (3) cross-layer self-supervision. On the one hand, the AF block explicitly dissociates the shared and specific representation and learns to weight the modal contribution by adjusting the \textit{proportion, region,} and \textit{pattern}, depending upon the quality. On the other hand,  our CFD initially decodes the shared feature and then refines the output through specificity-aware querying. Further, we enforce semantic consistency across the decoding layers to enable interaction across network hierarchies, improving feature discriminability. Exhaustive comparison on eleven datasets with depth or thermal clues, and on two challenging tasks, namely salient and camouflage object segmentation, validate our effectiveness in terms of both performance and robustness. 

# Train and Test

Please follow the training, inference, and evaluation steps:

```
python train.py
python test.py
python test_evaluation_maps.py
```
Make sure that you have changed the path to your dataset in the [config file](https://github.com/Zongwei97/XMSNet/blob/main/Code/utils/options.py) and in the abovementioned Python files.

We use the same evaluation protocol as [here](https://github.com/taozh2017/SPNet/blob/main/test_evaluation_maps.py)


# Citation

If you find this repo useful, please consider citing:

```
@INPROCEEDINGS{wu2023object,
  title={Object Segmentation by Mining Cross-Modal Semantics},
  author={Wu, Zongwei and Wang, Jingjing and Zhou, Zhuyun and An, Zhaochong and Jiang, Qiuping and Demonceaux, C{\'e}dric and Sun, Guolei and Timofte, Radu},
  booktitle={ACMMM}, 
  year={2023},
}
  
```


# Related works
- ICCV 23 - Source-free Depth for Object Pop-out [[Code](https://github.com/Zongwei97/PopNet)]
- TIP 23 - HiDANet: RGB-D Salient Object Detection via Hierarchical Depth Awareness [[Code](https://github.com/Zongwei97/HIDANet)]
- 3DV 22 - Robust RGB-D Fusion for Saliency Detection [[Code](https://github.com/Zongwei97/RFnet)]
- 3DV 21 - Modality-Guided Subnetwork for Salient Object Detection [[Code](https://github.com/Zongwei97/MGSnet)]

