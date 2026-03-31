# Visual Recognition: Multi-Object Apparel Detection & Instance Segmentation

This repository contains the complete codebase for our Visual Recognition Mini-Project. The objective of this project is to design a comprehensive computer vision pipeline capable of multi-label classification, object detection, and instance segmentation of clothing items within person-centric images.

## Team Members
* **Shrujan Teja** (IMT2023599)
* **Sathvik Jakkampudi** (IMT2023124)
* **Rajdeep Saha** (IMT2023600)
* **Nachiappan N** (IMT2023605)

---

## Project Overview
With the rapid growth of e-commerce, automated visual recognition systems are crucial for identifying and analyzing clothing items in images. This system addresses three core tasks on a subset of the **DeepFashion2** dataset:
1. **Multi-Label Classification:** Assigning an image to one or more of 5 product categories (*Shorts, Skirts, Trousers, Short Sleeve Top, Long Sleeve Top*).
2. **Object Detection:** Locating clothing items by predicting bounding boxes.
3. **Instance Segmentation:** Generating pixel-level masks for each clothing item instance.

---

## Repository Structure

```text
├── Data preprocessing/
│   ├── Sampling/
│   │   └── Data_preprocess_vr.ipynb
│   ├── Top5_Categories/
│   │   └── Data_Preprocess.ipynb
│   └── Train_Test_Validation split/
│       └── top5categories(1).ipynb
├── Preprocessing_classification.ipynb
├── Multi_Label_Scratch/
│   ├── EfficientNetB0/
│   │   └── eff-b0-scratch (1).ipynb
│   ├── MobileNetV3/
│   │   └── mobile-net-scratch-final.ipynb
│   └── Resnet50/
│       └── resnet-scratch.ipynb
├── Multi_Label_Fine-tuning/
│   └── multi_label_classification.ipynb
├── Detection_Segmentation_Scratch/
│   ├── MaskRCNN/
│   │   └── Mask-RCNN_scratch_updated.ipynb
│   ├── U-Net/
│   │   └── u-net-scratch.ipynb
│   └── Yolo/
│       └── yolo-scratch.ipynb
├── Detection_Segmentation_Fine-tuning/
│   ├── MaskRCNN/
│   │   └── mask-rcnn_tl.ipynb
│   ├── U-Net/
│   │   └── u-net-tl.ipynb
│   └── Yolo/
│       └── yolo-tl.ipynb
├── Inference_MP1/                 # Final inference scripts for hidden test set evaluation
│   ├── predictor.py               
│   ├── validator_local.py         
│   └── requirements.txt           
└── README.md
