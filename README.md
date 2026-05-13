# Visual Recognition: Product Search Engine & Apparel Detection

This repository contains our comprehensive computer vision pipeline for the Visual Recognition course. It is divided into two main phases: 
1. **Final Project:** A Visual Product Search Engine using YOLO, BLIP-2, and CLIP.
2. **Mini-Project 1:** Multi-label classification, object detection, and instance segmentation of apparel.

## Team Members
* **Shrujan Teja** (IMT2023599)
* **Sathvik Jakkampudi** (IMT2023124)
* **Rajdeep Saha** (IMT2023600)
* **Nachiappan N** (IMT2023605)

---

## Final Project: Visual Product Search Engine

### Problem Statement
Online shoppers often struggle to find specific clothing items using text alone due to language mismatches and inconsistent seller tags. We built a **Query-by-Image** product search system. A user uploads an image, selects the desired clothing region (Upper Body, Lower Body, or Full Body), and the system returns visually and semantically similar products from the catalog.

### System Architecture
* **Localization (YOLOv8):** Detects and crops the primary clothing item based on user selection (Upper/Lower/Full).
* **Semantic Captioning (BLIP-2):** Generates a rich, consistent text description of the cropped item.
* **Multimodal Embedding (CLIP):** Fuses the visual features of the crop and the semantic features of the caption into a single vector.
* **Vector Search (FAISS/HNSW):** Rapidly retrieves the top-K similar items from the indexed catalog.

### Ablation Studies & Configurations
We tested three configurations to optimize retrieval metrics (Recall@K, NDCG@K, mAP@K):
* `Config-A`: Vision-only baseline (CLIP image embeddings).
* `Config-B`: Frozen CLIP + Frozen BLIP-2 (Late fusion of text and image modalities). **(Best Performing Model)**
* `Config-C`: Fine-tuned CLIP + Frozen BLIP-2.

### Streamlit Frontend Features
* Upload a custom query image.
* **Target Selection:** Choose between searching for `Upper Body`, `Lower Body`, or `Full Body` apparel.
* **Interactive Cropping:** System highlights the YOLO-detected bounding box and asks for user confirmation ("Confirm Crop" or "Re-crop").
* **Retrieval Display:** Shows the top-K matching products alongside similarity scores and generated metadata.

---

## Mini-Project 1: Multi-Object Apparel Detection

*Located in the root project folders, addressing Tasks 3.1 and 3.2 on the DeepFashion2 dataset.*

### Highlights
* **Multi-Label Classification:** ResNet-50 / EfficientNet-B0 / MobileNetV3. Addressed class imbalance using `pos_weight` in BCEWithLogitsLoss.
* **Detection & Segmentation:** YOLOv8-seg, Mask R-CNN, U-Net. Addressed instance-level imbalance using custom Inverse-Frequency Weighting and monkey-patching the YOLO dataloader with a `WeightedRandomSampler`.

---

## Repository Structure

```text
├── Final_Project_Retrieval/
│   ├── Ablation_Studies/
│   │   ├── Config-A.ipynb  # Vision-Only
│   │   ├── Config-B.ipynb  # Frozen Multimodal Fusion (Best Model)
│   │   └── Config-C.ipynb  # Fine-Tuned Multimodal
│   └── app/
│       ├── app.py          # Streamlit UI
│       └── requirements.txt
├── Data preprocessing/     # (MP1) Scripts for Train/Val/Test splits
├── Multi_Label_Scratch/    # (MP1) Classification models from scratch
├── Multi_Label_Fine-tuning/# (MP1) Transfer learning for classification
├── Detection_Segmentation_Scratch/     # (MP1) Detection models from scratch
├── Detection_Segmentation_Fine-tuning/ # (MP1) Transfer learning for detection
└── README.md
