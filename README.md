# Knowledge Distillation for CNN Model Compression on CIFAR-10

## Overview

This project explores **Knowledge Distillation** and **Model Compression** using PyTorch on the CIFAR-10 image classification dataset.

A pretrained **ResNet18 Teacher Network** is first trained on CIFAR-10. Knowledge is then transferred to smaller custom CNN student architectures to study the trade-off between model size and predictive performance.

The project investigates:

- Teacher–Student Learning
- Knowledge Distillation
- CNN Model Compression
- Accuracy vs Parameter Trade-offs
- Transfer Learning with ResNet18

---

## Dataset

**CIFAR-10**

- 60,000 color images
- 10 classes
- Image size: 32 × 32 × 3
- 50,000 training images
- 10,000 testing images

Classes:

```
airplane, automobile, bird, cat, deer,
dog, frog, horse, ship, truck
```

---

## Methodology

### 1. Teacher Network

A pretrained **ResNet18** model was fine-tuned on CIFAR-10.

Features:

- Transfer Learning
- Data Augmentation
- Cosine Annealing Learning Rate Scheduler
- Adam Optimizer

---

### 2. Student Network

A lightweight custom CNN was designed and trained from scratch.

Architecture:

- Convolution Layers
- Batch Normalization
- Max Pooling
- Fully Connected Classification Head

---

### 3. Compressed Student Network

To aggressively reduce model size, the dense classifier was replaced with:

- Global Average Pooling
- Single Linear Layer

This significantly reduced the number of trainable parameters.

---

### 4. Knowledge Distillation

Knowledge Distillation was implemented using:

- Soft Targets from Teacher Network
- KL-Divergence Loss
- Cross-Entropy Loss
- Temperature Scaling

The distillation objective is:

Distillation Loss:

L = α × CrossEntropyLoss +
    (1 − α) × T² × KLDivergenceLoss

where:

- \(L_{CE}\) = Cross Entropy Loss
- \(L_{KD}\) = Distillation Loss
- \(T\) = Temperature
- \(\alpha\) = Balancing Parameter

---

## Experimental Results

| Model | Accuracy (%) | Parameters |
|---------|---------:|---------:|
| Teacher (ResNet18) | 88.50 | 11,181,642 |
| Student CNN | 87.79 | 2,193,674 |
| Compressed Student CNN | 83.53 | 94,986 |
| Distilled Compressed Student | 83.48 | 94,986 |

---

## Compression Analysis

| Model | Compression vs Teacher |
|---------|---------:|
| Student CNN | 5.10× Smaller |
| Compressed Student CNN | 117.72× Smaller |
| Distilled Compressed Student | 117.72× Smaller |

---

## Key Findings

- The Teacher Network achieved **88.50%** accuracy.
- The custom Student CNN achieved **87.79%** accuracy while using significantly fewer parameters.
- The Compressed Student CNN reduced model size by over **117×** compared to the Teacher Network.
- Knowledge Distillation was successfully implemented and evaluated on the compressed architecture.
- Experimental results suggest that extremely aggressive compression introduced a capacity bottleneck that limited the effectiveness of distillation.

---

## Visualizations

The repository includes:

- Training Loss Curves
- Confusion Matrix
- Accuracy Comparison
- Accuracy vs Model Size Analysis

---

## Technologies Used

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn

---

## Repository Structure

```text
knowledge-distillation-cifar10/
│
├── knowledge_distillation.ipynb
├── README.md
│
└── results/
    ├── loss_curve.png
    ├── confusion_matrix.png
    ├── accuracy_comparison.png
    └── accuracy_vs_parameters.png
```
