# Industrial_Surface_Defect_Classification
End-to-end computer vision pipeline for industrial steel surface defect classification using CNNs and transfer learning.
# Industrial Surface Defect Classification using CNNs

This project focuses on classifying industrial steel surface defects using deep learning.  
A complete computer vision pipeline was built using the NEU surface defect dataset, starting from raw XML annotations to model evaluation and analysis.

---

## Problem Statement
Automated surface inspection is critical in manufacturing industries.  
The goal of this project is to classify steel surface images into one of six defect categories using convolutional neural networks.

---

## Dataset
- Dataset: **NEU Surface Defect Dataset**
- Classes:
  - Crazing
  - Inclusion
  - Patches
  - Pitted Surface
  - Rolled-in Scale
  - Scratches
- The original dataset contains XML annotations and nested image folders.
- A preprocessing pipeline was implemented to convert the dataset into a classification-ready format.

> Note: The dataset is not included in this repository.

---

## Methodology

### 1. Data Preprocessing
- Parsed XML annotations to extract class labels
- Handled missing files and naming inconsistencies
- Converted dataset into `ImageFolder` format
- Applied grayscale conversion and resizing

### 2. Baseline CNN
- Designed a convolutional neural network from scratch using PyTorch
- Applied domain-specific data augmentation
- Achieved ~77% validation accuracy

### 3. Transfer Learning (ResNet18)
- Used a pretrained ResNet18 model
- Modified the network for grayscale input and 6-class output
- Trained only the classifier layers
- Achieved ~83% validation accuracy without fine-tuning deeper layers.

---

## Results

### Validation Accuracy Comparison
| Model | Validation Accuracy |
|------|---------------------|
| CNN (from scratch) | ~77% |
| ResNet18 (transfer learning) | ~83% |

### Confusion Matrix
The confusion matrix highlights common misclassifications between visually similar defect types such as scratches and crazing.

---

## Visualizations
Key plots generated in this project include:
- Training Loss vs Epoch
- Validation Accuracy vs Epoch
- Confusion Matrix

(Plots are available in the `Figures/` directory.)

---

## Tools & Libraries
- Python
- PyTorch
- Torchvision
- Matplotlib
- NumPy

---

## Key Learnings
- Handling real-world industrial datasets with imperfect annotations
- Importance of data augmentation for model generalization
- Performance trade-offs between training from scratch and transfer learning
- Using confusion matrices for detailed error analysis

---

## Future Work
- Fine-tuning deeper layers of ResNet
- Class-weighted loss to handle imbalance
- Grad-CAM visualization for explainability

---

## Author
**Parth Pardeshi**  
B.Tech, IIT Guwahati
