# Determination of Fauna in Sea Exploration

## Project Overview
Marine biodiversity research often involves analyzing large volumes of underwater images to identify species.  
However, manual classification is time-consuming and prone to errors.  
The goal of this project is to automate the identification of known marine species and filter them out so that novel species can be detected.  
I classified 10 distinct animals using a dataset of 5900 images.

## Model Architecture
- **Initial Layers**:
  - 3×3 Convolution (stride=1, padding=1) + BatchNorm + ReLU
  - 3×3 MaxPooling (stride=2)
- **Three Residual Block Units**:
  - Each with two 3×3 convolutions (padding=1) and batch normalization.
- **Output Layers**:
  - Adaptive average pooling → 1×1 feature vector
  - Fully connected layer for class logits

## Key Features
- **Input Size**: 150×150
- **Optimizer**: Adam with weight_decay
- **Loss Function**: Cross-Entropy Loss with `label_smoothing=0.1`
- **Learning Rate Scheduler**: Cosine Annealing LR Scheduler
- **Number of Classes**: 10
- **Framework**: PyTorch
- **Compute**: Google Colab and Kaggle GPU

**Note**:  
Low validation accuracy is expected due to noisy features like landmass, etc.

## Dataset
- **Source**: Sea-Animal-Classification dataset (Kaggle)
- The dataset was not split initially.
- Manually reduced from 22 classes to 10 classes.

## Additional Notes
- **Validation Accuracy**: ~50% despite using CosineAnnealingLR and multiple architecture tweaks.
- **Expected data for testing**: For predict.py file, the classes are sorted based on training datasetpath. It can be changed to classes inside ./data folder if above thing doesn't work.
