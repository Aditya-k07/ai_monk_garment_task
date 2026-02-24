# Multilabel Image Classification – Fashion Dataset
dataset Link - https://drive.google.com/drive/u/0/folders/1ZAShdBc1dsDaYanVLQTASUOZbXbViSTU
Google colab link - https://colab.research.google.com/drive/1TrS1GAZQMTCRCCjATQUD4OLXXFbOgxhD?usp=sharing
Note - use dataset path from the google drive for ease and the fine tuning part may take a long time. 

## Problem

- Each image has 4 attributes.
- Labels can be 1 (present), 0 (absent), or NA (unknown).
- This is a multilabel classification problem with missing labels and class imbalance.

---

## Approach

- Used EfficientNet-B0 pretrained on ImageNet.
- Replaced final classifier with 4 output neurons.
- Used BCEWithLogitsLoss for multilabel learning.
- Applied sigmoid during inference.

---

## Handling NA Labels

- Created a mask tensor.
- Loss computed only on known labels.
- NA labels ignored during training.

---

## Handling Class Imbalance

- Computed pos_weight for each attribute:
  
  pos_weight = negatives / positives

- Passed pos_weight to BCEWithLogitsLoss.

---

## Training Strategy

- 80% training, 20% validation split.
- Phase 1: Frozen backbone, trained classifier.
- Phase 2: Unfroze backbone and fine-tuned with lower learning rate.
- Plotted training loss (iteration_number vs training_loss).

---

## Outputs

- multilabel_model.pth (trained weights)
- loss_curve.png (training loss plot)

---

## Possible Improvements

- Add F1-score and mAP evaluation.
- Tune threshold per attribute.
- Use Focal Loss for imbalance.
- Add stronger augmentations.
- Apply early stopping.
