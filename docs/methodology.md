# Methodology

## Classification task
The model predicts one of four dataset labels: glioma, meningioma, notumor, or pituitary.

## Input pipeline
Images are decoded as RGB, resized to 224 × 224, batched, and prefetched using `tf.data`.

## Model
EfficientNetB0 pretrained on ImageNet is used as a feature extractor. A global average pooling layer, dropout, dense layer, dropout, and four-unit softmax output head are attached. The training pipeline first freezes the backbone and then fine-tunes the final 30 backbone layers while keeping Batch Normalization layers frozen.

## Data split
The notebook uses `StratifiedGroupKFold` with filename-derived groups for recognized meningioma original/augmented pairs. Other images are treated as individual groups by the current grouping heuristic. This is not patient-level grouping.

## Explainability
Grad-CAM is computed from the EfficientNetB0 `top_conv` activation and gradients of the selected class score. The heatmap is an approximate explanation, not segmentation.
