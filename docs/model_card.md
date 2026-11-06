# MEDVISION AI Model Card

## Model
The project uses an EfficientNetB0 image backbone pretrained on ImageNet, followed by global average pooling, dropout, a dense layer, and a four-class softmax head.

## Input and output
- Input: RGB image resized to 224 × 224.
- Output labels: `glioma`, `meningioma`, `notumor`, `pituitary`.
- The output scores are softmax values and are not calibrated probabilities.

## Intended use
Academic experimentation, demonstration, and research workflow development only.

## Out-of-scope use
The model is not clinically validated and must not be used to diagnose, rule out disease, triage patients, recommend treatment, or replace a qualified radiologist.

## Explainability
Grad-CAM is an approximate visualization of model activation. It is not tumour segmentation, localization, or proof of causal reasoning.

## Evaluation caveats
The reported metrics are notebook validation results. The grouping heuristic is filename-based and not patient-level. The notebook includes exploratory use of the dataset Testing folder, so no pristine independent test result is claimed.
