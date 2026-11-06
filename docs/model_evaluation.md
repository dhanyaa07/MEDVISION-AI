# Model evaluation

The selected notebook model reports 90.99% validation accuracy, 91.13% macro precision, 90.99% macro recall, 90.98% macro F1, and 90.98% weighted F1 on 1,121 validation images.

Class-wise notebook results:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Glioma | 94.92% | 86.48% | 90.50% | 281 |
| Meningioma | 83.04% | 83.93% | 83.48% | 280 |
| No tumor | 97.13% | 96.79% | 96.96% | 280 |
| Pituitary | 89.44% | 96.79% | 92.97% | 280 |

The dataset test folder was used exploratorily in notebook experiments. Therefore, these figures must not be described as an untouched independent test result. The group split is based on a filename heuristic for identified meningioma original/augmented pairs, not patient identity. No clinical validation is established.
