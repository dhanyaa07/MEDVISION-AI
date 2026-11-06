# MEDVISION AI Dataset Card

## Task
Four-class classification of brain MRI images using the dataset labels `glioma`, `meningioma`, `notumor`, and `pituitary`.

## Repository representation
The repository contains the original notebook and training/application code. The MRI image files are intentionally excluded from Git because redistribution rights must be checked.

## Directory layout expected by the training script
```text
data/medvision_dataset/
└── Training/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

The training script reads the `Training` directory. The notebook also records exploratory use of the dataset `Testing` directory; that use is not represented as an untouched independent test.

## Known limitations
- The split is filename-group-aware for recognized meningioma original/augmented pairs, not patient-level.
- Dataset provenance, consent, licensing, and redistribution permissions must be verified independently.
- Dataset labels are reproduced as supplied and are not clinical ground truth guarantees.
