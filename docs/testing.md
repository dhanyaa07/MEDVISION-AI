# Testing Guide

The lightweight checks do not import TensorFlow or load the MRI dataset.

Run repository validation:
```bash
python scripts/validate_repository.py
```

Run contract tests:
```bash
python -m unittest discover -s tests -v
```

These checks cover expected files, declared class order, image size, batch-size validity, group-aware split markers, EfficientNetB0 configuration, and Python syntax.

They do not replace end-to-end tests with the authorized dataset and trained model. Before release, manually verify inference, Grad-CAM, and PDF export in the configured environment.
