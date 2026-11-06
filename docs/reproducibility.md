# Reproducibility Guide

## Environment
Create an isolated Python environment and install the packages listed in `requirements.txt`.

## Data
Obtain the MRI dataset from its authorized source and place it under `data/medvision_dataset/Training/`, with one directory per class.

## Training
```bash
python -m training.train_model --data-dir data/medvision_dataset --output-dir models
```

The training script uses a fixed `random_state=42` for the group split and a fixed shuffle seed for the training dataset. GPU availability, package versions, and hardware can still affect numerical results.

## Evaluation
```bash
python -m training.evaluate_model --data-dir data/medvision_dataset --model models/MEDVISION_AI_FINAL_V2.keras --output results/evaluation.json
```

## Artifacts
Keep trained models and generated evaluation artifacts outside version control unless their size and redistribution terms have been reviewed.
