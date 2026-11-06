# MEDVISION AI

**Brain MRI classification and explainable AI research prototype**

MEDVISION AI is an academic deep-learning project that classifies uploaded brain MRI images into four dataset categories and provides a Grad-CAM visualization, an expanded educational analysis report, and a downloadable PDF report.

> **Medical safety:** This is an academic research prototype, not a clinically validated diagnostic system. It must not be used to diagnose, rule out disease, guide treatment, or replace radiologist interpretation.

## Project capabilities

- Four-class image classification: `glioma`, `meningioma`, `notumor`, and `pituitary`.
- EfficientNetB0 transfer-learning model.
- Grad-CAM visualization for approximate model explainability.
- Basic image-quality measurements (brightness, contrast, sharpness).
- Category-specific general medical background in the report.
- PDF report containing classification results, image, Grad-CAM, and model evaluation information.
- Gradio interface for interactive use.

## Repository structure

```text
MEDVISION-AI/
├── app.py
├── requirements.txt
├── training/
│   ├── train_model.py
│   └── evaluate_model.py
├── notebooks/
│   └── MEDVISION_AI_original.ipynb
├── models/
├── results/
├── docs/
└── assets/
```

## Model and dataset

The notebook records a dataset with 7,200 images across the `Training` and `Testing` folders, with four categories. The application expects the trained Keras model at:

`models/MEDVISION_AI_FINAL_V2.keras`

The model file and dataset are not included in this repository because they are large artifacts and redistribution permissions must be checked. Obtain the dataset from its authorized source and place it under `data/medvision_dataset/` using the expected folder structure.

```text
data/medvision_dataset/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

## Recorded validation results

The notebook reports the following validation results for its selected model:

| Metric | Result |
|---|---:|
| Accuracy | 90.99% |
| Macro precision | 91.13% |
| Macro recall | 90.99% |
| Macro F1-score | 90.98% |
| Weighted F1-score | 90.98% |

These are notebook-reported validation metrics, not a claim of independent clinical performance. The split groups identified meningioma original/augmented filename pairs; it is not patient-level validation. The notebook also includes exploratory use of the dataset's `Testing` folder, so this project does not claim a pristine, untouched external test evaluation. Softmax scores are not calibrated probabilities.

## Installation

Use a Python environment compatible with the packages in `requirements.txt`.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Place the trained model in `models/MEDVISION_AI_FINAL_V2.keras`, or set `MEDVISION_MODEL_PATH` to its location.

## Run the application

```bash
python app.py
```

The Gradio app runs locally by default. To enable a share link in a suitable environment, set `GRADIO_SHARE=true` before starting the app.

Windows PowerShell:

```powershell
$env:GRADIO_SHARE="true"
python app.py
```

## Training

The training pipeline uses the notebook's group-aware split strategy for identified meningioma original/augmented pairs, image resizing to 224 × 224, EfficientNetB0 transfer learning, and fine-tuning.

```bash
python -m training.train_model --data-dir data/medvision_dataset --output-dir models
```

Training downloads ImageNet weights if they are not cached. Training may require a GPU and substantial memory/time.

Evaluate an existing model:

```bash
python -m training.evaluate_model --data-dir data/medvision_dataset --model models/MEDVISION_AI_FINAL_V2.keras
```

## Explainability and limitations

- Grad-CAM highlights regions that influenced a model prediction; it is not tumour segmentation or localization.
- Image brightness, contrast, sharpness, and activation-area values are technical image/model measurements, not tumour measurements.
- A single uploaded image cannot establish tumour size, volume, grade, cause, extent, or patient prognosis.
- The `notumor` class means the model matched that dataset label; it does not rule out all abnormalities or establish overall brain health.
- Patient-level independent validation and clinical validation have not been established.

## Research reproducibility

The original Colab notebook is retained in `notebooks/`. It documents exploratory experiments and iterations; the final application implementation is preserved in `app.py`. Some earlier notebook cells are experiments or superseded versions and should not be interpreted as the final deployed pipeline.

## License and data use

No license is asserted for the MRI dataset or trained model. Check the original dataset and model redistribution terms before publishing or sharing them. Add a repository license only after confirming ownership and permissions.

## Acknowledgements

Built as an academic AI/ML project using TensorFlow/Keras, EfficientNetB0, OpenCV, Grad-CAM, ReportLab, and Gradio.

## Repository checks

Run lightweight repository checks without loading TensorFlow or the MRI dataset:

```bash
python scripts/validate_repository.py
python -m unittest discover -s tests -v
```

These checks validate repository structure, source syntax, and selected training configuration contracts. They do not replace end-to-end testing with the authorized dataset and trained model. See `docs/testing.md`.
