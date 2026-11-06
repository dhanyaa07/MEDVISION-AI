# Project Structure

- `app.py`: Gradio application, inference workflow, Grad-CAM, report generation, and PDF export.
- `training/train_model.py`: dataset manifest, group-aware split, TensorFlow input pipeline, EfficientNetB0 training, and validation artifacts.
- `training/evaluate_model.py`: evaluation of a saved model on the configured validation split.
- `notebooks/MEDVISION_AI_original.ipynb`: retained Colab notebook with experiments and application work.
- `tests/`: lightweight repository contract tests.
- `scripts/validate_repository.py`: static repository and syntax validation.
- `docs/`: methodology, limitations, evaluation, reproducibility, governance, and release guidance.

Large model and dataset artifacts are intentionally excluded from version control.
