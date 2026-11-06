# Repository preparation status

- Original Colab notebook copied unchanged to `notebooks/MEDVISION_AI_original.ipynb`.
- Final application cell copied to `app.py`; model path and Gradio share behaviour are environment-configurable.
- Training pipeline organized from the notebook's final group-aware split and EfficientNetB0 training/fine-tuning cells.
- Evaluation script added.
- README, requirements, limitations, methodology and 60-commit roadmap added.
- Trained model and MRI dataset intentionally excluded from Git.

Before public release:
1. Obtain/confirm rights to redistribute the dataset and model.
2. Put the trained `.keras` model in `models/` locally (it is ignored by Git).
3. Install requirements in a clean environment.
4. Run the app and test all four classes, Grad-CAM, report and PDF download.
5. Check the generated PDF and README claims.
6. Review the notebook for private paths, credentials, and any sensitive data before publishing.
