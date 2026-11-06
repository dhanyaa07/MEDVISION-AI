# Local Application Deployment

## Prerequisites
Install dependencies from `requirements.txt`, then place the trained model at `models/MEDVISION_AI_FINAL_V2.keras` or configure `MEDVISION_MODEL_PATH`.

## Start
```bash
python app.py
```

The application runs locally by default. A public Gradio share link should only be enabled deliberately in an environment where sharing the uploaded medical images is appropriate.

## Configuration
- `MEDVISION_MODEL_PATH`: path to the trained Keras model.
- `GRADIO_SHARE`: set to `true` only when a share link is intentionally required.

Do not expose private patient data through a public demo.
