# Troubleshooting

## Model file not found
Confirm that `models/MEDVISION_AI_FINAL_V2.keras` exists or set `MEDVISION_MODEL_PATH` to the correct path.

## Dataset folder not found
Confirm that `data/medvision_dataset/Training/` contains the four expected class folders.

## TensorFlow installation problems
Use a supported Python version and install from a clean virtual environment. Consult the TensorFlow installation guidance for the operating system and hardware in use.

## Out-of-memory during training
Reduce the batch size in the training configuration and retry. Record any changed training settings when comparing results.

## Gradio sharing
The application is local by default. Enable sharing only when intentionally publishing a demo and when uploaded image privacy has been considered.
