# Inference Workflow

1. The user supplies an MRI image through the Gradio interface.
2. The application prepares the image for the model input.
3. The loaded Keras model returns four class scores.
4. The application selects the highest-scoring dataset category.
5. Grad-CAM produces an approximate activation visualization for the selected class.
6. The report combines model output, technical image measurements, and explanatory context.
7. The PDF export packages the report and relevant visual artifacts.

The class score is not a calibrated probability. The activation map is not a tumour boundary or clinical localization. The workflow does not infer tumour size, grade, cause, prognosis, or treatment.
