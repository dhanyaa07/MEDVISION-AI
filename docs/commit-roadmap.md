# 60-commit development roadmap

These are planned commits, not commits already made. Create each commit only after the corresponding code/documentation change is implemented, reviewed, and working. Do not create empty commits or backdate history.

01. `chore: initialize MEDVISION AI repository` — Add README, .gitignore, requirements and repository metadata.
02. `docs: document project scope and intended use` — Describe the academic research objective and medical safety boundary.
03. `docs: add initial repository architecture` — Document application, training, notebooks, model and results folders.
04. `chore: add Python package markers` — Add package initialization where required.
05. `chore: define supported runtime dependencies` — Record tested dependency ranges and installation instructions.
06. `data: document four-class dataset layout` — Document expected Training/Testing class folders.
07. `data: add dataset path configuration` — Support configurable dataset root rather than hard-coded Colab paths.
08. `data: implement supported image discovery` — Collect JPG, JPEG and PNG training images by class.
09. `data: validate dataset class directories` — Fail clearly when expected class folders are missing.
10. `data: record class-index mapping` — Keep glioma, meningioma, notumor and pituitary ordering explicit.
11. `data: implement meningioma filename grouping` — Group recognized original and augmented filename pairs.
12. `data: add stratified group-aware split` — Reproduce the notebook's StratifiedGroupKFold split.
13. `test: assert no groups cross split boundaries` — Verify the grouping split invariant.
14. `docs: explain limits of filename-based grouping` — Clarify that grouping is not patient-level validation.
15. `data: implement RGB image decoding` — Use TensorFlow image decoding with three channels.
16. `data: standardize model input dimensions` — Resize images to 224 by 224 pixels.
17. `data: build tf.data training pipeline` — Batch and prefetch training examples.
18. `data: add deterministic training shuffle` — Use the notebook's fixed shuffle seed.
19. `model: implement EfficientNetB0 backbone` — Load ImageNet-pretrained EfficientNetB0 without its top layer.
20. `model: add global average pooling head` — Implement the notebook's classification head.
21. `model: add dropout regularization` — Preserve the notebook's 0.3 and 0.2 dropout layers.
22. `model: add four-class softmax output` — Create the four-output classification layer.
23. `model: compile frozen-backbone model` — Use Adam and sparse categorical cross-entropy.
24. `train: add early stopping callback` — Restore best weights based on validation loss.
25. `train: add best-checkpoint saving` — Save the best stage-A model.
26. `train: implement initial transfer-learning stage` — Train the frozen-backbone model.
27. `train: reload best stage-A checkpoint` — Start fine-tuning from the saved best checkpoint.
28. `model: unfreeze final 30 backbone layers` — Reproduce the notebook's fine-tuning scope.
29. `model: freeze BatchNormalization layers` — Preserve stable BatchNormalization behaviour during fine-tuning.
30. `model: compile fine-tuning stage` — Use the lower learning rate from the notebook.
31. `train: add fine-tuning callbacks` — Save best fine-tuned weights.
32. `train: implement fine-tuning stage` — Train the selected backbone layers.
33. `evaluate: load selected final checkpoint` — Evaluate the saved fine-tuned model.
34. `evaluate: generate validation predictions` — Produce class scores for the validation dataset.
35. `evaluate: add class-wise precision recall and F1` — Save classification report output.
36. `evaluate: add confusion matrix output` — Record the four-class confusion matrix.
37. `evaluate: export evaluation JSON` — Persist validation metrics in machine-readable form.
38. `docs: record notebook validation metrics` — Document the reported validation scores and caveats.
39. `app: preserve final Gradio application` — Move the final notebook application cell into app.py.
40. `app: configure portable model loading` — Read model path from MEDVISION_MODEL_PATH.
41. `app: define stable class ordering` — Keep inference class order aligned with training.
42. `explainability: initialize EfficientNet feature model` — Expose top_conv activation for Grad-CAM.
43. `explainability: implement Grad-CAM gradients` — Compute class-specific gradients and channel weights.
44. `explainability: normalize Grad-CAM heatmap` — Preserve the notebook heatmap normalization.
45. `analysis: add image-quality measurements` — Report brightness, contrast, sharpness and intensity range.
46. `report: add class-specific medical background` — Preserve category-level medical knowledge as general context.
47. `report: add classification score section` — Report predicted class and model score.
48. `report: add technical image information` — Include dimensions, format and pixel statistics.
49. `report: add explainability caveats` — State that activation is not tumour segmentation.
50. `report: add research limitations section` — Document patient-level and clinical validation limits.
51. `pdf: add PDF report document generation` — Generate a structured report with ReportLab.
52. `pdf: include uploaded MRI in report` — Embed the original uploaded image.
53. `pdf: include Grad-CAM visualization` — Embed the explainability image.
54. `pdf: include validation metrics and confusion matrix` — Include model-level evaluation in the PDF.
55. `ui: add MRI upload component` — Preserve Gradio PIL image input.
56. `ui: add classification result display` — Display class score distribution.
57. `ui: add Grad-CAM output panel` — Display the generated overlay.
58. `ui: add report text output` — Display the comprehensive report in the app.
59. `ui: add PDF generation and download control` — Connect the PDF report workflow.
60. `docs: finalize setup and reproducibility guide` — Document local run, training, evaluation and model placement.
