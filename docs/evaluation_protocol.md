# Evaluation Protocol

1. Build the image manifest from the four class folders under `Training`.
2. Assign integer labels using the declared `CLASS_NAMES` order.
3. Derive filename groups for recognized meningioma original/augmented pairs.
4. Create a `StratifiedGroupKFold` split with five folds and `random_state=42`; use the first fold.
5. Check that no group occurs in both training and validation subsets.
6. Evaluate predictions against the validation labels and report accuracy, per-class precision/recall/F1, and a confusion matrix.

## Interpretation
This is a reproducible project validation procedure, not patient-level validation. It does not establish clinical performance or generalization to an external hospital population.
