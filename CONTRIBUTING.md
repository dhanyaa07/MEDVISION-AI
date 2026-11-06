# Contributing

## Development setup
Create a virtual environment and install `requirements.txt`.

## Before submitting changes
- Keep dataset files, trained model weights, credentials, and private information out of Git.
- Preserve the four declared class labels and their order unless the model and documentation are updated together.
- Run `python scripts/validate_repository.py`.
- Run `python -m unittest discover -s tests -v`.
- Review documentation for accurate validation and medical-safety claims.

## Research integrity
Do not describe validation metrics as independent clinical performance. Document split methodology, dataset limitations, and any changes to training configuration.
