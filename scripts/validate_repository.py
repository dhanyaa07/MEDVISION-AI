from pathlib import Path
import ast
import sys

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_FILES = [
    "app.py",
    "requirements.txt",
    "README.md",
    "training/train_model.py",
    "training/evaluate_model.py",
    "notebooks/MEDVISION_AI_original.ipynb",
    "docs/methodology.md",
    "docs/limitations.md",
]

missing = [name for name in REQUIRED_FILES if not (ROOT / name).is_file()]
if missing:
    print("Missing required files:")
    for name in missing:
        print(f" - {name}")
    sys.exit(1)

python_files = [
    ROOT / "app.py",
    ROOT / "training" / "train_model.py",
    ROOT / "training" / "evaluate_model.py",
]
for path in python_files:
    ast.parse(path.read_text(encoding="utf-8"))
    print(f"Syntax OK: {path.relative_to(ROOT)}")

requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
for package in ("tensorflow", "gradio", "numpy", "pandas", "scikit-learn", "reportlab"):
    if package not in requirements:
        print(f"Missing requirement declaration: {package}")
        sys.exit(1)

print("Repository structure and dependency declarations look consistent.")
