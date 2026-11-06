import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAINING_FILE = ROOT / "training" / "train_model.py"
APP_FILE = ROOT / "app.py"


def read_training_tree():
    return ast.parse(TRAINING_FILE.read_text(encoding="utf-8"))


def top_level_assignment(tree, name):
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"Top-level assignment {name!r} was not found")


class RepositoryContractTests(unittest.TestCase):
    def test_application_entrypoint_exists(self):
        self.assertTrue(APP_FILE.is_file())

    def test_training_entrypoint_exists(self):
        self.assertTrue(TRAINING_FILE.is_file())

    def test_class_order_is_explicit(self):
        tree = read_training_tree()
        self.assertEqual(
            top_level_assignment(tree, "CLASS_NAMES"),
            ["glioma", "meningioma", "notumor", "pituitary"],
        )

    def test_image_size_is_224(self):
        self.assertEqual(top_level_assignment(read_training_tree(), "IMG_SIZE"), 224)

    def test_batch_size_is_positive(self):
        self.assertGreater(top_level_assignment(read_training_tree(), "BATCH_SIZE"), 0)

    def test_training_uses_group_aware_split(self):
        source = TRAINING_FILE.read_text(encoding="utf-8")
        self.assertIn("StratifiedGroupKFold", source)
        self.assertIn("groups=dataframe[", source)
        self.assertIn("isdisjoint", source)

    def test_training_uses_efficientnet_b0(self):
        source = TRAINING_FILE.read_text(encoding="utf-8")
        self.assertIn("EfficientNetB0", source)

    def test_training_script_is_syntactically_valid(self):
        ast.parse(TRAINING_FILE.read_text(encoding="utf-8"))

    def test_application_is_syntactically_valid(self):
        ast.parse(APP_FILE.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
