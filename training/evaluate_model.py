import argparse
import json
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from train_model import CLASS_NAMES, build_dataframe, split_data, create_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.getenv("MEDVISION_DATA_DIR", "data/medvision_dataset"))
    parser.add_argument("--model", default=os.getenv("MEDVISION_MODEL_PATH", "models/MEDVISION_AI_FINAL_V2.keras"))
    parser.add_argument("--output", default="results/evaluation.json")
    args = parser.parse_args()
    df = build_dataframe(args.data_dir)
    _, val_df = split_data(df)
    val_ds = create_dataset(val_df)
    model = tf.keras.models.load_model(args.model)
    probs = model.predict(val_ds, verbose=1)
    pred = np.argmax(probs, axis=1)
    truth = val_df["label"].to_numpy()
    report = classification_report(truth, pred, target_names=CLASS_NAMES, output_dict=True)
    matrix = confusion_matrix(truth, pred)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({
            "model": os.path.basename(args.model),
            "validation_images": len(val_df),
            "accuracy": float(np.mean(pred == truth)),
            "classification_report": report,
            "confusion_matrix": matrix.tolist()
        }, f, indent=2)
    print(classification_report(truth, pred, target_names=CLASS_NAMES, digits=4))
    print("Confusion matrix:\n", matrix)

if __name__ == "__main__":
    main()
