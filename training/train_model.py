import os
import re
import json
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import classification_report, confusion_matrix

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
IMG_SIZE = 224
BATCH_SIZE = 32

def build_dataframe(data_dir):
    rows = []
    for label, class_name in enumerate(CLASS_NAMES):
        folder = os.path.join(data_dir, "Training", class_name)
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Missing class folder: {folder}")
        for filename in sorted(os.listdir(folder)):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            path = os.path.join(folder, filename)
            if class_name == "meningioma":
                match = re.match(r"Tr-(?:aug-)?me_(\d+)", filename)
                group = f"meningioma_{match.group(1)}" if match else path
            else:
                group = path
            rows.append({"image": path, "label": label, "group": group})
    if not rows:
        raise ValueError("No supported images found in Training directory.")
    return pd.DataFrame(rows)

def split_data(dataframe):
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, val_idx = next(splitter.split(
        dataframe["image"], dataframe["label"], groups=dataframe["group"]
    ))
    train_df = dataframe.iloc[train_idx].reset_index(drop=True)
    val_df = dataframe.iloc[val_idx].reset_index(drop=True)
    if not set(train_df["group"]).isdisjoint(set(val_df["group"])):
        raise RuntimeError("Group leakage detected between training and validation.")
    return train_df, val_df

def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    return image, label

def create_dataset(dataframe, shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices((
        dataframe["image"].values, dataframe["label"].values
    ))
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(2000, seed=42, reshuffle_each_iteration=True)
    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

def build_model():
    base = tf.keras.applications.EfficientNetB0(
        weights="imagenet", include_top=False, input_shape=(224, 224, 3)
    )
    base.trainable = False
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(4, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=os.getenv("MEDVISION_DATA_DIR", "data/medvision_dataset"))
    parser.add_argument("--output-dir", default="models")
    parser.add_argument("--initial-epochs", type=int, default=10)
    parser.add_argument("--fine-tune-epochs", type=int, default=8)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    dataframe = build_dataframe(args.data_dir)
    train_df, val_df = split_data(dataframe)
    print(f"Training images: {len(train_df)}")
    print(f"Validation images: {len(val_df)}")
    print("Training distribution:\n", train_df["label"].value_counts().sort_index())
    print("Validation distribution:\n", val_df["label"].value_counts().sort_index())

    train_ds = create_dataset(train_df, shuffle=True)
    val_ds = create_dataset(val_df)

    stage_a_path = os.path.join(args.output_dir, "MEDVISION_AI_stageA_best.keras")
    stage_b_path = os.path.join(args.output_dir, "MEDVISION_AI_FINAL_V2.keras")
    callbacks_a = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(stage_a_path, monitor="val_loss", save_best_only=True)
    ]
    model = build_model()
    model.fit(train_ds, validation_data=val_ds, epochs=args.initial_epochs, callbacks=callbacks_a)

    model = tf.keras.models.load_model(stage_a_path)
    base = model.get_layer("efficientnetb0")
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
        loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    callbacks_b = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(stage_b_path, monitor="val_loss", save_best_only=True)
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=args.fine_tune_epochs, callbacks=callbacks_b)

    final_model = tf.keras.models.load_model(stage_b_path)
    probabilities = final_model.predict(val_ds, verbose=1)
    predictions = np.argmax(probabilities, axis=1)
    y_true = val_df["label"].to_numpy()
    report = classification_report(y_true, predictions, target_names=CLASS_NAMES, output_dict=True)
    matrix = confusion_matrix(y_true, predictions)
    evaluation = {
        "model": "EfficientNetB0",
        "split": "StratifiedGroupKFold with filename-based grouping for identified meningioma original/augmented pairs",
        "validation_images": int(len(val_df)),
        "validation_accuracy": float(np.mean(predictions == y_true)),
        "classification_report": report,
        "confusion_matrix": matrix.tolist()
    }
    with open(os.path.join(args.output_dir, "evaluation.json"), "w", encoding="utf-8") as f:
        json.dump(evaluation, f, indent=2)
    with open(os.path.join(args.output_dir, "classes.json"), "w", encoding="utf-8") as f:
        json.dump({str(i): name for i, name in enumerate(CLASS_NAMES)}, f, indent=2)
    print(classification_report(y_true, predictions, target_names=CLASS_NAMES, digits=4))
    print("Saved model and evaluation artifacts to:", args.output_dir)

if __name__ == "__main__":
    main()
