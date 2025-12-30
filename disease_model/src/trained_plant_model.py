import os
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

DATA_DIR = Path("backend/disease_model/data/processed")
OUT_DIR = Path("backend/disease_model/models")

PLANT_MODEL_OUT = OUT_DIR / "plant_model.keras"
PLANT_NAMES_OUT = OUT_DIR / "plant_class_names.json"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5  # plant model is easy; 3–5 is enough

def make_datasets():
    """
    Loads the existing disease dataset folders, but maps disease labels -> plant labels:
    Potato___* -> 0 (Potato)
    Tomato_*   -> 1 (Tomato)
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / "train",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=True,
        seed=42,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / "val",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=True,
        seed=42,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / "test",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
    )

    disease_class_names = train_ds.class_names  # folder names (13 classes)

    # Build a mapping: disease_class_index -> plant_index
    # 0 = Potato, 1 = Tomato
    mapping = []
    for name in disease_class_names:
        if name.startswith("Potato"):
            mapping.append(0)
        elif name.startswith("Tomato"):
            mapping.append(1)
        else:
            raise ValueError(f"Unknown class prefix in: {name}")

    mapping = tf.constant(mapping, dtype=tf.int32)

    def map_to_plant(images, labels):
        plant_labels = tf.gather(mapping, labels)
        return images, plant_labels

    train_ds = train_ds.map(map_to_plant)
    val_ds = val_ds.map(map_to_plant)
    test_ds = test_ds.map(map_to_plant)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)

    plant_class_names = ["Potato", "Tomato"]
    return train_ds, val_ds, test_ds, plant_class_names

def build_model(num_classes: int):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.10),
    ])

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = augment(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base_model

def evaluate_with_sklearn(model, test_ds, class_names):
    y_true, y_pred = [], []

    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        preds = np.argmax(probs, axis=1)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(preds.tolist())

    print("\n=== classification report (Plant model) ===")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    print("=== confusion matrix (Plant model) ===")
    print(confusion_matrix(y_true, y_pred))

def main():
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    if not (DATA_DIR / "train").exists():
        print("Can't find processed/train. Run train.py first.")
        return

    train_ds, val_ds, test_ds, plant_names = make_datasets()
    print("Plant classes:", plant_names)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(PLANT_NAMES_OUT, "w") as f:
        json.dump(plant_names, f)
    print(f"Saved plant class names to: {PLANT_NAMES_OUT}")

    model, base_model = build_model(num_classes=len(plant_names))
    model.summary()

    # Stage 1: train head
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

    # Stage 2: quick fine-tune last layers (optional but helps)
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(train_ds, validation_data=val_ds, epochs=2)

    model.save(PLANT_MODEL_OUT)
    print(f"\nSaved plant model to: {PLANT_MODEL_OUT}")

    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"\nPlant Test accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")

    evaluate_with_sklearn(model, test_ds, plant_names)

if __name__ == "__main__":
    main()
