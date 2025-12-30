import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # hides INFO logs (keeps errors)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # optional: stops oneDNN message
import sys
import json
import numpy as np
import warnings
from pathlib import Path
warnings.filterwarnings("ignore", category=FutureWarning)

import tensorflow as tf


PLANT_MODEL_PATH = Path("backend/disease_model/models/plant_model.keras")
PLANT_NAMES_PATH = Path("backend/disease_model/models/plant_class_names.json")
IMG_SIZE = (224, 224)
THRESHOLD = 0.50  # optional for "not recognised"

def preprocess_image(img_path: Path):
    img = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img).astype("float32")  # keep 0..255
    arr = np.expand_dims(arr, axis=0)
    return arr

def main():
    if len(sys.argv) < 2:
        print("Usage: python backend/disease_model/src/infer_plant.py <path_to_image>")
        sys.exit(1)

    img_path = Path(sys.argv[1])
    if not img_path.exists():
        print(f"Error: image not found at {img_path}")
        sys.exit(1)

    if not PLANT_MODEL_PATH.exists():
        print(f"Error: plant model not found at {PLANT_MODEL_PATH}")
        sys.exit(1)

    with open(PLANT_NAMES_PATH, "r") as f:
        plant_names = json.load(f)

    model = tf.keras.models.load_model(PLANT_MODEL_PATH)

    x = preprocess_image(img_path)
    probs = model.predict(x, verbose=0)[0]

    pred_idx = int(np.argmax(probs))
    pred_plant = plant_names[pred_idx]
    conf = float(probs[pred_idx])

    print("\nImage:", img_path)
    if conf < THRESHOLD:
        print("Plant: PLANT_NOT_RECOGNISED")
        print("Confidence:", f"{conf:.4f}")
        return

    print("Plant:", pred_plant)
    print("Confidence:", f"{conf:.4f}")

    top2 = np.argsort(probs)[::-1][:2]
    print("\nTop 2:")
    for i in top2:
        print(f"  {plant_names[int(i)]}: {float(probs[int(i)]):.4f}")

if __name__ == "__main__":
    main()
