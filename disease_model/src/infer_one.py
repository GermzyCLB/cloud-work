import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
import json

MODEL_PATH = Path("backend/disease_model/models/disease_model.keras")
DATA_DIR = Path("backend/disease_model/data/processed/train")  # used only to get class names
IMG_SIZE = (224, 224)
CLASS_NAMES_PATH = Path("backend/disease_model/models/class_names.json")


def load_class_names():
    """
    Load class names saved during training to keep the same label index mapping.
    """
    if not CLASS_NAMES_PATH.exists():
        raise RuntimeError(
            f"{CLASS_NAMES_PATH} not found. Run predict.py once to generate it."
        )

    with open(CLASS_NAMES_PATH, "r") as f:
        return json.load(f)

def preprocess_image(img_path: Path):
    """
    Load an image and format it as a batch tensor for the model.
    NOTE: Do NOT call preprocess_input here because the saved model
    already contains MobileNetV2 preprocessing inside it.
    """
    img = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img).astype("float32")  # keep 0..255
    arr = np.expand_dims(arr, axis=0)  # (1, H, W, C)
    return arr
    

def main():

    print("infer_one.py started")

    if len(sys.argv) < 2:
        print("Usage: python backend/disease_model/src/infer_one.py <path_to_image>")
        print("Example: python backend/disease_model/src/infer_one.py backend/disease_model/data/processed/test/Tomato_healthy/0a1b.jpg")
        sys.exit(1)

    img_path = Path(sys.argv[1])

    if not MODEL_PATH.exists():
        print(f"Error: model not found at {MODEL_PATH}. Run predict.py first to create it.")
        sys.exit(1)

    if not img_path.exists():
        print(f"Error: image not found at {img_path}")
        sys.exit(1)

    class_names = load_class_names()


    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    x = preprocess_image(img_path)
    probs = model.predict(x, verbose=0)[0]

    pred_idx = int(np.argmax(probs))
    pred_class = class_names[pred_idx]
    confidence = float(probs[pred_idx])
    
    #confidence threshold check ..if below 0.5 in confidence then it will just return for user
    #to try again

    image_threshold=0.50
    
    if confidence < image_threshold:
        print("\nImage ",img_path)
        print("\nprediction: IMAGE_NOT_RECOGNISED")
        print("Confidence:", f"{confidence:.4f}")
        return
    
    

    print("\nImage:", img_path)
    print("Prediction:", pred_class)
    print("Confidence:", f"{confidence:.4f}")

    # Optional: show top 3 predictions (nice for report screenshots)
    top3 = np.argsort(probs)[::-1][:3]
    print("\nTop 3:")
    for i in top3:
        print(f"  {class_names[int(i)]}: {float(probs[int(i)]):.4f}")

if __name__ == "__main__":
    main()
