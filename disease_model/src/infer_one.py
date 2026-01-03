import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
import json

#----
# this section holds the model and data configs
#----

#this variable holds the path to the trained disease classifier model
MODEL_PATH = Path("backend/disease_model/models/disease_model.keras")

#this holds the path to the training data and it is only used top gain
#the most consistent class labels
DATA_DIR = Path("backend/disease_model/data/processed/train")  # used only to get class names

#holds the image size expected by the model 
IMG_SIZE = (224, 224)

#during training,class label mappings are created
#and put into a json file which is stored here.
CLASS_NAMES_PATH = Path("backend/disease_model/models/class_names.json")


def load_class_names():
    """
    this function evidently is in charge of loading class names
    and is saved during the training process

    as result this makes sure that the predicted class index that is returned
    by the model maps correctly to its own original disease label
    """
    if not CLASS_NAMES_PATH.exists():
        raise RuntimeError(
            f"{CLASS_NAMES_PATH} not found. Run predict.py once to generate it."
        )

    with open(CLASS_NAMES_PATH, "r") as f:
        return json.load(f)





def preprocess_image(img_path: Path):
    """
   this function is in charge of loading and preparing a single image 
   for inference and analysis to from predictions.

   it once again ,resizes the image to match to the model input size
   it also adds a batch dimension and also convers i o a nump array 

   also,preprocessing is already included inside the trained model ,therefore
   in this instance it wont be applied
    """
    img = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img).astype("float32")  # keep 0..255
    arr = np.expand_dims(arr, axis=0)  # (1, H, W, C)
    return arr
    

def main():

#this is the main function in charge of the main inference for predicting 
#the disease from a single image.
    print("infer_one.py started")


#this is to ensure tha the user provides a path via command line 
    if len(sys.argv) < 2:
        print("Usage: python backend/disease_model/src/infer_one.py <path_to_image>")
        print("Example: python backend/disease_model/src/infer_one.py backend/disease_model/data/processed/test/Tomato_healthy/0a1b.jpg")
        sys.exit(1)

    img_path = Path(sys.argv[1])


    #this checks that the trained model actually exists 
    if not MODEL_PATH.exists():
        print(f"Error: model not found at {MODEL_PATH}. Run predict.py first to create it.")
        sys.exit(1)
    
    #this checks that the provided image exists 
    #and it is genuine for inspection
    if not img_path.exists():
        print(f"Error: image not found at {img_path}")
        sys.exit(1)
    
    #in charge of loading class labels
    class_names = load_class_names()

    #dedicated to loading the trained TensorFlow model
    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    #in charge of preprocessing the image
    #and then it runs the inference
    x = preprocess_image(img_path)
    probs = model.predict(x, verbose=0)[0]
    
    #this is in charge of predicting class and computing confidence
    pred_idx = int(np.argmax(probs))
    pred_class = class_names[pred_idx]
    confidence = float(probs[pred_idx])
    
    #confidence threshold check ..if below 0.5 in confidence then it will just return for user
    #to try again

    image_threshold=0.50
    
    #this computes that if confidence is too low 
    #it will output that the image is not recognised 
    if confidence < image_threshold:
        print("\nImage ",img_path)
        print("\nprediction: IMAGE_NOT_RECOGNISED")
        print("Confidence:", f"{confidence:.4f}")
        return
    
    
    #print statements that output to user the final prediction
    print("\nImage:", img_path)
    print("Prediction:", pred_class)
    print("Confidence:", f"{confidence:.4f}")

    # displays the top 3 predicted classess and their respected probabilities 
    #which is acually usefull for debugging
    top3 = np.argsort(probs)[::-1][:3]
    print("\nTop 3:")
    for i in top3:
        print(f"  {class_names[int(i)]}: {float(probs[int(i)]):.4f}")


#entry point for script 
if __name__ == "__main__":
    main()
