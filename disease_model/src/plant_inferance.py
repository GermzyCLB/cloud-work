import os

#reduces the overall tensorflow informational logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # hides INFO logs (keeps errors)

#
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # optional: stops oneDNN message
import sys
import json
import numpy as np
import warnings
from pathlib import Path

#used to reduce the future warnings from tensor flow or from numpy 
warnings.filterwarnings("ignore", category=FutureWarning)

import tensorflow as tf





#------
#SECTION OF CONSTANTS AND CONFIGURATION PATHS 
#------

#this is the path to the produced trained plant classifier model
#in a 'keras' file format 
PLANT_MODEL_PATH = Path("backend/disease_model/models/plant_model.keras")
 
#this is a json file  that contains the PLANT CLASS LABELS
PLANT_NAMES_PATH = Path("backend/disease_model/models/plant_class_names.json")

#a usefull confidence threshold for rejecting low-confidence predictions
THRESHOLD = 0.50  # optional for "not recognised"

#this is the target image size expected by the model 
IMG_SIZE = (224, 224)




def image_preprocessor(img_path: Path):

  ###------
  # THIS FUNCTION LOADS AND PREPROCESSES A SINGLE IMAGE FOR INFERENC AND ANALYSIS
  #TO SEE WHETHER PREDICTIONS ARE CORRECT 

  #its adds a batch dimension which is required by the keras models
  #and it convers it to a numpy array 
  #then it resizes the image to the models input size

  #the image values used are kept in the range of 0-255 
  #as the pre-processing is already embedded inside the trained model
  #----

    img = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img).astype("float32")  # keep 0..255
    arr = np.expand_dims(arr, axis=0)
    return arr

def main():

    #this is the entry point for a single-image plant inference.
    #also used in testing phase to see if model correctly predicts 
    if len(sys.argv) < 2:
        print("Usage: python backend/disease_model/src/infer_plant.py <path_to_image>")
        sys.exit(1)

    img_path = Path(sys.argv[1])

    #checks if the image path is valid 
    if not img_path.exists():
        print(f"Error: image not found at {img_path}")
        sys.exit(1)
    
    #checks and validates the models existence 
    if not PLANT_MODEL_PATH.exists():
        print(f"Error: plant model not found at {PLANT_MODEL_PATH}")
        sys.exit(1)
    
    #this loads plant class labels to preserve correct index-to-name mapping
    with open(PLANT_NAMES_PATH, "r") as f:
        plant_names = json.load(f)
    
    #this is incharge of loading in the trained tensor flow model 
    model = tf.keras.models.load_model(PLANT_MODEL_PATH)
    
    #this altogether perfoms prediction and carries out the preprocess
    #of image.
    x = image_preprocessor(img_path)
    probs = model.predict(x, verbose=0)[0]
    
    #in charge of extracting the predicted class 
    #and the confidence score that is achieved
    pred_idx = int(np.argmax(probs))
    pred_plant = plant_names[pred_idx]
    conf = float(probs[pred_idx])
    

    #outputs and prints the image path
    print("\nImage:", img_path)
    if conf < THRESHOLD:
        print("Plant: PLANT_NOT_RECOGNISED")
        print("Confidence:", f"{conf:.4f}")
        return
    
    #these print statements print out the final prediction.
    print("Plant:", pred_plant)
    print("Confidence:", f"{conf:.4f}")
    
    #shows top-2 class probabilities for transparency/debugging 
    top2 = np.argsort(probs)[::-1][:2]
    print("\nTop 2:")
    for i in top2:
        print(f"  {plant_names[int(i)]}: {float(probs[int(i)]):.4f}")

#entry point for script
if __name__ == "__main__":
    main()
