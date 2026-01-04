import functions_framework
from google.cloud import storage

import os
import sys
import json
import numpy as np
import tensorflow as tf
import keras
from tensorflow.python.lib.io import file_io
from tensorflow.keras.models import load_model
import tempfile


def preprocess_image(img_path):
    """
    Load an image and format it as a batch tensor for the model.
    NOTE: Do NOT call preprocess_input here because the saved model
    already contains MobileNetV2 preprocessing inside it.
    """
    IMG_SIZE = (224, 224)
    img = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE)
    arr = tf.keras.utils.img_to_array(img).astype("float32")  # keep 0..255
    arr = np.expand_dims(arr, axis=0)  # (1, H, W, C)
    return arr


@functions_framework.http
def hello_http(request):
    """HTTP Cloud Function.
    Args:
        request (flask.Request): The request object.
        <https://flask.palletsprojects.com/en/1.1.x/api/#incoming-request-data>
    Returns:
        The response text, or any set of values that can be turned into a
        Response object using `make_response`
        <https://flask.palletsprojects.com/en/1.1.x/api/#flask.make_response>.
    """
    request_json = request.get_json(silent=True)
    request_args = request.args

    print("infer_one.py started")

    # set base path to copy files from gcs bucket into

    bucket_mount_path = os.environ.get('MOUNT_PATH', '/mnt/storage')
    IMG_SIZE = (224, 224)

    # import disease class names json


    print("reading json file")
    
    CLASS_NAMES_PATH = os.path.join(bucket_mount_path, "class_names.json")

    with open(CLASS_NAMES_PATH, 'r') as f:
        class_names = json.load(f)

    
    # load image from request
    image_file = request.files['image']

    if image_file.filename == '':
        return "error: no image selected"

    # make a copy of image in ram
    img_path = os.path.join('/tmp/', image_file.filename)

    print(image_file.filename)

    image_file.save(img_path)    
    

    # load model from gcs bucket

    local_model_path = os.path.join(tempfile.gettempdir(), "disease_model.keras") # download the model only once to /tmp directory
    if not os.path.exists(local_model_path):
        print("Downloading model...")
        tf.io.gfile.copy("gs://plant-data-bucket-140/disease_model.keras", local_model_path, overwrite=True)
    
    print("Loading model...")
    model = load_model(local_model_path)
    

    print("model loaded successfully")
    


    # process image into format accepted by model

    x = preprocess_image(img_path)

    # predict plant and disease
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
        return 'confidence too low'
    
    

    print("\nImage:", img_path)
    print("Prediction:", pred_class)
    print("Confidence:", f"{confidence:.4f}")

    # retrieve recommended treatment from json file
    TREATMENT_PATH = os.path.join(bucket_mount_path, "treatment.json")

    with open(TREATMENT_PATH, 'r') as f:
        data = json.load(f)

    confidence = f"{confidence:.4f}"

    confidence = float(confidence) * 100

    confidence = str(confidence) + "%"

    pred_class_formatted = pred_class.replace("_", " ").capitalize()

    pred_class_split = pred_class_formatted.split()

    plant = pred_class_split[0]

    condition = " ".join(pred_class_split[1:])

    treatment = data.get(pred_class, "Treatment not found")

    text = "Plant: " + plant + "\nCondition: " + condition + "\nConfidence: " + confidence + "\n\nTreatment: " + treatment    

    # return findings to app
    return text
