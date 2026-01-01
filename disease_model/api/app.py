import functions_framework
from google.cloud import storage

import os
import sys
import json
import numpy as np
import tensorflow as tf

def preprocess_image(img_path):
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

    storage_client = storage.Client()
    bucket = storage_client.bucket("plant-data-bucket-140")

    # import keras disease model

    #blob = bucket.blob("disease_model.keras")
    
    MODEL_PATH = "gs://plant-data-bucket-140/disease_model.kera"
    
    #blob.download_to_filename(MODEL_PATH)

    IMG_SIZE = (224, 224)


    """
    # import disease class names json

    blob = bucket.blob("class_names.json")
    
    CLASS_NAMES_PATH = "/tmp/class_names.json"
    
    blob.download_to_filename(CLASS_NAMES_PATH)

    with open(CLASS_NAMES_PATH, 'r') as f:
        class_names = json.load(f)



    # import test image

    blob = bucket.blob("tomato_septoria_test.jpg")
    
    img_path = "/tmp/tomato_septoria_test.jpg"
    
    blob.download_to_filename(img_path)



    """
    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print("model loaded successfully")
    
    return 'success'

    """
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
        return 'confidence too low'
    
    

    print("\nImage:", img_path)
    print("Prediction:", pred_class)
    print("Confidence:", f"{confidence:.4f}")

    # Optional: show top 3 predictions (nice for report screenshots)
    top3 = np.argsort(probs)[::-1][:3]
    top3_text = "top 3: "
    
    print("\nTop 3:")
    for i in top3:
        print(f"  {class_names[int(i)]}: {float(probs[int(i)]):.4f}")
        top3_text += f"  {class_names[int(i)]}: {float(probs[int(i)]):.4f}\n"

    return top3_text  
    """  


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

    text = main()



    if request_json and 'name' in request_json:
        name = request_json['name']
    elif request_args and 'name' in request_args:
        name = request_args['name']
    else:
        name = 'World' + text
    return 'Hello {}!'.format(name)
