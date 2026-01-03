import os
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter


#---
#below are my pah configurations respective to my folder on my computer
#---

#this has the path to he processed raw plant village dataset in which
#it contains many the folders rgarding training/testing/validation folders
DATA_DIR = Path("backend/disease_model/data/processed")

##holds the output for the path that holds the trained classification model
MODEL_OUT = Path("backend/disease_model/models/disease_model.keras")





##-----
#   THE TRAINING HYPERPARAMATERS
##-----


#THis is the initial number of epochs that will be used in the first run through of training
#this will be increased laterto 8 for fine tuning purposes to imporve the accuracy

EPOCHS = 5  # start with 5 for speed purposes



#this is the target image size that will be expected by MobileNetv2
IMAGE_SIZE = (224, 224)



#has the number of images that will be processed  by each batch
BATCH_SIZE = 32


##THIS FUNCTION WILL LOAD THE TRAINING,TEST AND VALIDATION FROM DISK

#FURTHERMORE THE DATASET WILL BE LOADED USING DIRECTORY STRUCTURE,
#AND EACH FOLDER NAME WILL REPRESENT ITS OWN CLASS LABEL
def MADE_DATSET():


    #in charge of loading the dataset 
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / "train",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=True,
        seed=42,
    )

    #in charge of loading the validation dataset 
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / "val",
       
        image_size=IMAGE_SIZE,

        batch_size=BATCH_SIZE,

        label_mode="int",

        shuffle=True,
        seed=42,
    )


    #in charge of loading the test dataset and there will be no shuffling 
    #as the label order needs to be preserved 
    test_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR / "test",

        image_size=IMAGE_SIZE,

        batch_size=BATCH_SIZE,

        label_mode="int",
        shuffle=False,
    )
    
    #his is the class that will be directly inferred from the folder names
    class_names = train_ds.class_names

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)


    return train_ds, val_ds, test_ds, class_names

#---
#FUNCTION THAT INVOLVES THE CREATION OF MODELS
#---

def MODEL_BUILT(num_classes: int):
    #--
    #this builds and compiles the disease classification model using 
    #transfer learning with Mobilenetv2

    #1)it parses in the number of classes that are outputed
    #2)it then compiles the keras model 
    #3)and the base model is mobilenetv2 which is used often for fine-tuning 

    # This is the original pretrained Base CNN (Imagenet weights)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMAGE_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )

    #freeze base model weights during initial training 
    base_model.trainable = False  # freeze for the first training stage

    # these are applied during training only and it is data augmentation
    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.10),
    ])
    
    #these are for model input
    inputs = tf.keras.Input(shape=IMAGE_SIZE + (3,))
    x = augment(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)


    #these contribute to the extraction of features which uses the pretrained
    #model 
    x = base_model(x, training=False)
    
    #by carrying out global pooling it reduces spatial dimensions 
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    #to ensure no overfiting has occured we use dropout to do so
    x = tf.keras.layers.Dropout(0.2)(x)

    #this is the output layer for multi-class classification
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    #this builds the model
    model = tf.keras.Model(inputs, outputs)

    #
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base_model



##----
#FUNCTION THAT IS IN CHARGE OF EVALUATING FUNCTIONS (USE OF SKLEARN)
def evaluate_sklearn(model, test_ds, class_names):
    y_true, y_pred = [], []


    ##FROM SKLEARN TO MEASURE METRICS ,USES THINGS LIKE:
    #PRECISION,RECALL,F1-SCORE AND CONFUSION MATRIX 

#COLLECTS PREDICTIONS AND TRUE LABELS(IMPORTANT) 
    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        preds = np.argmax(probs, axis=1)
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(preds.tolist())

    print("\n=== scikit-learn classification report ===")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    print("=== confusion matrix ===")
    print(confusion_matrix(y_true, y_pred))


## This is the main function
# this carries out the dataset loading,model training,fine-tuning,
# evaluation and model saving all in one         
def main():

    #reduces the tensorflow logging noise
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    if not (DATA_DIR / "train").exists():
        print(" Can't find processed/train. Run train.py first.")
        return
    
    #loads the dataset throught made_dataset 
    train_ds, val_ds, test_ds, class_names = MADE_DATSET()
    print(f" Classes ({len(class_names)}): {class_names}")

    # ----------
    # this saves class labels for later mapping    , class label mappings for later inference
    # ----------
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)

    with open(MODEL_OUT.parent / "class_names.json", "w") as f:
        json.dump(class_names, f)

    print("Saved class names to models/class_names.json")
    
    #in charge of building models
    model, base_model = MODEL_BUILT(num_classes=len(class_names))
    model.summary()
    

    #----
    # in charge of class weighting (handles imbalanced dataset)
    #----
    # Build class weights from the training dataset to reduce class imbalance
    label_counts = Counter()
    for _, labels in train_ds.unbatch():
        label_counts[int(labels.numpy())] += 1

    max_count = max(label_counts.values())
    class_weight = {cls: max_count / count for cls, count in label_counts.items()}
    
    print("Class weights:", class_weight)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        class_weight=class_weight,
    )
    
    # -----------
    # a small fine tuning phase
    # -----------
    base_model.trainable = True

    # this preserves low-level features by freezing the earlier layers
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    #next ,this then recompiles with the lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        #below holds the training dataset containing images and labels 
        #and it lears nicely by altering its internal weights
        train_ds,


        #this holds is own validation set that in turn is used to evalute
        #the model after every epoch(5+3)
        validation_data=val_ds,
 
        #the number of times it will make a complete pass over the training data
        #and in this phase a small number of epochs is used as this is very 
        #much a fine tuning stage
        epochs=3,
    
        #helps to address class imbalance in dataset as well as improves
        #performance on less prevelant plant/disease areas/categories
        class_weight=class_weight,
    )
    

    #---
    #THE FINAL EVAL + SAVED MODEL SECTION
    #----


    
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_OUT)
    print(f"\n model is saved to: {MODEL_OUT}")

    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"\n Test accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")

    evaluate_sklearn(model, test_ds, class_names)

if __name__ == "__main__":
    main()
