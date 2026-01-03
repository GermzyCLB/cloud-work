import os
import json
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix



#----
#section contain paths and configurations 
#----


#THIS IS THE ROOT DIRECTORY THAT HOLDS THET VAL/TRAIN/TEST FOLDERS FOR PROJECT 
DATA_DIRECTORY = Path("backend/disease_model/data/processed")

#THIS IS THE OUTPUT DIRECTORY THAT CONTAINS THE TRAINED UP MODELS AND 
#ITS RESPECTIVE METADATA
OUTPUT_DIRECTORY = Path("backend/disease_model/models")


#THIS HOLDS THE OUTPUT FILES FOR THE PLANT ONLY CLASSIFIER
PLANT_MODEL_OUTPUT = OUTPUT_DIRECTORY / "plant_model.keras"
PLANT_NAMES_OUTPUT = OUTPUT_DIRECTORY / "plant_class_names.json"

#HOLDS THE IMAGE SIZE HELD AND EXPECTED BY MOBILENETV2
IMG_SIZE = (224, 224)

#BATCH SIZE FOR TRAINING 
BATCH_SIZE = 32

#this holds the number of epochs for the plant model 
EPOCHS = 5  # plant model is easier compared to the disease model so 3–5 is enough



def make_datasets():
    """
   in charge of loading the existing disease dataset and turns the 
   labels into plant-level labels.
    
   
   ptoato = 0
   tomato = 1
    
   so same dataset can be reused train a seperae plant model if needed
    """

    #in charge of loading the disease-level datasets
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIRECTORY / "train",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=True,
        seed=42,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIRECTORY / "val",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=True,
        seed=42,
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIRECTORY / "test",
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
    )


    #in charge of extracting the appropriate disease classes
    #(13 disease classses to be precise)
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



#---
#this function converts the disease labels into its respective plan labels
#---
    def plant_mapping(images, labels):
        plant_labels = tf.gather(mapping, labels)
        return images, plant_labels
    
    #applies label mapping to all the respective datasets
    train_ds = train_ds.map(plant_mapping)
    val_ds = val_ds.map(plant_mapping)
    test_ds = test_ds.map(plant_mapping)
    
    #optimises data pipeline performances
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)
    
    #left with the final plant class labels
    plant_class_names = ["Potato", "Tomato"]
    return train_ds, val_ds, test_ds, plant_class_names



def build_model(num_classes: int):


#----
#this fuction genereally builds a mobileNetV2-based plant classification model 

#uses imageNet pretrained weights 
#adds data augmentation and a custom made classification head
#freezes the base cnn in the first instance
#----
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    #carries out data augmentation that imporves the generalisation
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



#-----
#This function is involved in evaluating the trained up models using sci-kit learn metrics,
#this includes the precision,recall,f1-score and also the 
#confusion matrix 
#-----
def sklearn_evaluation(model, test_ds, class_names):
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

    #runs the main training pipleine that the plant identification model uses.
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


    #iterative function in charge of ensure  that the processed data exists
    if not (DATA_DIRECTORY / "train").exists():
        print("Can't find processed/train. Run train.py first.")
        return
    
    
    #loads the datasets into the function
    train_ds, val_ds, test_ds, plant_names = make_datasets()
    print("Plant classes:", plant_names)


    #saves the plant class names for inference consistency
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    with open(PLANT_NAMES_OUTPUT, "w") as f:
        json.dump(plant_names, f)

    print(f"Saved plant class names to: {PLANT_NAMES_OUTPUT}")
    
    #responsible for building and training a model 
    model, base_model = build_model(num_classes=len(plant_names))
    model.summary()

    # Step 1: responsible for training the classification head only 
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

    # Step 2:fine tunes the last layers of the base cnn
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    #then compiles model 
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(train_ds, validation_data=val_ds, epochs=2)
    
    #saves the trained plant model 
    model.save(PLANT_MODEL_OUTPUT)
    print(f"\nSaved plant model to: {PLANT_MODEL_OUTPUT}")
    
    #carries out final evaluation
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"\nPlant Test accuracy: {test_acc:.4f} | Test loss: {test_loss:.4f}")

    sklearn_evaluation(model, test_ds, plant_names)

if __name__ == "__main__":
    main()
