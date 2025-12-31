import random
import shutil
from pathlib import Path


##----
#all paths
##----

#  The respected path of where my raw PlantVillage class folders live currently.
RAW_DATA_DIR = Path("backend/disease_model/data/raw")

# Where we will create train/val/test this is the path of which  our training,validation and test will be created ansd stored
PROCESSED_DATA_DIR = Path("backend/disease_model/data/processed")

##----
#Data split ratios that will be used for image processing 
#-----

TRAIN_SPLIT_RATIO = 0.70
VALIDATION_SPLIT_RATIO = 0.20
TEST_SPLIT_RATIO = 0.10

# This outlines the image file formas that will be allowed in this process
Valid_image_types_allowed = {".jpg", ".jpeg", ".png"}

##----
#this section displays all my helper functions that are being used 
##---


##this function will delete a direcory if i already exists and will then recreate it.This ensures old dataset splits do not intterfere with new ones.
def recreation_directory(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


#function that checks if the file is valid based on its exstension
def valid_image_checker(path: Path) -> bool:
    
    return path.suffix.lower() in Valid_image_types_allowed


#~simple function that is in charge of splitting the images into train,validation and test individual sets 
#each class directory is shuffled and then divided into training,
#validation and test subsets using an index based slicing
#done through the pre-defined ratios .

def split_class_directory(class_directory: Path):
    """
    Splits images from a single class into train, validation, and test sets.
    """
    
    
    #in charge of collecting all valid images from the class directory
    #important for this process

    all_images = [
        file for file in class_directory.iterdir()
        if file.is_file() and valid_image_checker(file)
    ]
    
    #if the file has no valid all_images ,skips processing 

    if not all_images:
        print(f"No images found in {class_directory.name}")
        return


    #This will randomly split the images so that the split is unbiased
    random.shuffle(all_images)


    #calculate how many images should go into each subset of train,validation
    image_total  = len(all_images)
    train_end = int(image_total  * TRAIN_SPLIT_RATIO)
    val_end = train_end + int(image_total* VALIDATION_SPLIT_RATIO)
     
    #in charge of splitting image list into train,test and validation sets..
    #IMPORTANT PROCESS
    splits = {
        "train": all_images[:train_end],
        "val": all_images[train_end:val_end],
        "test": all_images[val_end:]
    }
    
    #for loop that goes through the split names and files and creates the output directories
    #and puts them in accordingly 
    for split_name, split_files in splits.items():
        the_output_path = PROCESSED_DATA_DIR / split_name / class_directory.name
        the_output_path.mkdir(parents=True, exist_ok=True)

        for image in split_files:
            shutil.copy2(image, the_output_path / image.name)
    
    #displays a summary of how many images were assigned to each subset
    print(
        f"{class_directory.name}: "
        f"{len(splits['train'])} train, "
        f"{len(splits['test'])} test"
        f"{len(splits['val'])} val, "
        
    )

def main():

    """"
     This is he main funcion in his class in charge of perfoming dataset splitting for
     all class folders in which are conained within the raw dataset directory 

    """

    #These print statememts are purely for debug purposes 
    #crucial for dealiing with path issues
    print("RAW_DATA_DIR =", RAW_DATA_DIR.resolve())
    print("PROCESSED_DATA_DIR =", PROCESSED_DATA_DIR.resolve())
    
    #if statement makes sure that the raw dataset directory exists
    if not RAW_DATA_DIR.exists():
        print(" RAW_DATA_DIR does not exist. Check your folder path.")
        return
  

    #in charge of clearly identifying class directories
    #within the raw dataset
    class_folders = [directory for directory in RAW_DATA_DIR.iterdir() if directory.is_dir()]
    print(f"Found {len(class_folders)} class folders in raw/")


    # this random seed is used for reproducibility
    random.seed(42)



    #in charge of resetting the processed dataset directory 
    recreation_directory(PROCESSED_DATA_DIR)
    
    #splits each class directoy independently
    for class_directory in class_folders:
        split_class_directory(class_directory)

    print("\n Done. Output created at:", PROCESSED_DATA_DIR)





    ##entry point for script 

if __name__ == "__main__":
    main()
