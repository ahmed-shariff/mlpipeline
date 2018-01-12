MODELS_DIR="models"
DATA_FILE_LOCATION = "/media/Files/Research/FoodClassification/Datasets/Food-101/food-101/images/"
TEST_FILE_LOCATION = None
OUTPUT_FILE = MODELS_DIR + "/output"

#tf.python.tools.inspect_checkpoint.print_tensors_in_checkpoint_file

#contains the file that has the list of models that were sussessfully executed
#the attached time is the time of the model files last modification
#that is, it logs that the given verion of the model saved under this name on this time
#has trained sucessfully
HISTORY_FILE = MODELS_DIR + "/history"

TRAINING_HISTORY_LOG_FILE = MODELS_DIR + "/t_history"

LOG_FILE = MODELS_DIR + "/log"
NO_LOG = False
DATA_CODE_MAPPING = {}
EXECUTED_MODELS = {}

#if use_blacklist is true, LISTED_MODELS is blacklisted files
#else LISTED_MODELS is whitelisted models
USE_BLACKLIST=True
LISTED_MODELS=[]
EPOC_COUNT = 20

TEST_MODE = True

mtime='mtime'
version='version'
train_time="train_time"
vless="vless"
BATCH_SIZE = 15
USE_ALL_CLASSES = False
CLASSES_COUNT = 25
CLASSES_OFFSET = ["pizza", "fried_rice", "hamburger", "ice_cream", "red_velvet_cake", "samosa", "spring_rolls", "breakfast_burrito", "chocolate_cake", "club_sandwich", "chicken_wings", "cup_cakes", "donuts", "french_fries"]
#CLASSES_OFFSET = ["french_toast", "pizza", "fried_rice", "grilled_cheese_sandwich", "hamburger", "ice_cream", "frozen_yogurt", "garlic_bread", "macaroni_and_cheese", "omelette", "red_velvet_cake", "samosa", "spring_rolls", "breakfast_burrito", "chocolate_cake", "club_sandwich", "chicken_wings", "cup_cakes", "donuts", "french_fries"]
ALLOW_DELETE_MODEL_DIR = True
RESTART_GLOBAL_STEP = False
MODEL_DIR_SUFFIX=""
VERSION_HOOKS=[]
