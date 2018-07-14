MODELS_DIR="models"
OUTPUT_FILE = MODELS_DIR + "/output"

#contains the file that has the list of models that were sussessfully executed
#the attached time is the time of the model files last modification
#that is, it logs that the given verion of the model saved under this name on this time
#has trained sucessfully
HISTORY_FILE = MODELS_DIR + "/history"

TRAINING_HISTORY_LOG_FILE = MODELS_DIR + "/t_history"

LOG_FILE = MODELS_DIR + "/log"
NO_LOG = False
EXECUTED_MODELS = {}

#if use_blacklist is true, LISTED_MODELS is blacklisted files
#else LISTED_MODELS is whitelisted models
USE_BLACKLIST=True
LISTED_MODELS=[]

TEST_MODE = True

mtime='mtime'
version='version'
train_time="train_time"
vless="vless"
