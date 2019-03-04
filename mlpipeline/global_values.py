EXPERIMENTS_DIR="experiments"
OUTPUT_FILE = EXPERIMENTS_DIR + "/output"

#contains the file that has the list of experiments that were sussessfully executed
#the attached time is the time of the experiment files last modification
#that is, it logs that the given verion of the experiment saved under this name on this time
#has trained sucessfully
HISTORY_FILE = EXPERIMENTS_DIR + "/history"

TRAINING_HISTORY_LOG_FILE = EXPERIMENTS_DIR + "/t_history"

LOG_FILE = EXPERIMENTS_DIR + "/log"
NO_LOG = False
EXECUTED_EXPERIMENTS = {}

#if use_blacklist is true, LISTED_EXPERIMENTS is blacklisted files
#else LISTED_EXPERIMENTS is whitelisted experiments
USE_BLACKLIST=True
LISTED_EXPERIMENTS=[]

TEST_MODE = True

mtime='mtime'
version='version'
train_time="train_time"
vless="vless"
