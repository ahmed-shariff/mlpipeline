# TODO: move this file to a config file?
# This file came about as a result of historical reason,
# no perticular reason to have this as a py file at this point

# The driectory in which the experiments scripts are placed
EXPERIMENTS_DIR = "experiments"
OUTPUT_DIR = "{}/outputs"
OUTPUT_FILE = "{}/output"

# contains the file that has the list of experiments that were sussessfully executed
# the attached time is the time of the experiment files last modification
# that is, it logs that the given verion of the experiment saved under this name on this time
# has trained sucessfully
HISTORY_FILE = "{}/history"

TRAINING_HISTORY_LOG_FILE = "{}/t_history"

LOG_FILE = "{}/log"
NO_LOG = False
EXECUTED_EXPERIMENTS = {}

# if use_blacklist is true, LISTED_EXPERIMENTS is blacklisted files
# else LISTED_EXPERIMENTS is whitelisted experiments
USE_BLACKLIST = True
LISTED_EXPERIMENTS = []

EXPERIMENT_MODE = 'Test'
