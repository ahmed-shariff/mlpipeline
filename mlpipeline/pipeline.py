import os
import sys

# why not check for this
if sys.version_info < (3,5):
    sys.stderr.write("ERROR: python version should be greater than or equal 3.5\n")
    sys.exit(1)

import subprocess
import shutil
import configparser
import socket
import argparse

from mlpipeline.utils import log
from mlpipeline.utils import set_logger

from mlpipeline.global_values import MODELS_DIR
from mlpipeline.global_values import TEST_MODE
from mlpipeline.global_values import NO_LOG
from mlpipeline.global_values import USE_BLACKLIST
USE_HISTORY = False

def _main():
    current_model_name = _get_model()
    print(USE_HISTORY, 123)
    while current_model_name is not None:
        #exec subprocess
        args = ["python3", "_pipeline_subprocess.py", current_model_name, MODELS_DIR]
        if NO_LOG:
            args.append("-n")
        if not TEST_MODE:
            args.append("-r")
        if USE_HISTORY:
            args.append("-u")
        output = subprocess.call(args, universal_newlines = True)
        if TEST_MODE:
            break
        current_model_name  = _get_model()

def _get_model(just_return_model=False):
    _config_update()
    for rdir, dirs, files in os.walk(MODELS_DIR):
        for f in files:
            if f.endswith(".py"):
              file_path = os.path.join(rdir,f)
              # TODO: Should remove this check, prolly has no use!
              if USE_BLACKLIST and file_path in LISTED_MODELS:
                  continue
              if not USE_BLACKLIST and file_path not in LISTED_MODELS:
                  continue
              return file_path
    return None

def _config_update():
    if TEST_MODE:
        config_from = "models_test.config"
    else:
        config_from = "models.config"
    config = configparser.ConfigParser(allow_no_value=True)
    config_file = config.read(config_from)
  
    global USE_BLACKLIST
    global LISTED_MODELS
  
    if len(config_file)==0:
        print("\033[1;031mWARNING:\033[0:031mNo 'models.config' file found\033[0m")
    else:
        try:
            config["MLP"]
        except KeyError:
            print("\033[1;031mWARNING:\033[0:031mNo MLP section in 'models.config' file\033[0m")
        USE_BLACKLIST =  config.getboolean("MLP", "use_blacklist", fallback=USE_BLACKLIST)
        try:
            if USE_BLACKLIST:
                LISTED_MODELS = config["BLACKLISTED_MODELS"]
            else:
                LISTED_MODELS = config["WHITELISTED_MODELS"]
            l = []
            for model in LISTED_MODELS:
                l.append(os.path.join(MODELS_DIR, model))
            LISTED_MODELS = l
            print("\033[1;036m{0}\033[0;036m: {1}\033[0m".format(
                ["BLACKLISTED_MODELS" if USE_BLACKLIST else "WHITELISTED_MODELS"][0].replace("_"," "),
                LISTED_MODELS).lower())
        except KeyError:
            print("\033[1;031mWARNING:\033[0:031mNo {0} section in 'cnn.config' file\033[0m".format(
                ["BLACKLISTED_MODELS" if USE_BLACKLIST else "WHITELISTED_MODELS"][0]))


def main(argv = None):
    #if argv is None:
    parser = argparse.ArgumentParser(description="Machine Learning Pipeline")
    parser.add_argument('-r','--run', help='Will set the pipeline to execute the pipline fully, if not set will be executed in test mode', action = 'store_true')
    parser.add_argument('-u','--use-history', help='If set will use the history log to determine if a model script has been executed.', action = 'store_true')
    parser.add_argument('-n','--no_log', help='If set non of the logs will be appended to the log files.', action = 'store_true')
    argv = parser.parse_args()
    config = configparser.ConfigParser(allow_no_value=True)
    config_file = config.read("mlp.config")
    global TEST_MODE
    global NO_LOG
    global MODELS_DIR
    global USE_HISTORY
    
    if len(config_file)==0:
        print("\033[1;031mWARNING:\033[0:031mNo 'mlp.config' file found\033[0m")
    else:
        try:
            config["MLP"]
        except KeyError:
            print("\033[1;031mWARNING:\033[0:031mNo MLP section in 'mlp.config' file\033[0m")
        MODELS_DIR = config.get("MLP", "models_dir", fallback=MODELS_DIR)


    hostName = socket.gethostname()
    MODELS_DIR_OUTPUTS = MODELS_DIR + "/outputs"
    if not os.path.exists(MODELS_DIR_OUTPUTS):
        os.makedirs(MODELS_DIR_OUTPUTS)
    log_file = MODELS_DIR_OUTPUTS + "/log-{0}".format(hostName)
    
    open(log_file, "a").close()
    print(argv)
    if argv is not None:#len(unused_argv)> 0:
        if argv.run:#any("r" in s for s in unused_argv) :
            TEST_MODE = False
        else:
            TEST_MODE = True
      
        if argv.use_history:#any("h" in s for s in unused_argv):
            USE_HISTORY = True
        else:
            USE_HISTORY = False

    _config_update()
    LOGGER = set_logger(test_mode = TEST_MODE, no_log = NO_LOG, log_file = log_file)
    print(USE_HISTORY)
    log("=====================ML-Pipeline session started")
    _main()
    log("=====================ML-Pipeline Session ended")


    
if __name__ == "__main__":  
    #print(parser.parse_args().r)
    # output = subprocess.run(["python3", "--version"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines = True)
    # if int(output.stdout.replace("Python ", "").split(".")[1]) < 5:
    #     print("ERROR: Requires python 3.5 or greater")
    #     sys.exit(1)
    main()
    
