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
import logging

from mlpipeline.utils import log
from mlpipeline.utils import set_logger

from mlpipeline.global_values import MODELS_DIR
from mlpipeline.global_values import TEST_MODE
from mlpipeline.global_values import NO_LOG
from mlpipeline.global_values import USE_BLACKLIST
USE_HISTORY = False

def _main():
    completeed_models = []
    current_model_name = _get_model()
    while current_model_name is not None:
        #exec subprocess
        args = ["_mlpipeline_subprocess", current_model_name, MODELS_DIR]
        if NO_LOG:
            args.append("-n")
        if not TEST_MODE:
            args.append("-r")
        if USE_HISTORY:
            args.append("-u")
        output = subprocess.call(args, universal_newlines = True)
        if output == 3 or output == 1:
            completeed_models.append(current_model_name)
        if TEST_MODE:
            break
        current_model_name  = _get_model(completeed_models)

def _get_model(completeed_models = []):
    _config_update()
    for rdir, dirs, files in os.walk(MODELS_DIR):
        for f in files:
            if f.endswith(".py"):
                file_path = os.path.join(rdir,f)
                if completeed_models is not None and file_path in completeed_models:
                    continue
                # TODO: Should remove this check, prolly has no use!
                if USE_BLACKLIST and file_path in LISTED_MODELS:
                    continue
                if not USE_BLACKLIST and file_path not in LISTED_MODELS:
                    continue
                skip_model_for_now = False

                # Ensure the files loaded are in the order they are
                # specified in the config file
                for listed_model_file in LISTED_MODELS:
                    if listed_model_file != file_path:
                        if listed_model_file not in completeed_models:
                            skip_model_for_now = True
                            break
                    else:
                        break
                if skip_model_for_now:
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
        log("\033[1;031mWARNING:\033[0:031mNo 'models.config' file found\033[0m", log_to_file = True)
    else:
        try:
            config["MLP"]
        except KeyError:
            log("\033[1;031mWARNING:\033[0:031mNo MLP section in 'models.config' file\033[0m", log_to_file = True, level = logging.WARNING)
        USE_BLACKLIST =  config.getboolean("MLP", "use_blacklist", fallback=USE_BLACKLIST)
        try:
            if USE_BLACKLIST:
                LISTED_MODELS = config["BLACKLISTED_MODELS"]
            else:
                LISTED_MODELS = config["WHITELISTED_MODELS"]
            l = []
            for model in LISTED_MODELS:
                l.append(os.path.join(MODELS_DIR, model))

            for model in l:
                if not os.path.exists(model):
                    l.remove(model)
                    log("Script missing: {}".format(model), level = logging.WARNING)
            LISTED_MODELS = l
            log("\033[1;036m{0}\033[0;036m: {1}\033[0m".format(
                ["BLACKLISTED_MODELS" if USE_BLACKLIST else "WHITELISTED_MODELS"][0].replace("_"," "),
                LISTED_MODELS).lower(), log_to_file = True)
        except KeyError:
            log("\033[1;031mWARNING:\033[0:031mNo {0} section in 'cnn.config' file\033[0m".format(
                ["BLACKLISTED_MODELS" if USE_BLACKLIST else "WHITELISTED_MODELS"][0]), log_to_file = True, level = logging.ERROR)


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
    try:
      open(log_file, "a").close()
    except FileNotFoundError:
      if os.path.isdir(MODELS_DIR_OUTPUTS):
        os.makedirs(MODELS_DIR_OUTPUTS)
        open(log_file, "a").close()
      else:
        raise

    if argv is not None:#len(unused_argv)> 0:
        if argv.run:#any("r" in s for s in unused_argv) :
            TEST_MODE = False
        else:
            TEST_MODE = True
      
        if argv.use_history:#any("h" in s for s in unused_argv):
            USE_HISTORY = True
        else:
            USE_HISTORY = False

    LOGGER = set_logger(test_mode = TEST_MODE, no_log = NO_LOG, log_file = log_file)
    log("=====================ML-Pipeline session started")
    _main()
    log("=====================ML-Pipeline Session ended")


    
if __name__ == "__main__":  
    main()
    
