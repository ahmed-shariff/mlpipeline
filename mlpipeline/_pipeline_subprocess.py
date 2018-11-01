import os
import sys
import importlib.util
import shutil
import socket
import argparse
import logging
from datetime import datetime

from mlpipeline.utils import ExecutionModeKeys
from mlpipeline.utils import version_parameters
from mlpipeline.utils import VersionLog
from mlpipeline.utils import console_colors
from mlpipeline.utils import log
from mlpipeline.utils import set_logger
from mlpipeline.utils import add_script_dir_to_PATH

from mlpipeline.global_values import MODELS_DIR
from mlpipeline.global_values import NO_LOG
from mlpipeline.global_values import EXECUTED_MODELS
from mlpipeline.global_values import USE_BLACKLIST
from mlpipeline.global_values import TEST_MODE

from mlpipeline.global_values import mtime
from mlpipeline.global_values import version
from mlpipeline.global_values import train_time
from mlpipeline.global_values import vless

def _main(file_path):
    current_model, version_name, clean_model_dir = _get_model(file_path)
    if current_model is None:
        sys.exit(3)
    _add_to_and_return_result_string("Model: {0}".format(current_model.name), True)
    _add_to_and_return_result_string("Version: {0}".format(version_name))
    log("Model loaded: {0}".format(current_model.name))
    if version_name is None:
        log("No Version Specifications",
	    logging.WARNING,
	    modifier_1 = console_colors.RED_FG,
	    modifier_2 = console_colors.BOLD)
    else:
        log("version loaded: {0}".format(version_name),
	  modifier_1 = console_colors.GREEN_FG,
	  modifier_2 = console_colors.BOLD)
      
    #print("\033[1;32mMode: {0}\033[0m".format(modestring))
    if TEST_MODE:
        log("Mode: {}TESTING".format(console_colors.YELLOW_FG),
	    modifier_1 = console_colors.BOLD,
	    modifier_2 = console_colors.GREEN_FG)
    else:
        log("Mode: {}RUNNING MODEL TRAINING".format(console_colors.RED_FG),
	    modifier_1 = console_colors.BOLD,
	    modifier_2 = console_colors.GREEN_FG)
      
    version_spec = current_model.versions.getVersion(version_name)
      
    batch_size = version_spec[version_parameters.BATCH_SIZE]
    model_dir_suffix = version_spec[version_parameters.MODEL_DIR_SUFFIX]
    dataloader = version_spec[version_parameters.DATALOADER]

    log("Version_spec: {}".format(version_spec))
    
    if TEST_MODE:
        record_training = False
        model_dir = "{0}/outputs/model_ckpts/temp".format(MODELS_DIR.rstrip("/"))
        shutil.rmtree(model_dir, ignore_errors=True)
    else:
        record_training = True
        model_dir="{0}/outputs/model_ckpts/{1}-{2}".format(MODELS_DIR.rstrip("/"),
							 current_model.name.split(".")[-2],
							 model_dir_suffix)
    eval_complete=False
    #LOGGER.setLevel(logging.INFO)

    train_results = ""
    eval_results = ""
    
    try:
        if clean_model_dir and current_model.allow_delete_model_dir:
            current_model.clean_model_dir(model_dir)
            log("Cleaned model dir", modifier_1 = console_colors.RED_FG)
        current_model.pre_execution_hook(version_spec, model_dir)
        if TEST_MODE:
            test__eval_steps = 1
            train_eval_steps = 1
        else:
            test__eval_steps = dataloader.get_test_sample_count()
            train_eval_steps = dataloader.get_train_sample_count()

        _save_training_time(current_model, version_name)
        classification_steps = _get_training_steps(ExecutionModeKeys.TRAIN, current_model, clean_model_dir)
        log("Steps: {0}".format(classification_steps))
        if classification_steps > 0:
            train_output = current_model.train_model(dataloader.get_train_input(), classification_steps)
            log("Model traning output: {0}".format(train_output))
            log("Model trained")
        else:
            log("No training. Loaded pretrained model")

        try:
            log("Training evaluation started: {0} steps".format(train_eval_steps))
            train_results = current_model.evaluate_model(dataloader.get_train_input(mode = ExecutionModeKeys.TEST),
                                                         steps = train_eval_steps)
        except Exception as e:
            train_results = "Training evaluation failed: {0}".format(str(e))
            log(train_results, logging.ERROR)
            if TEST_MODE:
                raise
            
        try:
            log("Testing evaluation started: {0} steps".format(test__eval_steps))
            eval_results = current_model.evaluate_model(dataloader.get_test_input(),
						      steps = test__eval_steps)
        except Exception as e:
            eval_results = "Test evaluation failed: {0}".format(str(e))
            log(eval_results, logging.ERROR)
            if TEST_MODE:
                raise
	
        log("Model evaluation complete")
        log("Eval on train set: {0}".format(train_results))
        log("Eval on test set:  {0}".format(eval_results))
        _add_to_and_return_result_string("Eval on train set: {0}".format(train_results))
        _add_to_and_return_result_string("Eval on test  set: {0}".format(eval_results))
        _add_to_and_return_result_string("-------------------------------------------")
        _add_to_and_return_result_string("EXECUTION SUMMERY:")
        _add_to_and_return_result_string("Number of epocs: {0}".format(version_spec[version_parameters.EPOC_COUNT]))
        _add_to_and_return_result_string("Parameters for this version: {0}".format(version_spec))
        _add_to_and_return_result_string("-------------------------------------------")
        _add_to_and_return_result_string("MODEL SUMMERY:")
        _add_to_and_return_result_string(current_model.summery)
        _add_to_and_return_result_string("-------------------------------------------")
        _add_to_and_return_result_string("DATALOADER	 SUMMERY:")
        _add_to_and_return_result_string(dataloader.summery)
        if record_training and not NO_LOG:
            _save_results_to_file(_add_to_and_return_result_string(), current_model)

    except Exception as e:
        if TEST_MODE is True:
            raise
        else:
            log("Exception: {0}".format(str(e)), logging.ERROR)
            sys.exit(1)

    
def _get_training_steps(mode, model, clean_model_dir):
    if TEST_MODE:
        return 1
    else:
        current_version = model.get_current_version()
        complete_steps =  current_version[version_parameters.EPOC_COUNT] * \
            current_version[version_parameters.DATALOADER].get_train_sample_count() / \
            current_version[version_parameters.BATCH_SIZE]
        global_step = model.get_trained_step_count()
        if global_step is None or model.reset_steps:
            return complete_steps
        
                #TODO: why did i add the reset_step here?
        elif clean_model_dir and not model.allow_delete_model_dir and model.reset_steps:
            return complete_steps
        else:
            if complete_steps > global_step:
                return complete_steps - global_step
            else:
                return 0
      
def _get_model(file_path, just_return_model=False):
    # Import and load the model
    spec = importlib.util.spec_from_file_location(file_path.split("/")[-1],file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    clean_model_dir = False
    model = None
    try:
        model = module.MODEL
        model.name = file_path
    except:
        log("{0} is not a model script. It does not contain a `MODEL` global variable".format(file_path))
        return None, None, False

    # TODO: why did i add this in the first place??
    # if just_return_model:
    #	  print("\033[1;33mJust returning module\033[1;0m")
    #	  return module

    # Figure our which version should be executed next
    returning_version = None
    try:
        versions = model.versions
    except:
        versions = None
    log("{0}{1}Processing model: {2}{3}".format(console_colors.BOLD,
						console_colors.BLUE_FG,
						model.name,
						console_colors.RESET))

    ## Get the training history. i.e. the time stamps of each training launched
    with open(TRAINING_HISTORY_LOG_FILE, "r") as t_hist_file:
        t_history = [line.rstrip("\n") for line in t_hist_file]
        all_history = [t_entry.split("::") for t_entry in t_history]
        module_history = [(v,float(t)) for n,v,t in all_history if n == model.name]

    if file_path not in EXECUTED_MODELS:
        EXECUTED_MODELS[model.name] = {}
        EXECUTED_MODELS[model.name][train_time]=0
        EXECUTED_MODELS[model.name][version]=VersionLog()

    EXECUTED_MODELS[model.name][mtime] = os.path.getmtime(file_path)

    ## Determine if training should be started from scratch or should resume training
    ## Here, the modified time is used as an indicator.
    ## If there is a entry in the training shitory that is greater than the the modified time,
    ## that implies the the model was modeified later, hence the training should restart from scratch.
    reset_model_dir = True
    modified_time = os.path.getmtime(file_path)
    for v,t in module_history:
        if t > modified_time:
            reset_model_dir = False
    if reset_model_dir:
        clean_model_dir = True
        EXECUTED_MODELS[model.name][version].clean()
    else:
        # If a training had started and not completed, resume the training of that version
        versions__ = [v_ for v_ in versions.versions]
        for v,t in module_history:
            if t > modified_time:
                if EXECUTED_MODELS[model.name][version].executed(v) is not VersionLog.EXECUTED and v in versions__:
                    modified_time = t
                    returning_version = v
    ## If there are no training sessions to be resumed, decide which version to execute next based on the ORDER set in the version
    if returning_version is None:
        #TODO: check if this line works:
        for v,k in sorted(versions.versions.items(), key=lambda x:x[1][version_parameters.ORDER]):
            if EXECUTED_MODELS[model.name][version].executed(v) is not VersionLog.EXECUTED:
                returning_version = v
                clean_model_dir = True
    log("Executed versions: {0}".format(EXECUTED_MODELS[model.name][version].executed_versions),
        log_to_file=False)
    if returning_version is None:
        return None, None, False
    return model, returning_version, clean_model_dir
    

def _add_to_and_return_result_string(message=None, reset_result_string = False, indent = True):
    global result_string
    if message is not None:
        if indent:
            message = "\t\t" + message
        if reset_result_string:
            result_string = message + "\n"
        else:
            result_string += message + "\n"
    return result_string

def _save_training_time(model, version_):
    if TEST_MODE:
        return
    name = model.name
    with open(TRAINING_HISTORY_LOG_FILE, "a") as log_file:
        time = datetime.now().timestamp()
        EXECUTED_MODELS[name][version].addExecutingVersion(version_, time)
        log("Executing version: {0}".format(EXECUTED_MODELS[model.name][version].executing_version),
            log_to_file=False)
        log_file.write("{0}::{1}::{2}\n".format(name,
                                                EXECUTED_MODELS[name][version].executing_version,
                                                time))

    
def _save_results_to_file(resultString, model):#model, result, train_result, dataloader, training_done, model_dir):
    modified_dt = datetime.isoformat(datetime.fromtimestamp(EXECUTED_MODELS[model.name][mtime]))
    result_dt = datetime.now().isoformat()
  
    #_add_to_and_return_result_string("\n[{0}]:ml-pipline: output: \n".format(result_dt))
    with open(OUTPUT_FILE, 'a', encoding = "utf-8") as outfile:
        outfile.write("\n[{0}]:ml-pipline: output: \n".format(result_dt))
        outfile.write(resultString)
    with open(HISTORY_FILE, 'a', encoding = "utf-8") as hist_file:
        hist_file.write("{0}::{1}::{2}\n".format(model.name,
                                                 EXECUTED_MODELS[model.name][mtime],
                                                 EXECUTED_MODELS[model.name][version].executing_version))
    
    EXECUTED_MODELS[model.name][version].moveExecutingToExecuted()


def main():
    parser = argparse.ArgumentParser(description="Machine Learning Pipeline")
    parser.add_argument("file_path", help='The file path of the model to be executed')
    parser.add_argument("models_dir", help='The directory in which the models reside, also where the results are to stored')
    parser.add_argument('-r','--run', help='Will set the pipeline to execute the pipline fully, if not set will be executed in test mode', action = 'store_true')
    parser.add_argument('-u','--use-history', help='If set will use the history log to determine if a model script has been executed.', action = 'store_true')
    parser.add_argument('-n','--no_log', help='If set non of the logs will be appended to the log files.', action = 'store_true')
    argv = parser.parse_args()
    
    global TEST_MODE
    global NO_LOG
    global LOGGER
    global MODELS_DIR
    global HISTORY_FILE
    global LOG_FILE
    global OUTPUT_FILE
    global TRAINING_HISTORY_LOG_FILE

    file_path = argv.file_path
    MODELS_DIR = argv.models_dir
    
    hostName = socket.gethostname()
    MODELS_DIR_OUTPUTS = MODELS_DIR + "/outputs"
    OUTPUT_FILE = MODELS_DIR_OUTPUTS + "/output-{0}".format(hostName)
    HISTORY_FILE = MODELS_DIR_OUTPUTS + "/history-{0}".format(hostName)
    TRAINING_HISTORY_LOG_FILE = MODELS_DIR_OUTPUTS + "/t_history-{0}".format(hostName)
    LOG_FILE = MODELS_DIR_OUTPUTS + "/log-{0}".format(hostName)
    open(OUTPUT_FILE, "a").close()
    open(HISTORY_FILE, "a").close()
    open(TRAINING_HISTORY_LOG_FILE, "a").close()
    open(LOG_FILE, "a").close()

    if argv.run:#any("r" in s for s in unused_argv) :
        TEST_MODE = False
    else:
        TEST_MODE = True

    if argv.use_history:#any("h" in s for s in unused_argv):
        if not os.path.isfile(HISTORY_FILE) and not os.path.isfile(TRAINING_HISTORY_LOG_FILE):
            print("\033[1;31mWARNING: No 'history' file in 'models' folder. No history read\033[0m")
        else:
            with open(HISTORY_FILE, 'r', encoding = "utf-8") as hist_file:
                history = [line.rstrip("\n") for line in hist_file]
                for hist_entry in history:
                    hist_entry = hist_entry.split("::")
                    name=hist_entry[0]
                    time=hist_entry[1]
                    ttime=0
                    v = None
                    if len(hist_entry) > 2:
                        v = hist_entry[2]
                    if name not in EXECUTED_MODELS:
                        EXECUTED_MODELS[name] = {}
                        EXECUTED_MODELS[name][mtime] = float(time)
                        EXECUTED_MODELS[name][version] = VersionLog()
                        if v is not None and v is not "":
                            EXECUTED_MODELS[name][version].addExecutedVersion(v)
                            #needs to be taken from seperate file
                            #EXECUTED_MODELS[name][train_time] = float(ttime)
                    else:
                        if EXECUTED_MODELS[name][mtime] < float(time):
                            EXECUTED_MODELS[name][mtime] = float(time)
                            EXECUTED_MODELS[name][version].clean()
                        if v is not None and v is not "":
                            EXECUTED_MODELS[name][version].addExecutedVersion(v)
                            #EXECUTED_MODELS[name][train_time] = float(ttime)
            with open(TRAINING_HISTORY_LOG_FILE, "r") as t_hist_file:
                t_history = [line.rstrip("\n") for line in t_hist_file]
                for t_entry in t_history:
                    name,v,t = t_entry.split("::")
                    t = float(t)
                    if name in EXECUTED_MODELS:
                        if EXECUTED_MODELS[name][mtime] < t and EXECUTED_MODELS[name][version].executed(v) is not VersionLog.EXECUTED:
                            EXECUTED_MODELS[name][version].addExecutingVersion(v,t)

                            
    if argv.no_log:
        NO_LOG = True
    else:
        NO_LOG = False
        
    LOGGER = set_logger(test_mode = TEST_MODE, no_log = NO_LOG, log_file = LOG_FILE)
    add_script_dir_to_PATH(MODELS_DIR)
    _main(file_path)
    
    
if __name__ == "__main__":  
    main()
  
