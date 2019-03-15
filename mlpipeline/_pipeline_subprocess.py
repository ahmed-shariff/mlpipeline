import os
import sys
import importlib.util
import shutil
import socket
import argparse
import logging
try:
    import mlflow
except ImportError:
    pass
    
from datetime import datetime

from mlpipeline.utils import (_experimentModeKeys,
                              ExecutionModeKeys,
                              version_parameters,
                              log_special_tokens,
                              VersionLog,
                              console_colors,
                              log,
                              set_logger,
                              add_script_dir_to_PATH,
                              use_mlflow,
                              MetricContainer)


from mlpipeline.global_values import (EXPERIMENTS_DIR,
                                      NO_LOG,
                                      EXECUTED_EXPERIMENTS,
                                      USE_BLACKLIST,
                                      EXPERIMENT_MODE,
                                      mtime,
                                      version,
                                      train_time,
                                      vless)

def _main(file_path):
    current_experiment, version_name_s, clean_experiment_dir = _get_experiment(file_path)
    if current_experiment is None:
        sys.exit(3)

    log("Experiment loaded: {0}".format(current_experiment.name))
    if EXPERIMENT_MODE == _experimentModeKeys.TEST:
        log("Mode: {}{}".format(console_colors.YELLOW_FG, log_special_tokens.MODE_TESTING),
	    modifier_1 = console_colors.BOLD,
	    modifier_2 = console_colors.GREEN_FG)
    elif EXPERIMENT_MODE == _experimentModeKeys.EXPORT:
        log("Mode: {}{}".format(console_colors.YELLOW_FG, log_special_tokens.MODE_EXPORTING),
	    modifier_1 = console_colors.BOLD,
	    modifier_2 = console_colors.MEGENTA_FG)
    else:
        log("Mode: {}{}".format(console_colors.RED_FG, log_special_tokens.MODE_RUNNING),
	    modifier_1 = console_colors.BOLD,
	    modifier_2 = console_colors.GREEN_FG)
        
    if EXPERIMENT_MODE == _experimentModeKeys.EXPORT:
        for version_name in version_name_s:
            experiment_dir_suffix = "-" + experiment_dir_suffix if experiment_dir_suffix is not None else version_name
            output_dir = "{}/outputs".format(EXPERIMENTS_DIR.rstrip("/"))
            experiment_dir="{}/experiment_ckpts/{}{}".format(output_dir,
                                                   current_experiment.name.split(".")[-2],
                                                   experiment_dir_suffix)

            version_spec = current_experiment.versions.get_version(version_name)
            current_experiment.pre_execution_hook(version_spec, experiment_dir)
            current_experiment.setup_model(version_spec)
            log("Exporting model for version: {}".format(version_spec))
            current_experiment.export_model(version_spec)
            log("Exported model".format(version_spec))
    else:
        version_name = version_name_s
        _add_to_and_return_result_string("Experiment: {0}".format(current_experiment.name), True)
        _add_to_and_return_result_string("Version: {0}".format(version_name))
        if version_name is None:
            log("No Version Specifications",
                logging.WARNING,
                modifier_1 = console_colors.RED_FG,
                modifier_2 = console_colors.BOLD)
        else:
            log("version loaded: {0}".format(version_name),
              modifier_1 = console_colors.GREEN_FG,
              modifier_2 = console_colors.BOLD)
      

        version_spec = current_experiment.versions.get_version(version_name)
      
        batch_size = version_spec[version_parameters.BATCH_SIZE]
        experiment_dir_suffix = version_spec[version_parameters.EXPERIMENT_DIR_SUFFIX]
        dataloader = version_spec[version_parameters.DATALOADER]

        log("Version_spec: {}".format(version_spec))

        record_training = False
        if EXPERIMENT_MODE == _experimentModeKeys.TEST:
            experiment_dir = "{0}/outputs/experiment_ckpts/temp".format(EXPERIMENTS_DIR.rstrip("/"))
            shutil.rmtree(experiment_dir, ignore_errors=True)
        else:
            experiment_dir_suffix = "-" + experiment_dir_suffix if experiment_dir_suffix is not None else version_name
            output_dir = "{}/outputs".format(EXPERIMENTS_DIR.rstrip("/"))
            experiment_dir="{}/experiment_ckpts/{}{}".format(output_dir,
                                                   current_experiment.name.split(".")[-2],
                                                   experiment_dir_suffix)
            record_training = True
            if use_mlflow:
                tracking_uri = os.path.abspath("{}/{}".format(output_dir, "mlruns"))
                mlflow.set_tracking_uri(tracking_uri)
                mlflow.set_experiment(current_experiment.name)
                # Delete runs with the same name as the current version
                mlflow_client = mlflow.tracking.MlflowClient(tracking_uri)
                experiment_ids = [exp.experiment_id
                              for exp in mlflow_client.list_experiments() if current_experiment.name == exp.name]
                if len(experiment_ids) > 0:
                    run_infos = mlflow_client.list_run_infos(experiment_ids[0])
                    run_uuids = [run_info.run_uuid for run_info in run_infos \
                                 for run_tag in mlflow_client.get_run(run_info.run_uuid).data.tags \
                                 if run_tag.key == mlflow.utils.mlflow_tags.MLFLOW_RUN_NAME and run_tag.value == version_name]
                    for run_uuid in run_uuids:
                        mlflow_client.delete_run(run_uuid)
                mlflow.start_run(run_name = version_name, source_name = current_experiment.name)

                # Logging the versions params
                for k,v in version_spec.items():
                    mlflow.log_param(k,str(v))
                
    eval_complete=False
    #LOGGER.setLevel(logging.INFO)

    
    train_results = ""
    eval_results = ""
    
    try:
        if clean_experiment_dir and current_experiment.allow_delete_experiment_dir:
            current_experiment.clean_experiment_dir(experiment_dir)
            log("Cleaned experiment dir", modifier_1 = console_colors.RED_FG)
        current_experiment.pre_execution_hook(version_spec, experiment_dir)
        current_experiment.setup_model(version_spec)
        os.makedirs(experiment_dir, exist_ok = True)
        current_experiment.copy_related_files(experiment_dir)
        if EXPERIMENT_MODE == _experimentModeKeys.TEST:
            test__eval_steps = 1
            train_eval_steps = 1
        else:
            test__eval_steps = dataloader.get_test_sample_count()
            train_eval_steps = dataloader.get_train_sample_count()

        _save_training_time(current_experiment, version_name)
        classification_steps = _get_training_steps(ExecutionModeKeys.TRAIN, current_experiment, clean_experiment_dir)
        log("Steps: {0}".format(classification_steps))
        if classification_steps > 0:
            train_output = current_experiment.train_loop(dataloader.get_train_input(), classification_steps, version_spec)
            if isinstance(train_output, MetricContainer):
                train_output = train_output.log_metrics(log_to_file = False, complete_epoc = True)
            if isinstance(train_output, str):
                log("Experiment traning loop output: {0}".format(train_output))
            log(log_special_tokens.TRAINING_COMPLETE)
        else:
            log("No training. Loaded previous experiment environment")

        try:
            log("Training evaluation started: {0} steps".format(train_eval_steps))
            train_results = current_experiment.evaluate_loop(dataloader.get_train_input(mode = ExecutionModeKeys.TEST),
                                                             steps = train_eval_steps,
                                                             version = version_spec)
            log("Eval on train set: ")
            if isinstance(train_results, MetricContainer):
                train_results = train_results.log_metrics(complete_epoc = True, name_prefix = "TRAIN_")
            elif isinstance(train_results, str):
                log("{0}".format(train_results))
            else:
                raise ValueError("The output of `evaluate_loop` should be a string or a `MetricContainer`")
        except Exception as e:
            train_results = "Training evaluation failed: {0}".format(str(e))
            log(train_results, logging.ERROR)
            if EXPERIMENT_MODE == _experimentModeKeys.TEST:
                raise
            
        try:
            log("Testing evaluation started: {0} steps".format(test__eval_steps))
            eval_results = current_experiment.evaluate_loop(dataloader.get_test_input(),
						            steps = test__eval_steps,
                                                             version = version_spec)
            log("Eval on train set:")
            if isinstance(eval_results, MetricContainer):
                eval_results = eval_results.log_metrics(complete_epoc = True, name_prefix = "TEST_")
            elif isinstance(eval_results, str):
                log("{0}".format(eval_results))
            else:
                raise ValueError("The output of `evaluate_loop` should be a string or a `MetricContainer`")
        except Exception as e:
            eval_results = "Test evaluation failed: {0}".format(str(e))
            log(eval_results, logging.ERROR)
            if EXPERIMENT_MODE == _experimentModeKeys.TEST:
                raise

        log("Experiment evaluation complete")
        _add_to_and_return_result_string("Eval on train set: {0}".format(train_results))
        _add_to_and_return_result_string("Eval on test  set: {0}".format(eval_results))
        _add_to_and_return_result_string("-------------------------------------------")
        _add_to_and_return_result_string("EXECUTION SUMMERY:")
        _add_to_and_return_result_string("Number of epocs: {0}".format(version_spec[version_parameters.EPOC_COUNT]))
        _add_to_and_return_result_string("Parameters for this version: {0}".format(version_spec))
        _add_to_and_return_result_string("-------------------------------------------")
        _add_to_and_return_result_string("EXPERIMENT SUMMERY:")
        _add_to_and_return_result_string(current_experiment.summery)
        _add_to_and_return_result_string("-------------------------------------------")
        _add_to_and_return_result_string("DATALOADER	 SUMMERY:")
        _add_to_and_return_result_string(dataloader.summery)
        if record_training and not NO_LOG:
            _save_results_to_file(_add_to_and_return_result_string(), current_experiment)

    except Exception as e:
        if EXPERIMENT_MODE == _experimentModeKeys.TEST:
            raise
        else:
            log("Exception: {0}".format(str(e)), logging.ERROR)
            sys.exit(1)
    if EXPERIMENT_MODE == _experimentModeKeys.RUN and use_mlflow:
        mlflow.end_run()

    
def _get_training_steps(mode, experiment, clean_experiment_dir):
    if EXPERIMENT_MODE == _experimentModeKeys.TEST:
        return 1
    else:
        current_version = experiment.get_current_version()
        complete_steps =  current_version[version_parameters.EPOC_COUNT] * \
            current_version[version_parameters.DATALOADER].get_train_sample_count() / \
            current_version[version_parameters.BATCH_SIZE]
        global_step = experiment.get_trained_step_count()
        if global_step is None or experiment.reset_steps:
            return complete_steps
        
                #TODO: why did i add the reset_step here?
        elif clean_experiment_dir and not experiment.allow_delete_experiment_dir and experiment.reset_steps:
            return complete_steps
        else:
            if complete_steps > global_step:
                return complete_steps - global_step
            else:
                return 0
      
def _get_experiment(file_path, just_return_experiment=False):
    # Import and load the experiment
    spec = importlib.util.spec_from_file_location(file_path.split("/")[-1],file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    clean_experiment_dir = False
    experiment = None
    try:
        experiment = module.EXPERIMENT
        experiment.name = file_path
    except:
        log("{0} is not a experiment script. It does not contain a `EXPERIMENT` global variable".format(file_path))
        return None, None, False
    experiment._collect_related_files(EXPERIMENTS_DIR, [os.path.abspath(module.__file__)])
    # TODO: why did i add this in the first place??
    # if just_return_experiment:
    #	  print("\033[1;33mJust returning module\033[1;0m")
    #	  return module

    # Figure our which version should be executed next
    returning_version = None
    try:
        versions = experiment.versions
    except:
        versions = None
    log("{0}{1}Processing experiment: {2}{3}".format(console_colors.BOLD,
						console_colors.BLUE_FG,
						experiment.name,
						console_colors.RESET))

    if EXPERIMENT_MODE == _experimentModeKeys.EXPORT:
        return experiment, [k for v,k in sorted(versions._versions.items(), key=lambda x:x[1][version_parameters.ORDER])], False
    
    ## Get the training history. i.e. the time stamps of each training launched
    with open(TRAINING_HISTORY_LOG_FILE, "r") as t_hist_file:
        t_history = [line.rstrip("\n") for line in t_hist_file]
        all_history = [t_entry.split("::") for t_entry in t_history]
        module_history = [(v,float(t)) for n,v,t in all_history if n == experiment.name]

    if file_path not in EXECUTED_EXPERIMENTS:
        EXECUTED_EXPERIMENTS[experiment.name] = {}
        EXECUTED_EXPERIMENTS[experiment.name][train_time]=0
        EXECUTED_EXPERIMENTS[experiment.name][version]=VersionLog()

    EXECUTED_EXPERIMENTS[experiment.name][mtime] = os.path.getmtime(file_path)

    ## Determine if training should be started from scratch or should resume training
    ## Here, the modified time is used as an indicator.
    ## If there is a entry in the training shitory that is greater than the the modified time,
    ## that implies the the experiment was modeified later, hence the training should restart from scratch.
    reset_experiment_dir = True
    modified_time = os.path.getmtime(file_path)
    for v,t in module_history:
        if t > modified_time:
            reset_experiment_dir = False
    if reset_experiment_dir:
        clean_experiment_dir = True
        EXECUTED_EXPERIMENTS[experiment.name][version].clean()
    else:
        # If a training had started and not completed, resume the training of that version
        versions__ = [v_ for v_ in versions._versions.keys()]
        for v,t in module_history:
            if t > modified_time:
                if EXECUTED_EXPERIMENTS[experiment.name][version].executed(v) is not VersionLog.EXECUTED and v in versions__:
                    modified_time = t
                    returning_version = v
    ## If there are no training sessions to be resumed, decide which version to execute next based on the ORDER set in the version
    if returning_version is None:
        #TODO: check if this line works:
        for v,k in sorted(versions._versions.items(), key=lambda x:x[1][version_parameters.ORDER]):
            if EXECUTED_EXPERIMENTS[experiment.name][version].executed(v) is not VersionLog.EXECUTED:
                returning_version = v
                clean_experiment_dir = True
    log("Executed versions: {0}".format(EXECUTED_EXPERIMENTS[experiment.name][version].executed_versions),
        log_to_file=False)
    if returning_version is None:
        return None, None, False
    return experiment, returning_version, clean_experiment_dir
    

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

def _save_training_time(experiment, version_):
    if EXPERIMENT_MODE == _experimentModeKeys.TEST:
        return
    name = experiment.name
    with open(TRAINING_HISTORY_LOG_FILE, "a") as log_file:
        time = datetime.now().timestamp()
        EXECUTED_EXPERIMENTS[name][version].addExecutingVersion(version_, time)
        log("Executing version: {0}".format(EXECUTED_EXPERIMENTS[experiment.name][version].executing_version),
            log_to_file=False)
        log_file.write("{0}::{1}::{2}\n".format(name,
                                                EXECUTED_EXPERIMENTS[name][version].executing_version,
                                                time))

    
def _save_results_to_file(resultString, experiment):#experiment, result, train_result, dataloader, training_done, experiment_dir):
    modified_dt = datetime.isoformat(datetime.fromtimestamp(EXECUTED_EXPERIMENTS[experiment.name][mtime]))
    result_dt = datetime.now().isoformat()
  
    #_add_to_and_return_result_string("\n[{0}]:ml-pipline: output: \n".format(result_dt))
    with open(OUTPUT_FILE, 'a', encoding = "utf-8") as outfile:
        outfile.write("\n[{0}]:ml-pipline: output: \n".format(result_dt))
        outfile.write(resultString)
    with open(HISTORY_FILE, 'a', encoding = "utf-8") as hist_file:
        hist_file.write("{0}::{1}::{2}\n".format(experiment.name,
                                                 EXECUTED_EXPERIMENTS[experiment.name][mtime],
                                                 EXECUTED_EXPERIMENTS[experiment.name][version].executing_version))
    
    EXECUTED_EXPERIMENTS[experiment.name][version].moveExecutingToExecuted()


def main():
    parser = argparse.ArgumentParser(description="Machine Learning Pipeline")
    parser.add_argument("file_path", help='The file path of the experiment to be executed')
    parser.add_argument("experiments_dir", help='The directory in which the experiments reside, also where the results are to stored')
    parser.add_argument('-r','--run', help='Will set the pipeline to execute the pipline fully, if not set will be executed in test mode', action = 'store_true')
    parser.add_argument('-u','--use-history', help='If set will use the history log to determine if a experiment script has been executed.', action = 'store_true')
    parser.add_argument('-n','--no_log', help='If set non of the logs will be appended to the log files.', action = 'store_true')
    parser.add_argument('-e','--export', help='If set, will run the experiment in export mode instead of training/eval loop.', action = 'store_true')
    argv = parser.parse_args()
    
    global EXPERIMENT_MODE
    global NO_LOG
    global LOGGER
    global EXPERIMENTS_DIR
    global HISTORY_FILE
    global LOG_FILE
    global OUTPUT_FILE
    global TRAINING_HISTORY_LOG_FILE
    
    file_path = argv.file_path
    EXPERIMENTS_DIR = argv.experiments_dir
    
    hostName = socket.gethostname()
    EXPERIMENTS_DIR_OUTPUTS = EXPERIMENTS_DIR + "/outputs"
    OUTPUT_FILE = EXPERIMENTS_DIR_OUTPUTS + "/output-{0}".format(hostName)
    HISTORY_FILE = EXPERIMENTS_DIR_OUTPUTS + "/history-{0}".format(hostName)
    TRAINING_HISTORY_LOG_FILE = EXPERIMENTS_DIR_OUTPUTS + "/t_history-{0}".format(hostName)
    LOG_FILE = EXPERIMENTS_DIR_OUTPUTS + "/log-{0}".format(hostName)
    open(OUTPUT_FILE, "a").close()
    open(HISTORY_FILE, "a").close()
    open(TRAINING_HISTORY_LOG_FILE, "a").close()
    open(LOG_FILE, "a").close()

    if argv.export:
        EXPERIMENT_MODE = _experimentModeKeys.EXPORT
    if argv.run:#any("r" in s for s in unused_argv) :
        EXPERIMENT_MODE = _experimentModeKeys.RUN
    else:
        EXPERIMENT_MODE = _experimentModeKeys.TEST

    if True:#argv.use_history:#any("h" in s for s in unused_argv):
        if not os.path.isfile(HISTORY_FILE) and not os.path.isfile(TRAINING_HISTORY_LOG_FILE):
            print("\033[1;31mWARNING: No 'history' file in 'experiments' folder. No history read\033[0m")
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
                    if name not in EXECUTED_EXPERIMENTS:
                        EXECUTED_EXPERIMENTS[name] = {}
                        EXECUTED_EXPERIMENTS[name][mtime] = float(time)
                        EXECUTED_EXPERIMENTS[name][version] = VersionLog()
                        if v is not None and v is not "":
                            EXECUTED_EXPERIMENTS[name][version].addExecutedVersion(v)
                            #needs to be taken from seperate file
                            #EXECUTED_EXPERIMENTS[name][train_time] = float(ttime)
                    else:
                        if EXECUTED_EXPERIMENTS[name][mtime] < float(time):
                            EXECUTED_EXPERIMENTS[name][mtime] = float(time)
                            EXECUTED_EXPERIMENTS[name][version].clean()
                        if v is not None and v is not "":
                            EXECUTED_EXPERIMENTS[name][version].addExecutedVersion(v)
                            #EXECUTED_EXPERIMENTS[name][train_time] = float(ttime)
            with open(TRAINING_HISTORY_LOG_FILE, "r") as t_hist_file:
                t_history = [line.rstrip("\n") for line in t_hist_file]
                for t_entry in t_history:
                    name,v,t = t_entry.split("::")
                    t = float(t)
                    if name in EXECUTED_EXPERIMENTS:
                        if EXECUTED_EXPERIMENTS[name][mtime] < t and EXECUTED_EXPERIMENTS[name][version].executed(v) is not VersionLog.EXECUTED:
                            EXECUTED_EXPERIMENTS[name][version].addExecutingVersion(v,t)

                            
    if argv.no_log:
        NO_LOG = True
    else:
        NO_LOG = False
        
    LOGGER = set_logger(experiment_mode = EXPERIMENT_MODE, no_log = NO_LOG, log_file = LOG_FILE)
    add_script_dir_to_PATH(EXPERIMENTS_DIR)
    _main(file_path)
    
    
if __name__ == "__main__":  
    main()
  
