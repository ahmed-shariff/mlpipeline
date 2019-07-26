import os
import sys
import shutil
import socket
import argparse
import logging
import traceback
import mlflow

from datetime import datetime

from mlpipeline.utils import (ExperimentModeKeys,
                              ExecutionModeKeys,
                              version_parameters,
                              log_special_tokens,
                              _VersionLog,
                              console_colors,
                              log,
                              set_logger,
                              add_script_dir_to_PATH,
                              is_no_log,
                              MetricContainer,
                              _PipelineConfig,
                              _load_file_as_module)
from mlpipeline.base._utils import DummyDataloader
import mlpipeline._default_configurations as _default_config
CONFIG = _PipelineConfig()


class _ExecutedExperiment():
    '''
    Class used to track the state of the experiments and their versions.
    '''
    def __init__(self, version, modified_time):
        # uses the proprtty setter
        self.version = version
        self._modified_time = modified_time

    @property
    def version(self):
        '''
        The _VersionLog object of the corresponding experiment
        '''
        return self._version

    @version.setter
    def version(self, value):
        assert isinstance(value, _VersionLog)
        self._version = value

    @property
    def modified_time(self):
        '''
        Time stamp of the last time the experiment was modified
        '''
        return self._modified_time

    @modified_time.setter
    def modified_time(self, value):
        self._modified_time = value


class _AddToAndReturnResultString():
    def __init__(self,):
        self.result_string = ""

    def __call__(self,  message=None, reset_result_string=False, indent=True):
        if message is not None:
            if indent:
                message = "\t\t" + message
            if reset_result_string:
                self.result_string = message + "\n"
            else:
                self.result_string += message + "\n"
        return self.result_string


def _experiment_main_loop(file_path, whitelist_versions=None, blacklist_versions=None):
    '''
    Returns False if there are no more versions to execute or a version resulted in an exception
    Returns True otherwise.
    '''
    current_experiment, version_name_s, \
        clean_experiment_dir = _get_experiment(file_path,
                                               whitelist_versions=whitelist_versions,
                                               blacklist_versions=blacklist_versions)
    _add_to_and_return_result_string = _AddToAndReturnResultString()
    if current_experiment is None:
        if CONFIG.cmd_mode:
            sys.exit(3)
        else:
            return False
    log_special_tokens.log_experiment_started()
    log("Experiment loaded: {0}".format(current_experiment.name))
    if CONFIG.experiment_mode == ExperimentModeKeys.TEST:
        log_special_tokens.log_mode_test()
    elif CONFIG.experiment_mode == ExperimentModeKeys.EXPORT:
        log_special_tokens.log_mode_exporting()
    else:
        log_special_tokens.log_mode_train()

    if CONFIG.experiment_mode == ExperimentModeKeys.EXPORT:
        for version_name, version_spec in version_name_s:
            experiment_dir, _ = _get_experiment_dir(current_experiment.name.split(".")[-2],
                                                    version_spec,
                                                    CONFIG.experiment_mode)
            current_experiment._current_version = version_spec
            current_experiment._experiment_dir = experiment_dir

            try:
                current_experiment.setup_model()
            except NotImplementedError:
                log("`setup_model` not implemented. Ignoring.")
            log("Exporting model for version: {}".format(version_name))
            current_experiment.export_model()
            log("Exported model {}".format(version_name))
        log_special_tokens.log_experiment_ended()
        if CONFIG.cmd_mode:
            sys.exit(3)
        else:
            return False
    else:
        version_name = version_name_s
        _add_to_and_return_result_string("Experiment: {0}".format(current_experiment.name), True)
        _add_to_and_return_result_string("Version: {0}".format(version_name))
        if version_name is None:
            log("No Version Specifications",
                logging.WARNING,
                modifier_1=console_colors.RED_FG,
                modifier_2=console_colors.BOLD)
        else:
            log("version loaded: {0}".format(version_name),
                modifier_1=console_colors.GREEN_FG,
                modifier_2=console_colors.BOLD)

        version_spec = current_experiment.versions.get_version(version_name)
        dataloader = version_spec[version_parameters.DATALOADER]
        if dataloader is not None:
            dataloader = dataloader()
        else:
            dataloader = DummyDataloader()

        log("Version_spec: {}".format(version_spec))

        experiment_dir, tracking_uri = _get_experiment_dir(current_experiment.name.split(".")[-2],
                                                           version_spec,
                                                           CONFIG.experiment_mode)
        record_training = True if CONFIG.experiment_mode != ExperimentModeKeys.TEST else False
        if clean_experiment_dir and current_experiment.allow_delete_experiment_dir:
            try:
                current_experiment.clean_experiment_dir(experiment_dir)
                log("Cleaned experiment dir", modifier_1=console_colors.RED_BG)
            except NotImplementedError:
                log("`experiment.clean_experiment_dir` not implemened."
                    "contents in the experiment_dir will not be changed", level=logging.WARNING)

        run_id = _get_mlflow_run_id(tracking_uri, current_experiment, clean_experiment_dir, version_name)

        current_experiment._current_version = version_spec
        current_experiment._experiment_dir = experiment_dir
        current_experiment._dataloader = dataloader
        
        mlflow.start_run(run_name=version_name, run_id=run_id)
        # Logging the versions params
        for k, v in version_spec.items():
            mlflow.log_param(k, str(v))

        # eval_complete=False
        # LOGGER.setLevel(logging.INFO)
        train_results = ""
        eval_results = ""

        try:
            try:
                current_experiment.setup_model()
            except NotImplementedError:
                log("`setup_model` not implemented. Ignoring.")
            try:
                current_experiment.pre_execution_hook(mode=CONFIG.experiment_mode)
            except NotImplementedError:
                log("`pre_execution_hook` not implemented. Ignoring.")
            os.makedirs(experiment_dir, exist_ok=True)
            current_experiment.copy_related_files(experiment_dir)
            try:
                test__eval_steps = dataloader.get_test_sample_count()
            except NotImplementedError:
                test__eval_steps = None
            try:
                train_eval_steps = dataloader.get_train_sample_count()
            except NotImplementedError:
                train_eval_steps = None
            if CONFIG.experiment_mode == ExperimentModeKeys.TEST:
                test__eval_steps = 1 if test__eval_steps is not None else None
                train_eval_steps = 1 if train_eval_steps is not None else None

            _save_training_time(current_experiment, version_name)
            classification_steps = _get_training_steps(ExecutionModeKeys.TRAIN,
                                                       current_experiment,
                                                       dataloader,
                                                       clean_experiment_dir,
                                                       version_spec)
            log("Steps: {0}".format(classification_steps
                                    if classification_steps is not None else 'unspecified'))

            try:
                input_fn = dataloader.get_train_input(mode=ExecutionModeKeys.TRAIN)
            except NotImplementedError:
                log('`get_train_input` not implemented for training. Ignoring')
            if classification_steps is not None and classification_steps > 0:
                try:
                    train_output = current_experiment.train_loop(
                        input_fn=input_fn,
                        steps=classification_steps)
                except Exception as e:
                    train_results = "Training loop failed: {0}".format(str(e))
                    log(train_results, logging.ERROR)
                    log(traceback.format_exc(), logging.ERROR)
                    if CONFIG.experiment_mode == ExperimentModeKeys.TEST:
                        raise
                if isinstance(train_output, MetricContainer):
                    train_output = train_output.log_metrics(log_to_file=False, complete_epoc=True)
                if isinstance(train_output, str):
                    log("Experiment traning loop output: {0}".format(train_output))
                log(log_special_tokens.TRAINING_COMPLETE)
            else:
                log("No training. Loaded previous experiment environment")

            try:
                input_fn = dataloader.get_train_input(mode=ExecutionModeKeys.TEST)
            except NotImplementedError:
                log('`get_train_input` not implemented for evaluation. Ignoring')
            try:
                log("Training evaluation started: {0} steps"
                    .format(train_eval_steps if train_eval_steps is not None else 'unspecified'))
                train_results = current_experiment.evaluate_loop(
                    input_fn=input_fn,
                    steps=train_eval_steps)
                log("Eval on train set: ")
                if isinstance(train_results, MetricContainer):
                    train_results = train_results.log_metrics(complete_epoc=True, name_prefix="TRAIN_")
                elif isinstance(train_results, str):
                    log("{0}".format(train_results))
                else:
                    raise ValueError("The output of `evaluate_loop` should be"
                                     " a string or a `MetricContainer`")
            except NotImplementedError:
                log('`evaluate_loop` not implemented. Ignoring')
            except Exception as e:
                train_results = "Training evaluation failed: {0}".format(str(e))
                log(train_results, logging.ERROR)
                log(traceback.format_exc(), logging.ERROR)
                if CONFIG.experiment_mode == ExperimentModeKeys.TEST:
                    raise

            try:
                input_fn = dataloader.get_test_input()
            except NotImplementedError:
                log('`get_test_input` not implemented.')
                input_fn = None
            try:
                log("Testing evaluation started: {0} steps".
                    format(test__eval_steps if test__eval_steps is not None else 'unspecified'))
                eval_results = current_experiment.evaluate_loop(input_fn=input_fn,
                                                                steps=test__eval_steps)
                log("Eval on train set:")
                if isinstance(eval_results, MetricContainer):
                    eval_results = eval_results.log_metrics(complete_epoc=True, name_prefix="TEST_")
                elif isinstance(eval_results, str):
                    log("{0}".format(eval_results))
                else:
                    raise ValueError("The output of `evaluate_loop` should"
                                     " be a string or a `MetricContainer`")
            except NotImplementedError:
                log('`evaluate_loop` not implemented. Ignoring')
            except Exception as e:
                eval_results = "Test evaluation failed: {0}".format(str(e))
                log(eval_results, logging.ERROR)
                log(traceback.format_exc(), logging.ERROR)
                if CONFIG.experiment_mode == ExperimentModeKeys.TEST:
                    raise
            try:
                current_experiment.post_execution_hook(mode=CONFIG.experiment_mode)
            except NotImplementedError:
                log("`post_execution_hook` not implemented. Ignoring.")

            log("Experiment evaluation complete")
            _add_to_and_return_result_string("Eval on train set: {0}".format(train_results))
            _add_to_and_return_result_string("Eval on test  set: {0}".format(eval_results))
            _add_to_and_return_result_string("-------------------------------------------")
            _add_to_and_return_result_string("EXECUTION SUMMERY:")
            _add_to_and_return_result_string("Number of epocs: {0}".format(
                version_spec[version_parameters.EPOC_COUNT]))
            _add_to_and_return_result_string("Parameters for this version: {0}".format(version_spec))
            _add_to_and_return_result_string("-------------------------------------------")
            _add_to_and_return_result_string("EXPERIMENT SUMMERY:")
            _add_to_and_return_result_string(current_experiment.summery)
            _add_to_and_return_result_string("-------------------------------------------")
            _add_to_and_return_result_string("DATALOADER	 SUMMERY:")
            _add_to_and_return_result_string(dataloader.summery)
            if record_training and not CONFIG.no_log:
                _save_results_to_file(_add_to_and_return_result_string(), current_experiment)

        except Exception as e:
            mlflow.end_run(mlflow.entities.RunStatus.to_string(mlflow.entities.RunStatus.FAILED))
            if CONFIG.experiment_mode == ExperimentModeKeys.TEST:
                raise
            else:
                log("Exception: {0}".format(str(e)), logging.ERROR)
                log(traceback.format_exc(), logging.ERROR)
                if CONFIG.cmd_mode:
                    sys.exit(1)
                else:
                    return False
        mlflow.end_run()
    log_special_tokens.log_experiment_ended()
    return True


def _get_training_steps(mode, experiment, dataloader, clean_experiment_dir, version_spec):
    exceptions = 0
    try:
        epoc_count = version_spec[version_parameters.EPOC_COUNT]
    except KeyError:
        epoc_count = 1
        exceptions += 1
    try:
        train_samples = dataloader.get_train_sample_count()
        if train_samples is None:
            raise NotImplementedError()
    except (NotImplementedError, KeyError):
        train_samples = 1
        exceptions += 1
    if exceptions == 2:
        return None

    if CONFIG.experiment_mode == ExperimentModeKeys.TEST:
        return 1
    else:
        try:
            batch_size = version_spec[version_parameters.BATCH_SIZE]
        except KeyError:
            batch_size = 1
        complete_steps = batch_size * epoc_count * train_samples
        try:
            global_step = experiment.get_trained_step_count()
        except NotImplementedError:
            global_step = 0
        if global_step is None or experiment.reset_steps:
            return complete_steps
        elif clean_experiment_dir and not experiment.allow_delete_experiment_dir:
            return complete_steps
        else:
            if complete_steps > global_step:
                return complete_steps - global_step
            else:
                return 0


def _get_experiment_dir(experiment_name, version_spec, mode):
    experiment_dir_suffix = version_spec[version_parameters.EXPERIMENT_DIR_SUFFIX]
    if mode == ExperimentModeKeys.TEST:
        experiment_dir = os.path.join(CONFIG.experiments_outputs_dir, "experiment_ckpts/temp")
        tracking_uri = os.path.abspath(os.path.join(experiment_dir, "mlruns_tmp"))
        shutil.rmtree(experiment_dir, ignore_errors=True)
    else:
        experiment_dir_suffix = experiment_dir_suffix \
            if experiment_dir_suffix is not None else version_spec.name
        experiment_dir_suffix = "-" + experiment_dir_suffix
        experiment_dir = os.path.join(CONFIG.experiments_outputs_dir,
                                      "experiment_ckpts/{}{}".format(experiment_name,
                                                                     experiment_dir_suffix))
        tracking_uri = CONFIG.mlflow_tracking_uri
    return experiment_dir, tracking_uri


def _get_mlflow_run_id(tracking_uri, current_experiment, clean_experiment_dir, version_name):
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(current_experiment.name)
    # Delete runs with the same name as the current version
    mlflow_client = mlflow.tracking.MlflowClient(tracking_uri)
    experiment_ids = [exp.experiment_id
                      for exp in mlflow_client.list_experiments()
                      if current_experiment.name == exp.name]
    current_experiment.mlflow_client = mlflow_client
    run_id = None
    if len(experiment_ids) > 0:
        runs = mlflow_client.search_runs(experiment_ids,
                                         f"tags.mlflow.runName = '{version_name}'")
        assert len(runs) <= 1, "There cannot be more than one active run for a version"
        if len(runs) > 0:
            if clean_experiment_dir and current_experiment.allow_delete_experiment_dir:
                mlflow_client.delete_run(runs[0].info.run_uuid)
            else:
                run_id = runs[0].info.run_id
    return run_id


def _get_experiment(file_path,
                    whitelist_versions=None,
                    blacklist_versions=None,
                    just_return_experiment=False):
    # Import and load the experiment
    module = _load_file_as_module(file_path)
    clean_experiment_dir = False
    experiment = None
    try:
        experiment = module.EXPERIMENT
        experiment.name = file_path
    except Exception:
        log("{0} is not a experiment script. "
            "It does not contain a `EXPERIMENT` global variable".format(file_path))
        return None, None, False

    if just_return_experiment:
        return experiment, None, None

    experiment._collect_related_files(CONFIG.experiments_dir, [os.path.abspath(module.__file__)])
    # Figure our which version should be executed next
    returning_version = None
    try:
        versions = experiment.versions
    except Exception:
        versions = None

    if whitelist_versions is not None or blacklist_versions is not None:
        versions.filter_versions(whitelist_versions=whitelist_versions,
                                 blacklist_versions=blacklist_versions)

    log("{0}{1}Processing experiment: {2}{3}".format(console_colors.BOLD,
                                                     console_colors.BLUE_FG,
                                                     experiment.name,
                                                     console_colors.RESET))

    if CONFIG.experiment_mode == ExperimentModeKeys.EXPORT:
        return experiment, versions.get_versions(), False

    # Get the training history. i.e. the time stamps of each training launched
    with open(CONFIG.training_history_log_file, "r") as t_hist_file:
        t_history = [line.rstrip("\n") for line in t_hist_file]
        all_history = [t_entry.split("::") for t_entry in t_history]
        module_history = [(v, float(t)) for n, v, t in all_history if n == experiment.name]

    modified_time = os.path.getmtime(file_path)
    if file_path not in CONFIG.executed_experiments:
        CONFIG.executed_experiments[experiment.name] = _ExecutedExperiment(version=_VersionLog(),
                                                                           modified_time=modified_time)
    else:
        CONFIG.executed_experiments[experiment.name].modified_time = modified_time

    # Determine if training should be started from scratch or should resume training
    # Here, the modified time is used as an indicator.
    # If there is a entry in the training hitory that is greater than the the modified time,
    # that implies the the experiment was modeified later, hence the training should restart from scratch.
    reset_experiment_dir = True
    modified_time = os.path.getmtime(file_path)
    for v, t in module_history:
        if t > modified_time:
            reset_experiment_dir = False
    if reset_experiment_dir:
        clean_experiment_dir = True
        CONFIG.executed_experiments[experiment.name].version.clean()
    else:
        # If a training had started and not completed, resume the training of that version
        versions__ = versions.get_version_names()
        for v, t in module_history:
            if t > modified_time:
                if CONFIG.executed_experiments[experiment.name].version.executed(v)\
                   is not _VersionLog.EXECUTED and v in versions__:
                    modified_time = t
                    returning_version = v
    # If there are no training sessions to be resumed, decide which version to execute next
    # based on the ORDER set in the version
    if returning_version is None:
        # TODO: check if this line works:
        for v, k in versions.get_versions():
            if CONFIG.executed_experiments[experiment.name].version.executed(v) is not _VersionLog.EXECUTED:
                returning_version = v
                clean_experiment_dir = True
    log("Executed versions: {0}".format(
        CONFIG.executed_experiments[experiment.name].version.executed_versions),
        log_to_file=False)
    if returning_version is None:
        return None, None, False
    return experiment, returning_version, clean_experiment_dir


def _save_training_time(experiment, version_):
    if CONFIG.experiment_mode == ExperimentModeKeys.TEST:
        return
    name = experiment.name
    with open(CONFIG.training_history_log_file, "a") as log_file:
        time = datetime.now().timestamp()
        CONFIG.executed_experiments[name].version.addExecutingVersion(version_, time)
        log("Executing version: {0}".format(
            CONFIG.executed_experiments[experiment.name].version.executing_version),
            log_to_file=False)
        log_file.write("{0}::{1}::{2}\n".format(name,
                                                CONFIG.executed_experiments[name].version.executing_version,
                                                time))


def _save_results_to_file(resultString, experiment):
    modified_dt = datetime.isoformat(datetime.fromtimestamp(
        CONFIG.executed_experiments[experiment.name].modified_time))
    result_dt = datetime.now().isoformat()

    with open(CONFIG.output_file, 'a', encoding="utf-8") as outfile:
        outfile.write("\n[{0}]:ml-pipline: output: \n".format(result_dt))
        outfile.write(resultString)
    with open(CONFIG.history_file, 'a', encoding="utf-8") as hist_file:
        hist_file.write("{0}::{1}::{2}\n".format(
            experiment.name,
            CONFIG.executed_experiments[experiment.name].modified_time,
            CONFIG.executed_experiments[experiment.name].version.executing_version))

    CONFIG.executed_experiments[experiment.name].version.moveExecutingToExecuted()


def main():
    parser = argparse.ArgumentParser(description="Machine Learning Pipeline\n"
                                     "(DEPRECATED) use `mlpipeline`")
    parser.add_argument("file_path", help='The file path of the experiment to be executed')
    parser.add_argument("experiments_dir",
                        help='The directory in which the experiments reside,'
                        ' also where the results are to stored')
    parser.add_argument('-r', '--run',
                        help='Will set the pipeline to execute the pipline fully,'
                        ' if not set will be executed in test mode',
                        action='store_true')
    parser.add_argument('-u', '--use-history',
                        help='If set will use the history log to determine'
                        ' if a experiment script has been executed.',
                        action='store_true')
    parser.add_argument('-n', '--no_log',
                        help='If set non of the logs will be appended to the log files.',
                        action='store_true')
    parser.add_argument('-e', '--export',
                        help='If set, will run the experiment in export mode instead of training/eval loop.',
                        action='store_true')
    parser.add_argument('--whitelist-versions',
                        help='Of the versions added in the experiment script, the versions to execute',
                        nargs="*")
    parser.add_argument('--blacklist-versions',
                        help='Of the versions added in the experiment script, the versions not to execute',
                        nargs="*")
    argv = parser.parse_args()

    if argv.run:
        experiment_mode = ExperimentModeKeys.RUN
        if argv.export:
            log("Ignoring `-e/--export`")
    elif argv.export:
        experiment_mode = ExperimentModeKeys.EXPORT
    else:
        experiment_mode = ExperimentModeKeys.TEST

    _execute_exeperiment(argv.file_path,
                         argv.experiments_dir,
                         experiment_mode,
                         argv.no_log,
                         whitelist_versions=argv.whitelist_versions,
                         blacklist_versions=argv.blacklist_versions,
                         _cmd_mode=True)


def _execute_exeperiment(file_path,
                         experiments_dir,
                         experiment_mode=ExperimentModeKeys.TEST,
                         no_log=False,
                         whitelist_versions=None,
                         blacklist_versions=None,
                         experiments_output_dir=None,
                         mlflow_tracking_uri=None,
                         _cmd_mode=False,
                         multiprocessing_version_quque=None):
    '''
    Returns False if there are no more versions to execute or a version resulted in an exception
    Returns True otherwise.
    '''
    experiments_dir = os.path.abspath(experiments_dir)
    CONFIG.experiments_dir = experiments_dir
    CONFIG.experiment_mode = experiment_mode
    CONFIG.executed_experiments = {}
    hostName = socket.gethostname()
    EXPERIMENTS_DIR_OUTPUTS = experiments_output_dir or _default_config.OUTPUT_DIR.format(experiments_dir)
    CONFIG.experiments_outputs_dir = EXPERIMENTS_DIR_OUTPUTS
    CONFIG.output_file = os.path.join(EXPERIMENTS_DIR_OUTPUTS, "output-{0}".format(hostName))
    CONFIG.history_file = os.path.join(EXPERIMENTS_DIR_OUTPUTS, "history-{0}".format(hostName))
    CONFIG.training_history_log_file = os.path.join(EXPERIMENTS_DIR_OUTPUTS, "t_history-{0}".format(hostName))
    CONFIG.log_file = os.path.join(EXPERIMENTS_DIR_OUTPUTS, "log-{0}".format(hostName))
    CONFIG.mlflow_tracking_uri = mlflow_tracking_uri or CONFIG.mlflow_tracking_uri
    CONFIG.mlflow_tracking_uri = os.path.abspath(CONFIG.mlflow_tracking_uri)
    CONFIG.cmd_mode = _cmd_mode
    if no_log:
        CONFIG.no_log = True
    else:
        CONFIG.no_log = False

    CONFIG.logger = set_logger(experiment_mode=CONFIG.experiment_mode,
                               no_log=CONFIG.no_log,
                               log_file=CONFIG.log_file)

    if not os.path.exists(EXPERIMENTS_DIR_OUTPUTS):
        os.makedirs(EXPERIMENTS_DIR_OUTPUTS)
    open(CONFIG.output_file, "a").close()
    open(CONFIG.history_file, "a").close()
    open(CONFIG.training_history_log_file, "a").close()
    open(CONFIG.log_file, "a").close()

    if CONFIG.experiment_mode != ExperimentModeKeys.TEST:  # argv.use_history:
        if not os.path.isfile(CONFIG.history_file) and not os.path.isfile(CONFIG.training_history_log_file):
            print("\033[1;31mWARNING: No 'history' file in 'experiments' folder. No history read\033[0m")
        else:
            with open(CONFIG.history_file, 'r', encoding="utf-8") as hist_file:
                history = [line.rstrip("\n") for line in hist_file]
                for hist_entry in history:
                    hist_entry = hist_entry.split("::")
                    name = hist_entry[0]
                    time = hist_entry[1]
                    ttime = 0
                    v = None
                    if len(hist_entry) > 2:
                        v = hist_entry[2]
                    if name not in CONFIG.executed_experiments:
                        CONFIG.executed_experiments[name] = _ExecutedExperiment(_VersionLog(), float(time))
                        if v is not None and v != "":
                            CONFIG.executed_experiments[name].version.addExecutedVersion(v)
                    else:
                        if CONFIG.executed_experiments[name].modified_time < float(time):
                            CONFIG.executed_experiments[name].modified_time = float(time)
                            CONFIG.executed_experiments[name].version.clean()
                        if v is not None and v != "":
                            CONFIG.executed_experiments[name].version.addExecutedVersion(v)
            with open(CONFIG.training_history_log_file, "r") as t_hist_file:
                t_history = [line.rstrip("\n") for line in t_hist_file]
                for t_entry in t_history:
                    name, v, t = t_entry.split("::")
                    t = float(t)
                    if name in CONFIG.executed_experiments:
                        if CONFIG.executed_experiments[name].modified_time < t and \
                           CONFIG.executed_experiments[name].version.executed(v) is not _VersionLog.EXECUTED:
                            CONFIG.executed_experiments[name].version.addExecutingVersion(v, t)

    add_script_dir_to_PATH(CONFIG.experiments_dir)
    cwd = os.getcwd()
    os.chdir(CONFIG.experiments_dir)
    file_path = os.path.relpath(os.path.abspath(file_path), CONFIG.experiments_dir)
    if multiprocessing_version_quque is not None:
        output = _get_experiment(file_path, whitelist_versions, blacklist_versions, True)
        multiprocessing_version_quque.put(output[0].versions.get_version_names())
    else:
        output = _experiment_main_loop(file_path,
                                       whitelist_versions=whitelist_versions,
                                       blacklist_versions=blacklist_versions)
    os.chdir(cwd)
    return output


if __name__ == "__main__":
    main()
