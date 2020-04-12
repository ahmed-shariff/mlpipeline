import os
from pathlib import Path
from mlpipeline._pipeline import (_mlpipeline_main_loop, _init_pipeline)
from mlpipeline._pipeline_subprocess import (_execute_exeperiment,
                                             _get_experiment_dir,
                                             _ExecutedExperiment,
                                             _experiment_main_loop)
from mlpipeline.utils import (_load_file_as_module, PipelineConfig, _VersionLog, set_logger)
from mlpipeline.base import ExperimentWrapper, ExperimentABC
from mlpipeline.entities import (ExperimentModeKeys, console_colors)
from mlpipeline import log

__all__ = ['mlpipeline_execute_exeperiment', 'mlpipeline_execute_exeperiment_from_script', 'mlpipeline_execute_pipeline', 'get_experiment', 'ExperimentWrapper']


def mlpipeline_execute_pipeline(experiments,
                                experiments_dir,
                                experiment_mode=ExperimentModeKeys.TEST,
                                no_log=False,
                                mlflow_tracking_uri=None,
                                experiments_output_dir=None):
    '''
    This function can be used to execute the same operation of executimg mlpipeline, programatically.
    '''
    experiments_dir = os.path.abspath(experiments_dir) if experiments_dir is not None else None
    assert any([isinstance(e, ExperimentWrapper) for e in experiments])
    for experiment in experiments:
        experiment.file_path = os.path.relpath(os.path.abspath(experiment.file_path), experiments_dir)
    _init_pipeline(experiment_mode, experiments_dir, no_log, experiments_output_dir, mlflow_tracking_uri=mlflow_tracking_uri, _cmd_mode=True)
    _mlpipeline_main_loop(experiments)


# Need to integrate the functionality of the pipeline tracking training processes.
# For now the mlpipeline_execute_exeperiment_from_script is recommended
def mlpipeline_execute_exeperiment(experiment,
                                   experiment_mode=ExperimentModeKeys.TEST,
                                   whitelist_versions=None,
                                   blacklist_versions=None,
                                   pipeline_config=None):
    '''
    Warning: Experimental interface
    '''
    if pipeline_config is None:
        pipeline_config = PipelineConfig(experiments_dir="",
                                         experiments_outputs_dir="outputs",
                                         mlflow_tracking_uri=".mlruns")
    pipeline_config.experiment_mode = experiment_mode
    pipeline_config.output_file = Path(os.path.join(pipeline_config.experiments_outputs_dir, "output"))
    pipeline_config.history_file = Path(os.path.join(pipeline_config.experiments_outputs_dir, "history"))
    pipeline_config.training_history_log_file = Path(os.path.join(pipeline_config.experiments_outputs_dir, "t_history"))
    pipeline_config.log_file = Path(os.path.join(pipeline_config.experiments_outputs_dir, "log"))

    pipeline_config.output_file.parent.mkdir(parents=True, exist_ok=True)
    pipeline_config.history_file.parent.mkdir(parents=True, exist_ok=True)
    pipeline_config.training_history_log_file.parent.mkdir(parents=True, exist_ok=True)
    pipeline_config.log_file.parent.mkdir(parents=True, exist_ok=True)

    pipeline_config.output_file.touch()
    pipeline_config.history_file.touch()
    pipeline_config.training_history_log_file.touch()
    pipeline_config.log_file.touch()
    
    pipeline_config.logger = set_logger(experiment_mode=experiment_mode,
                                        no_log=False,
                                        log_file=pipeline_config.log_file)
    if not isinstance(experiment, ExperimentABC):
        log("`experiment` is not of type `mlpipeline.base.ExperimentABC`", 20)
    experiment.name = experiment.__class__.__name__
    experiment._collect_related_files(pipeline_config.experiments_dir)
    versions = experiment.versions
    
    log("{0}{1}Processing experiment: {2}{3}".format(console_colors.BOLD,
                                                     console_colors.BLUE_FG,
                                                     experiment.name,
                                                     console_colors.RESET))
    if whitelist_versions is not None or blacklist_versions is not None:
        versions.filter_versions(whitelist_versions=whitelist_versions,
                                 blacklist_versions=blacklist_versions)

    pipeline_config.executed_experiments[experiment.name] = _ExecutedExperiment(_VersionLog(), 0)
    if experiment_mode != ExperimentModeKeys.RUN:
        versions_list = versions.get_versions()
        _experiment_main_loop(experiment,
                              versions_list if experiment_mode == ExperimentModeKeys.EXPORT else versions_list[0][0],
                              True, pipeline_config)
    else:
        for v, k in versions.get_versions():
            if _experiment_main_loop(experiment, v, True, pipeline_config):
                pipeline_config.executed_experiments[experiment.name].version.addExecutingVersion(v, 0)
            else:
                log("Pipeline Stoped", 30)
            
        
        
    

    
def mlpipeline_execute_exeperiment_from_script(file_path,
                                               experiments_dir,
                                               experiment_mode=ExperimentModeKeys.TEST,
                                               no_log=False,
                                               whitelist_versions=None,
                                               blacklist_versions=None,
                                               mlflow_tracking_uri=None,
                                               experiments_output_dir=None):
    experiments_dir = os.path.abspath(experiments_dir)
    file_path = os.path.relpath(os.path.abspath(file_path), experiments_dir)
    while _execute_exeperiment(file_path,
                               experiments_dir=experiments_dir,
                               experiment_mode=experiment_mode,
                               no_log=no_log,
                               whitelist_versions=whitelist_versions,
                               blacklist_versions=blacklist_versions,
                               experiments_output_dir=experiments_output_dir,
                               mlflow_tracking_uri=mlflow_tracking_uri):
        if experiment_mode == ExperimentModeKeys.TEST:
            break


# TODO: Need to track the root of the project, or this becomes kind of ridiculous.
def get_experiment(file_path, experiment_dir, version_name, mlflow_tracking_uri=None, load_dataloader=False):
    cwd = os.getcwd()
    try:
        experiment_dir = os.path.abspath(experiment_dir)
        file_path = os.path.relpath(os.path.abspath(file_path), experiment_dir)
        print(f"Setting root directory to: {experiment_dir}")
        print(f"Loading experiment: {file_path}")
        os.chdir(experiment_dir)
        experiment = _load_file_as_module(file_path).EXPERIMENT
        experiment.name = os.path.relpath(file_path, experiment_dir)
        version_spec = experiment.versions.get_version(version_name)
        experiment_dir, _ = _get_experiment_dir(experiment.name.split(".")[-2],
                                                version_spec,
                                                None, PipelineConfig(experiment_mode=ExperimentModeKeys.RUN))
        # if mlflow_tracking_uri is None:
        #     run_id = None
        # else:
        #     run_id = _get_mlflow_run_id(mlflow_tracking_uri, experiment, False, version_name)
        experiment._current_version = version_spec
        experiment._experiment_dir = None
        if load_dataloader:
            experiment._dataloader = version_spec.dataloader()
    finally:
        os.chdir(cwd)
    return experiment  # , experiment_dir, run_id
