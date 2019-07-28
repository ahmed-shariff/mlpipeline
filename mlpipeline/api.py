import os
from mlpipeline._pipeline import (_mlpipeline_main_loop, _init_pipeline)
from mlpipeline._pipeline_subprocess import (_execute_exeperiment,
                                             _get_experiment_dir)
from mlpipeline.utils import _load_file_as_module
from mlpipeline.base import ExperimentWrapper
from mlpipeline.entities import ExperimentModeKeys

__all__ = ['mlpipeline_execute_exeperiment', 'mlpipeline_execute_pipeline', 'get_experiment', 'ExperimentWrapper']


def mlpipeline_execute_pipeline(experiments,
                                experiments_dir,
                                experiment_mode=ExperimentModeKeys.TEST,
                                no_log=False,
                                experiments_output_dir=None):
    '''
    This function can be used to execute the same operation of executimg mlpipeline, programatically.
    '''
    experiments_dir = os.path.abspath(experiments_dir) if experiments_dir is not None else None
    assert any([isinstance(e, ExperimentWrapper) for e in experiments])
    for experiment in experiments:
        experiment.file_path = os.path.relpath(os.path.abspath(experiment.file_path), experiments_dir)
    _init_pipeline(experiment_mode, experiments_dir, no_log, experiments_output_dir)
    _mlpipeline_main_loop(experiments)


def mlpipeline_execute_exeperiment(file_path,
                                   experiments_dir,
                                   experiment_mode=ExperimentModeKeys.TEST,
                                   no_log=False,
                                   whitelist_versions=None,
                                   blacklist_versions=None,
                                   experiments_output_dir=None):
    experiments_dir = os.path.abspath(experiments_dir)
    file_path = os.path.relpath(os.path.abspath(file_path), experiments_dir)
    while _execute_exeperiment(file_path,
                               experiments_dir=experiments_dir,
                               experiment_mode=experiment_mode,
                               no_log=no_log,
                               whitelist_versions=whitelist_versions,
                               blacklist_versions=blacklist_versions,
                               experiments_output_dir=experiments_output_dir):
        if experiment_mode == ExperimentModeKeys.TEST:
            break


# TODO: Need to track the root of the project, or this becomes kind of ridiculous.
def get_experiment(file_path, experiment_dir, version_name, mlflow_tracking_uri=None):
    cwd = os.getcwd()
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
                                            None)
    # if mlflow_tracking_uri is None:
    #     run_id = None
    # else:
    #     run_id = _get_mlflow_run_id(mlflow_tracking_uri, experiment, False, version_name)
    experiment._current_version = version_spec
    experiment._experiment_dir = None
    os.chdir(cwd)
    return experiment  # , experiment_dir, run_id
