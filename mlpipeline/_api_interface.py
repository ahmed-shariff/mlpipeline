from mlpipeline._pipeline import (_mlpipeline_main_loop, _init_pipeline)
from mlpipeline._pipeline_subprocess import _execute_exeperiment
from mlpipeline.utils import (ExperimentWrapper,
                              ExperimentModeKeys)


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
