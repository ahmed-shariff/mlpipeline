from mlpipeline._pipeline import (_mlpipeline_main_loop, _init_pipeline)
from mlpipeline._pipeline_subprocess import _execute_exeperiment
from mlpipeline.utils import (ExperimentWrapper,
                              ExperimentModeKeys)


def mlpipeline_execute_pipeline(experiments,
                                experiments_dir=None,
                                experiment_mode=ExperimentModeKeys.TEST,
                                no_log=False,
                                experiments_output_dir=None):
    '''
    This function can be used to execute the same operation of executimg mlpipeline, programatically.
    '''
    assert any([isinstance(e, ExperimentWrapper) for e in experiments])
    _init_pipeline(experiment_mode, experiments_dir, no_log, experiments_output_dir)
    _mlpipeline_main_loop(experiments)


def mlpipeline_execute_exeperiment(file_path,
                                   experiments_dir,
                                   experiment_mode=ExperimentModeKeys.TEST,
                                   no_log=False,
                                   whitelist_versions=None,
                                   blacklist_versions=None,
                                   experiments_output_dir=None):
    while _execute_exeperiment(file_path,
                               experiments_dir=experiments_dir,
                               experiment_mode=experiment_mode,
                               no_log=no_log,
                               whitelist_versions=whitelist_versions,
                               blacklist_versions=blacklist_versions,
                               experiments_output_dir=experiments_output_dir):
        if experiment_mode == ExperimentModeKeys.TEST:
            break
