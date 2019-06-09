import subprocess
from mlpipeline._pipeline import (_execute_subprocess, _init_pipeline)
from mlpipeline._pipeline_subprocess import _execute_exeperiment
from mlpipeline.utils import (ExperimentWrapper,
                              ExperimentModeKeys)


def mlpipeline_execute_pipeline(experiments,
                                experiments_dir = None,
                                experiment_mode = ExperimentModeKeys.TEST,
                                no_log = False):
    '''
    This function can be used to execute the same operation of executimg mlpipeline, programatically.
    '''
    assert any([isinstance(e, ExperimentWrapper) for e in experiments])
    _init_pipeline(experiment_mode, experiments_dir, no_log)
    for experiment in experiments:
        _execute_subprocess(experiment.file_path,
                            experiment.whitelist_versions,
                            experiment.blacklist_versions)


def mlpipeline_execute_exeperiment(file_path,
                                   experiments_dir,
                                   experiment_mode = ExperimentModeKeys.TEST,
                                   no_log = False,
                                   whitelist_versions = None,
                                   blacklist_versions = None):
    while _execute_exeperiment(file_path,
                               experiments_dir = experiments_dir,
                               experiment_mode = experiment_mode,
                               no_log = no_log,
                               whitelist_versions = whitelist_versions,
                               blacklist_versions = blacklist_versions):
        pass
