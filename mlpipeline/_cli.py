import click
import sys
from multiprocessing import Queue

from mlpipeline.utils import (PipelineConfig,
                              log_special_tokens)

from mlpipeline.entities import ExperimentModeKeys
from mlpipeline.base import ExperimentWrapper
from mlpipeline._pipeline import (_init_pipeline,
                                  _mlpipeline_main_loop)
from mlpipeline._pipeline_subprocess import _execute_exeperiment_process


def get_run():
    @click.pass_obj
    def run(config):
        config.experiment_mode = ExperimentModeKeys.RUN
        return config
    return run


def get_export():
    @click.pass_obj
    def export(config):
        config.experiment_mode = ExperimentModeKeys.EXPORT
        return config
    return export


def get_test(all_option_enable=False, all_option_help=''):
    if all_option_enable:
        all_option = click.option('--all', is_flag=True, help=all_option_help)
    else:
        all_option = click.option('--all', is_flag=True, help=all_option_help, hidden=True)
    @all_option
    @click.pass_obj
    def test(config, all):
        config.experiment_mode = ExperimentModeKeys.TEST
        config.all_option = all
        return config
    return test


@click.group()
@click.option('--no-log', is_flag=True,
              help='If set, the `run` mode will not write anything, only print the logs in STDOUT '
              'Irrelevent in `test` and `export` mode, this is ignored')
@click.option('--experiments', multiple=True,
              help='The list of experiment scripts to execute. If not provided, will use either '
              '`experiments.config` or `experiments_test.config` to determine the '
              'experiment scripts to execute')
@click.option('--experiments-dir',
              help='The root directory in which the experiment scripts reside. '
              'This defaults to `experiments`.')
# @click.option('--experiments_output_dir',
#               help='The directory in which the log files, mlflow`s default directory and respective '
#               'experiments directories are saved. Defaults to `<experiments_dir>/outputs`')
@click.option('--mlflow_tracking_uri',
              help='The tracking uri to be used by mlflow. If not set will attempt to get value '
              'from `mlp.config`. If not refaults to `mlruns`')
@click.pass_context
def cli(ctx, no_log, experiments, experiments_dir, mlflow_tracking_uri):  # , experiments_output_dir):
    config = PipelineConfig()
    config.no_log = no_log
    config.listed_experiments = experiments
    config.experiments_dir = experiments_dir
    # config.output_file = experiments_output_dir
    config.mlflow_tracking_uri = mlflow_tracking_uri
    ctx.obj = config


@cli.resultcallback()
@click.pass_context
def process_pipeline(ctx, config, no_log, experiments, experiments_dir, mlflow_tracking_uri):  # , experiments_output_dir):
    if ctx.invoked_subcommand == 'single':
        return
    _init_pipeline(config.experiment_mode,
                   experiments_dir=experiments_dir,
                   no_log=config.no_log,
                   # experiments_output_dir=experiments_output_dir,
                   mlflow_tracking_uri=mlflow_tracking_uri,
                   _cmd_mode=True)
    log_special_tokens.log_session_started()
    if len(experiments) == 0:
        experiments = None
    else:
        experiments = [ExperimentWrapper(experiment) for experiment in experiments]
    _mlpipeline_main_loop(experiments)
    log_special_tokens.log_session_ended()


@cli.group(short_help='Execute one experiment')
@click.option('--whitelist-versions', multiple=True,
              help='A list of version to be executed from the experiment.')
@click.option('--blacklist-versions', multiple=True,
              help='A list of version to be excluded from the experiment.')
@click.option('-b', is_flag=True, hidden=True)
@click.pass_context
def single(ctx, whitelist_versions, blacklist_versions, b):
    '''
    Same as mlpipeline, but this mode takes only a single experiment as an input and
    allows to filter the versions being executed in the experiments.
    Note only one of whitelist_versions and blacklist_versions can be set.
    '''
    pass


@single.resultcallback()
def process_pipeline_single(config, whitelist_versions, blacklist_versions, b, **kwargs):
    if len(config.listed_experiments) != 1:
        click.echo("Error: Only one experiment should be passed with single mode", err=True)
        return
    if config.experiments_dir is None:
        click.echo("Error: `single` mode requires the `experiments_dir` option to be passed")
        return

    try:
        all_option = config.all_option
        if all_option:
            quque = Queue()
            p = _execute_exeperiment_process(file_path=config.listed_experiments[0],
                                             experiments_dir=config.experiments_dir,
                                             experiment_mode=config.experiment_mode,
                                             no_log=config.no_log,
                                             whitelist_versions=whitelist_versions,
                                             blacklist_versions=blacklist_versions,
                                             _cmd_mode=True,
                                             multiprocessing_version_quque=quque)
            p.start()
            p.join()
            whitelist_versions = quque.get()
            whitelist_versions_all = set(whitelist_versions)
            executed_versions = []
        else:
            executed_versions = None
    except AttributeError:
        executed_versions = None
    exitcode = 0
    if whitelist_versions is not None and len(whitelist_versions) == 0:
        whitelist_versions = None
    if blacklist_versions is not None and len(blacklist_versions) == 0:
        blacklist_versions = None
    while exitcode != 3 and exitcode != 1:
        if executed_versions is not None:
            try:
                whitelist_versions = whitelist_versions_all.difference(executed_versions).pop()
            except KeyError:
                break
            blacklist_versions = None
            executed_versions.append(whitelist_versions)
            whitelist_versions = [whitelist_versions]
        p = _execute_exeperiment_process(file_path=config.listed_experiments[0],
                                         experiments_dir=config.experiments_dir,
                                         experiment_mode=config.experiment_mode,
                                         no_log=config.no_log,
                                         whitelist_versions=whitelist_versions,
                                         blacklist_versions=blacklist_versions,
                                         _cmd_mode=True,
                                         mlflow_tracking_uri=config.mlflow_tracking_uri)
        p.start()
        p.join()
        exitcode = p.exitcode
        # _execute_exeperiment(file_path=config.listed_experiments[0],
        #                      experiments_dir=config.experiments_dir,
        #                      experiment_mode=config.experiment_mode,
        #                      no_log=config.no_log,
        #                      whitelist_versions=whitelist_versions,
        #                      blacklist_versions=blacklist_versions,
        #                      mlflow_tracking_uri=config.mlflow_tracking_uri,
        #                      _cmd_mode=True):
        # experiments_output_dir=experiments_output_dir):
        if b or config.experiment_mode == ExperimentModeKeys.TEST:
            if executed_versions is None:
                break
    sys.exit(exitcode)


single.command(short_help='Execute the experiments')(get_run())
single.command(short_help='Export models from all listed versions of all listed experiemnts')(get_export())
single.command(short_help='Execute the experiments in testing mode')(
    get_test(True,
             'If true, will run the pipeline for all versions'))
cli.command(short_help='Execute the experiments')(get_run())
cli.command(short_help='Export models from all listed versions of all listed experiemnts')(get_export())
cli.command(short_help='Execute the experiments in testing mode')(get_test())
