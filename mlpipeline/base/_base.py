from mlpipeline.utils._utils import (Versions,
                                     log,
                                     copy_related_files,
                                     _collect_related_files)
from mlpipeline.entities import ExecutionModeKeys
import logging


class ExperimentABC():
    '''
    each experiment script should have a global variable `EXPERIMENT` set with an instance of this class.
    Refer to the methods for more details.
    '''
    versions = None
    allow_delete_experiment_dir = False
    reset_steps = False
    summery = None
    __related_files = []

    def __init__(self, versions, allow_delete_experiment_dir=False, reset_steps=False):
        '''
        keyword arguments:
        version -- An instance of `Version`, which will be used to obtain the versios of the experiment to execute.
        allow_delete_experiment_dir -- if true, the directory specified by `experiment_dir`
                                       passed to the `pre_execution_hook` will be cleared, essentially removing any
                                       saved information of the experiment. This can be used when the experiment
                                       training needs to be reset.
        reset_steps -- (DEPRECATING) if true, the number of steps that has elapsed will be ignored and number of steps will be
                       calculated as if no training as occurred. if false, the steps will be calucated by deducting
                       the value returned by `get_trained_step_count`.
        '''
        if isinstance(versions, Versions):
            self.versions = versions
        else:
            raise ValueError("versions should be an instance of `Versions` class, but recived: {0}".format(type(versions)))
        self.allow_delete_experiment_dir = allow_delete_experiment_dir
        self.reset_steps = reset_steps
        self._current_version = None
        self._experiment_dir = None
        self._dataloader = None

    def _set_dataloader(self, value):
        self.log("`dataloader` is being set, which is not recommended.", level=logging.WARN)
        self._dataloader = value

    def _set_experiment_dir(self, value):
        self.log("`experiment_dir` is being set, which is not recommended.", level=logging.WARN)
        self._experiment_dir = value

    def _set_current_version(self, value):
        self.log("`current_version` is being set, which is not recommended.", level=logging.WARN)
        self._current_version = value

    def _get_dataloader(self):
        return self._dataloader

    def _get_experiment_dir(self):
        return self._experiment_dir

    def _get_current_version(self):
        return self._current_version

    current_version = property(
        fget=_get_current_version,
        fset=_set_current_version,
        doc="The current version being executed. Will set by the pipeline")
    experiment_dir = property(
        fget=_get_experiment_dir,
        fset=_set_experiment_dir,
        doc="The experiment directory for the current run. Will be set by the mlpipeline")
    dataloader = property(
        fget=_get_dataloader,
        fset=_set_dataloader,
        doc="The dataloader set for a given execution of a experiment. Will be set by the mlpipeline")

    # TODO: Does the exec_mode have to be here?
    def pre_execution_hook(self, mode=ExecutionModeKeys.TEST):
        '''
        Before execution, this method will be called to set the version obtained from `self.versions`. Also `experiment_dir` will provide the destination to save the experiment in as specified in the config file. The exec_mode will be passed, with on of the keys as specified in `ExecutionModeKeys`. This function can be used to define the experiment's hyperparameters based on the information of the version being executed duering an iteration. This method will be once called before `train_loop` for each version. 
        '''
        raise NotImplementedError()

    def post_execution_hook(self, mode=ExecutionModeKeys.TEST):
        raise NotImplementedError()

    def setup_model(self):
        '''
        This function will be called before the 'export_model' and 'pre_execution_hook'. It expects to set the 'self.model' of the Experiment class here. This will be callaed before the train_loop function and the 'export_model' methods. The current version spec will passed to this method.
'''
        raise NotImplementedError()

    def train_loop(self, input_fn):
        '''
This will be called when the experiment is entering the traning phase. Ideally, what needs to happen in this function is to use the `input_fn` and execute the training loop for a given number of steps which will be passed through `steps`. The input_fn passed here will be the object returned by the `get_train_input` method of the dataloader. In addition, other functionalities can be included here as well, such as saving the experiment parameters during training, etc. Th return value of the method will be logged. The current version spec will passed to this method.
'''
        raise NotImplementedError()

    def evaluate_loop(self, input_fn):
        '''
This will be called when the experiment is entering the testing phase following the training phase. Ideally, what needs to happen in this function is to use the input_fn to obtain the inputs and execute the evaluation loop for a given number of steps. The input function passed here will be the object returned by the `get_train_input` and `get_test_input` methods of the dataloader. In addition to that other functionalities can be included here as well, such as saving the experiment parameters, producing additional statistics etc. the return value of the method will be logged. The current version spec will passed to this method.
'''
        raise NotImplementedError()

    def export_model(self):
        '''
        This method is called when a model is called with the export settings. Either by setting the respecitve command line argument or passing the export parameter in the versions.
'''
        raise NotImplementedError()

    def get_trained_step_count(self):
        '''
This function must return either `None` or a positive integer. The is used to determine how many steps have been completed and assess the number of steps the training should take. This is delegated to the `Experiment` as the process of determining the number is library specific.
'''
        raise NotImplementedError()

    def clean_experiment_dir(self, experiment_dir):
        '''
This function will be called when a experiment needs to be reset and the directory `experiment_dir` needs to be cleared as well.
'''
        raise NotImplementedError()

    def add_to_summery(self, content):
        '''
This function can be used to set the summery of the experiment, which will be added to the output when the output is generated by the pipeline.
try to include the relevent information you would want to refer to when assessing the output.
'''
        if self.summery is None:
            self.summery = ""
        self.summery += "\t\t{0}\n".format(content)

    def log(self, message, log_to_file=False, agent=None, **kargs):
        '''
        This Function can be used to log details from within the experiment
    '''
        if agent is not None:
            message = "{}: {}".format(agent, message)
        log(message, agent="Expriment", log_to_file=log_to_file, **kargs)

    copy_related_files = copy_related_files
    _collect_related_files = _collect_related_files


class DataLoaderABC():
    summery = None

    def __init__(self, **kargs):
        pass

    #TODO: remove this method? as each version will be given it's own dataloader....
    #     def set_classes(self, use_all_classes, classes_count):
    #       '''
    # This function will be called before the execution of a specific verion of a experiment. This function can be used to modify the data provided by dataloader based in the needs of the version of the experiment being executed. 
    # '''
    #       raise NotImplementedError
    def get_train_input(self, mode=ExecutionModeKeys.TRAIN, **kargs):
        '''
        This function returns an object which will be passed to the `Experiment.train_loop` when executing the training function of the experiment, the same function will be used for evaluation following training using `Experiment.evaluate_loop` . The the object returned by this function would depend on the how the return function will be used in the experiment. (eg: for Tensorflow models the returnn value can be a function object, for pyTorch it can be a Dataset object. In both cases the output of this function will be providing the data used for training)
'''
        raise NotImplementedError()

    def get_test_input(self, **kargs):
        '''
    This function returns an object which will be used to execute the evaluataion following training using `Experiment.evaluate_loop`. The the object returned by this function would depend on the how the return function will be used in the experiment. (eg: for Tensorflow models the returnn value can be a function object, for pyTorch it can be a Dataset object.  In both cases the output of this function will be providing the data used for evaluation)
'''
        raise NotImplementedError()

    def get_dataloader_summery(self, **kargs):
        '''
This function will be called to log a summery of the dataloader when logging the results of a experiment
    '''
        raise NotImplementedError()

    def get_train_sample_count(self):
        '''
returns the number of datapoints being used as the training dataset. This will be used to assess the number of epochs during training and evaluating.
'''
        raise NotImplementedError()

    def get_test_sample_count(self):
        '''
returns the number of datapoints being used as the testing dataset. This will be used to assess the number of epochs during training and evaluating.
'''
        raise NotImplementedError()

    def add_to_summery(self, content):
        '''
This function can be used to set the summery of the dataloader, which will be added to the output when the output is generated by the pipeline.
'''
        if self.summery is None:
            self.summery = ""
        self.summery += "\t\t{0}\n".format(content)

    def log(self, message, log_to_file=False, agent=None, **kargs):
        '''
        This Function can be used to log details from within the dataloader
    '''
        if agent is not None:
            message = "{}: {}".format(agent, message)
        log(message, agent="DataLoader", log_to_file=log_to_file, **kargs)
