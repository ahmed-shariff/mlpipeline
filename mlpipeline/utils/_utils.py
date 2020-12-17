import string
import random
import importlib.util
import logging
import sys
import os
import re
import statistics
import shutil
from easydict import EasyDict
from inspect import getsourcefile
from datetime import datetime
import mlpipeline._default_configurations as _default_config
from mlpipeline.entities import (ExecutionModeKeys,
                                 ExperimentModeKeys,
                                 console_colors,
                                 version_parameters)
import mlflow

LOGGER = None


class log_special_tokens():
    MODE_RUNNING = "RUNNING"
    MODE_TESTING = "TESTING"
    MODE_EXPORTING = "EXPORTING"
    TRAINING_COMPLETE = "Training loop complete"
    EVALUATION_COMPLETE = "Evaluation loop complete"
    SESSION_STARTED = "=====================ML-Pipeline session started"
    SESSION_ENDED = "=====================ML-Pipeline Session ended"
    EXPERIMENT_STARTED = "-----------------Experiment run started"
    EXPERIMENT_ENDED = "-----------------Experiment run ended"

    @classmethod
    def log_session_started(cls):
        log(cls.SESSION_STARTED)

    @classmethod
    def log_session_ended(cls):
        log(cls.SESSION_ENDED)

    @classmethod
    def log_experiment_started(cls):
        log(cls.EXPERIMENT_STARTED)

    @classmethod
    def log_experiment_ended(cls):
        log(cls.EXPERIMENT_ENDED)

    @classmethod
    def log_mode_test(cls):
        log("Mode: {}{}".format(console_colors.YELLOW_FG, log_special_tokens.MODE_TESTING),
            modifier_1=console_colors.BOLD,
            modifier_2=console_colors.GREEN_FG)

    @classmethod
    def log_mode_train(cls):
        log("Mode: {}{}".format(console_colors.RED_FG, log_special_tokens.MODE_RUNNING),
            modifier_1=console_colors.BOLD,
            modifier_2=console_colors.GREEN_FG)

    @classmethod
    def log_mode_exporting(cls):
        log("Mode: {}{}".format(console_colors.YELLOW_FG, log_special_tokens.MODE_EXPORTING),
            modifier_1=console_colors.BOLD,
            modifier_2=console_colors.MEGENTA_FG)


class Versions():
    '''
    The class that holds the paramter versions.
    Also prvodes helper functions to define and add new parameter versions.
    '''
    def __init__(self,
                 dataloader=None,
                 batch_size=None,
                 epoch_count=None,
                 **kwargs):
        self._default_values = EasyDict(dataloader=dataloader,
                                        batch_size=batch_size,
                                        epoch_count=epoch_count,
                                        **kwargs)
        self._order_index = 0
        self._versions = {}

    def add_version(self,
                    name,
                    dataloader=None,
                    batch_size=None,
                    epoch_count=None,
                    experiment_dir_suffix=None,
                    order=None,
                    custom_paramters={},
                    **kwargs):
        v = EasyDict()
        v.update(kwargs)
        v.update(custom_paramters)
        v.name = name
        v.dataloader = self._default_values.dataloader if dataloader is None else dataloader
        v.batch_size = self._default_values.batch_size if batch_size is None else batch_size
        v.epoch_count = self._default_values.epoch_count if epoch_count is None else epoch_count
        if order is None:
            v.order = self._order_index
            self._order_index += 1
        else:
            v.order = order
        v.experiment_dir_suffix = experiment_dir_suffix
        self._versions[name] = v

    def add_versions(self, versions):
        '''
        Add the versions from a Versions object
        '''
        try:
            self._versions.update(versions._versions)
        except AttributeError:
            log("Passed value is not a version object", 30)
        
    def get_version(self, version_name):
        '''
        Returns the version with name `version_name`.
        The return value is a EasyDict object which represents the version
        '''
        try:
            return self._versions[version_name]
        except KeyError:
            raise ValueError("Version name '{0}' not found".format(version_name))

    def get_versions(self):
        '''
        Returns the list of versions (which are each an EasyDict object)
        sorted by the ORDER parameter of each version.
        '''
        return sorted(self._versions.items(), key=lambda x: x[1][version_parameters.ORDER])

    def get_version_names(self):
        '''
        Returns the list of name of the versions.
        '''
        return list(self._versions.keys())

    def filter_versions(self, *, blacklist_versions=None, whitelist_versions=None):
        '''
        This function can be used to filter the version to be executed.
        Only one of the two parameteres should be passed. And the values passed should be an iterable.
        If blacklist_versions is passed, the versions lised will be dropped from the versions.
        If whitelist_versions is passed, the versions not listed will be dropped from the versions.
        If nither parameters are passed, no changes will be made.
        '''
        if blacklist_versions is not None and whitelist_versions is not None:
            raise ValueError("Cannot pass both `whitelist_versions` and `blacklist_versions`!")
        elif blacklist_versions is None and whitelist_versions is None:
            return

        filtered_versions = {}
        if blacklist_versions is not None:
            for version_name in self.get_version_names():
                if version_name not in blacklist_versions:
                    filtered_versions[version_name] = self.get_version(version_name)
        elif whitelist_versions is not None:
            for version_name in whitelist_versions:
                filtered_versions[version_name] = self.get_version(version_name)

        log("Versions before filter: {}".format(self.get_version_names()))
        self._versions = filtered_versions
        log("Versions after filter: {}".format(self.get_version_names()))
        

class _VersionLog():
    '''
    used to maintain experiment version information.
    '''
    # list of version names
    executed_versions = []

    executing_version = None
    executing_v_time = 0.0
    EXECUTED = 0
    EXECUTING = 1
    NOT_EXECUTED = 2

    def __init__(self):
        self.executed_versions = []
        self.executing_version = None
        self.executing_v_time = 0.0

    def executed(self, version):
        if version is self.executing_version:
            return self.EXECUTING
        else:
            # for n, t in self.exectued_versions:
            if version in self.executed_versions:
                return self.EXECUTED
            return self.NOT_EXECUTED

    def addExecutedVersion(self, version_name):
        self.executed_versions.append(version_name)

    def moveExecutingToExecuted(self):
        self.addExecutedVersion(self.executing_version)
        self.executing_version = None
        self.executing_v_time = 0.0

    def addExecutingVersion(self, version_name, train_start_time):
        self.executing_version = version_name
        self.executing_v_time = train_start_time

    def clean(self):
        self.executed_versions = []
        self.executing_version = None
        self.executing_v_time = 0.0


def set_logger(experiment_mode=ExperimentModeKeys.TEST, no_log=True, log_file=None):
    global LOGGER
    formatter = logging.Formatter(fmt="%(asctime)s:{0}{1}%(levelname)s:{2}%(name)s{3}- %(message)s"
                                  .format(console_colors.BOLD,
                                          console_colors.BLUE_FG,
                                          console_colors.GREEN_FG,
                                          console_colors.RESET),
                                  datefmt="%Y-%m-%d %H:%M:%S")

    LOGGER = logging.getLogger("mlp")
    LOGGER.handlers = []
    LOGGER.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    LOGGER.addHandler(handler)
    LOGGER.EXPERIMENT_MODE = experiment_mode
    LOGGER.NO_LOG = no_log
    LOGGER.LOG_FILE = log_file
    logging.getLogger("mlflow").handlers[0].setFormatter(formatter)
    return LOGGER


def is_no_log():
    return _is_test_mode() or LOGGER.NO_LOG


def _is_test_mode():
    return LOGGER.EXPERIMENT_MODE == ExecutionModeKeys.TEST


def _genName():
    return ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(5))


def log(message, level=logging.INFO, log_to_file=True, agent=None, modifier_1=None, modifier_2=None):
    if agent is not None:
        message = "{}{}- {}{}".format(console_colors.CYAN_FG,
                                      agent,
                                      console_colors.RESET,
                                      message)

    if modifier_1 is None and modifier_2 is None:
        reset_string = ""
    else:
        reset_string = console_colors.RESET

    if modifier_1 is None:
        modifier_1 = ""
    if modifier_2 is None:
        modifier_2 = ""

    message = "{0}{1}{2}{3}".format(modifier_1, modifier_2, message, reset_string)
    if LOGGER is None:
        set_logger()
        log("'set_logger' not called. Setting up Logger with default settings. "
            "To override, call 'set_logger' before any calls to 'log'",
            level=logging.WARN, modifier_1=console_colors.RED_FG)
    LOGGER.log(level, message)
    # EXPERIMENT_MODE and NO_LOG will be set in the pipline subprocess script
    if not is_no_log() and log_to_file:
        with open(LOGGER.LOG_FILE, 'a', encoding="utf-8") as log_file:
            level = ["INFO" if level is logging.INFO else "ERROR"]
            time = datetime.now().isoformat()
            cleaned_message = re.sub("\[[0-9;m]*", "", message.translate(str.maketrans({"\x1b": None})))
            log_file.write("[{0}]::{1} - {2}\n".format(time, level[0], cleaned_message))


def add_script_dir_to_PATH(current_dir=None):
    if current_dir is None:
        current_dir = os.path.dirname(getsourcefile(lambda: 0))
    if current_dir not in sys.path:
        sys.path = [current_dir] + sys.path

    log("Added dir `{}` to PYTHOAPATH. New PYTHONPATH: {}".format(current_dir, sys.path))


def _collect_related_files(experiment, root, additional_files=[]):
    assert isinstance(additional_files, list)
    modules_list = additional_files
    root = os.path.abspath(root)
    for module in list(sys.modules.values()).copy():
        try:
            file_name = os.path.abspath(module.__file__)
            if root in file_name:
                if os.path.exists(file_name):
                    modules_list.append(file_name)
            else:
                pass
        except Exception:
            pass
    experiment.__related_files = modules_list


def copy_related_files(experiment, dst_dir):
    try:
        os.makedirs(dst_dir)
        log("Created directories(s): {}".format(dst_dir))
    except OSError:
        pass
    assert os.path.isdir(dst_dir)
    log("Copying imported custom scripts to {}".format(dst_dir))
    for file in experiment.__related_files:
        if is_no_log():
            log("Not copying in TEST mode and NO LOG mode: file - {}".format(file))
        else:
            shutil.copy(file, dst_dir)
            log("\tCopied {}".format(file))
            mlflow.log_artifact(file)


class Metric():
    def __init__(self,  track_average_epoch_count=1):
        self.count = 0
        self.value = 0
        # self.global_count = 0
        # self.global_value = 0
        self.epoch_count = 0
        self.epoch_value = 0
        self.track_average_epoch_count = track_average_epoch_count
        if self.track_average_epoch_count < 1:
            raise ValueError("`track_average_count` should be more than or equal to 0")
        self.track_value_list = []

    def update(self, value, count=1):
        if not isinstance(value, int) and not isinstance(value, float):
            raise Exception("Value should be int or float, but got {}".format(type(value)))
        self.count += count
        self.value += value
        self.epoch_count += count
        self.epoch_value += value

    def reset(self):
        self.count = 0
        self.value = 0

    def reset_epoch(self):
        if len(self.track_value_list) == self.track_average_epoch_count:
            try:
                self.track_value_list = self.track_value_list[1:] + [self.epoch_value/self.epoch_count]
            except ZeroDivisionError:
                self.track_value_list = self.track_value_list[1:] + [0]
        else:
            try:
                self.track_value_list.append(self.epoch_value/self.epoch_count)
            except ZeroDivisionError:
                self.track_value_list.append(0)

        self.epoch_count = 0
        self.epoch_value = 0

    def avg(self):
        try:
            return self.value/self.count
        except ZeroDivisionError:
            return 0

    def avg_epoch(self):
        try:
            return self.epoch_value/self.epoch_count
        except ZeroDivisionError:
            return 0

    def get_tracking_average(self):
        if len(self.track_value_list) < self.track_average_epoch_count:
            return 0
        try:
            return statistics.mean(self.track_value_list)
        except ZeroDivisionError:
            return 0

    def get_tracking_delta(self):
        if len(self.track_value_list) == self.track_average_epoch_count:
            return sum(
                [self.track_value_list[idx + 1] -
                 self.track_value_list[idx]
                 for idx in range(len(self.track_value_list) - 1)])
        else:
            return 0

    def get_tracking_stdev(self):
        try:
            return statistics.stdev(self.track_value_list)
        except statistics.StatisticsError:
            return 0


class MetricContainer(EasyDict):
    def __setattr__(self, name, value):
        # Blocking setting new attributes may not be pythonic, just too lazy to figure out the pythonic way
        # Just that blocking here give better clarity as to what could go wrong
        if name not in self.__class__.__dict__ and not isinstance(value, Metric):
            raise TypeError("Value set must be type of `Metric`. Better yet, avoid maually setting a value.")
        super(EasyDict, self).__setattr__(name, value)
        super(EasyDict, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def __init__(self, metrics=None, track_average_epoch_count=1, **kwargs):
        metrics_dict = {}
        if metrics is not None:
            if not isinstance(metrics, list):
                raise TypeError("`metrics` must be a list")

            if isinstance(metrics[0], dict):
                for metrics_set in metrics:
                    for metric in metrics_set["metrics"]:
                        metric_value = Metric(track_average_epoch_count=track_average_epoch_count)
                        try:
                            metric_value.track_average_epoch_count = metrics_set['track_average_epoch_count']
                        except KeyError:
                            pass
                        metrics_dict[metric] = metric_value
            else:
                for metric in metrics:
                    metrics_dict[metric] = Metric(track_average_epoch_count=track_average_epoch_count)

            for k, v in metrics_dict.items():
                setattr(self, k, v)

            for k in self.__class__.__dict__.keys():
                if not (k.startswith('__') and k.endswith('__')) and\
                   k not in ('update', 'pop', 'reset', 'log_metrics', 'reset_epoch', '_get_matrics_subset'):
                    setattr(self, k, getattr(self, k))

    def _get_matrics_subset(self, metrics, return_named_tuples=False):
        if metrics is None:
            if return_named_tuples:
                return self.items()
            else:
                return [v for k, v in self.items()]
        else:
            if return_named_tuples:
                return [(k, v) for k, v in self.items() if k in metrics]
            else:
                return [v for k, v in self.items() if k in metrics]

    def reset(self, metrics=None):
        for metric in self._get_matrics_subset(metrics):
            metric.reset()

    def log_metrics(self,
                    metrics=None,
                    log_to_file=True,
                    complete_epoch=False,
                    items_per_row=3,
                    charachters_per_row=50,
                    name_prefix="",
                    step=None):
        return_string = ""
        printable_string = ""
        row_item_count = 0
        row_char_count = 0
        for name, metric in self._get_matrics_subset(metrics, return_named_tuples=True):
            name = name_prefix + name
            if complete_epoch:
                value = metric.avg_epoch()
            else:
                value = metric.avg()

            s = "{}: {:.4f}    ".format(name, value)
            # EXPERIMENT_MODE is set in the pipeline subprocess script
            if log_to_file and not is_no_log():
                mlflow.log_metric(name, value, step=step)
            row_char_count += len(s)
            if row_char_count > charachters_per_row:
                log(message=printable_string, log_to_file=log_to_file)
                return_string += printable_string
                printable_string = s
                row_char_count = len(s)
                row_item_count = 0
            else:
                printable_string += s

            row_item_count += 1
            if row_item_count % items_per_row == 0:
                log(message=printable_string, log_to_file=log_to_file)
                return_string += printable_string
                printable_string = ""
                row_char_count = 0
                row_item_count = 0
        if printable_string != "":
            log(message=printable_string, log_to_file=log_to_file)
            return_string += printable_string
        return return_string

    def reset_epoch(self, metrics=None):
        for metric in self._get_matrics_subset(metrics):
            metric.reset_epoch()


# Implimented as a class with properties for clarity and safty of sanity
class PipelineConfig():
    '''
    Used by pipeline to maintin the configurations across multiple functions
    '''
    def __init__(self,
                 experiments_dir=_default_config.EXPERIMENTS_DIR,
                 experiments_outputs_dir=_default_config.OUTPUT_DIR,
                 output_file=None,
                 history_file=None,
                 training_history_log_file=None,
                 no_log=_default_config.NO_LOG,
                 executed_experiments={},
                 use_blacklist=False,
                 listed_experiments=[],
                 experiment_mode=ExperimentModeKeys.TEST,
                 mlflow_tracking_uri='.mlruns',
                 logger=None,
                 cmd_mode=False):
        self.experiments_dir = experiments_dir
        self.experiments_outputs_dir = experiments_outputs_dir
        self.output_file = output_file
        self.history_file = history_file
        self.training_history_log_file = training_history_log_file
        self.no_log = no_log
        self.executed_experiments = executed_experiments
        self.use_blacklist = use_blacklist
        self.listed_experiments = listed_experiments
        self.experiment_mode = experiment_mode
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.logger = logger
        self.cmd_mode = cmd_mode


def _load_file_as_module(file_path):
    spec = importlib.util.spec_from_file_location(file_path.split("/")[-1], file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class iterator():
    def __init__(self, iterable, test_iterations=1):
        self.iterable = iter(iterable)
        self.test_iterations = test_iterations
        self._current_iteration = 0

    def __iter__(self):
        return self

    def __next__(self):
        if _is_test_mode() and self.test_iterations is not None:
            if self._current_iteration >= self.test_iterations:
                raise StopIteration
        self._current_iteration += 1
        return next(self.iterable)


class Datasets():
    """Class to store the datasets"""
    # pylint disable:too-many-arguments

    def __init__(self,
                 train_data_asset=None,
                 test_data_asset=None,
                 validation_data_asset=None,
                 # train_dataset_file_path=None,
                 # test_dataset_file_path=None,
                 # validation_dataset_file_path=None,
                 class_encoding=None,
                 train_data_load_function=None,
                 test_data_load_function=None,
                 validation_data_load_function=None,
                 test_size=None,
                 # use_cache=True  # Need to implementate this one?
                 validation_size=None,
                 **kwargs):
        """
        Keyword arguments:
        train_data_asset      -- The path to the file containing the train dataset
        test_data_asset       -- The path to the file containing the test dataset.
                                        If this is None, a portion of the train dataset will be allocated as
                                        the test dataset based on the `test_size`.
        validation_data_asset -- The path to the file containing the validation dataset.
                                        If this is None, a portion of the train dataset will be allocated as
                                        the validation dataset based on the `validation_size` after
                                        allocating the test dataset.
        class_encoding               -- Dict. The index to class name mapping of the dataset. Will be logged.
        train_data_load_function     -- The function that will be used the content of the files passed above.
                                        This is a callable, that takes the file path and return the dataset.
                                        The returned value should allow selecting rows using python's slicing
                                        (eg: pandas.DataFrame, python lists, numpy.array). Will be used to
                                        load the file_passed through `train_daset_file_path`,
                                        `validation_data_asset`. Also will be used to load the
                                        `test_data_asset` if `test_data_load_function` is None.
        test_data_load_function      -- Similar to `train_data_load_function`. This parameter can be used to
                                        define a seperate loading process for the test_dataset. If
                                        `test_data_asset` is not None, this callable will be used to
                                        load the file's content. Also, if this parameter is set and
                                        `test_data_asset` is None, instead of allocating a portion of
                                        the train_dataset as test_dataset, the files`train_data_asset`
                                        passed will be loaded using this callable. Note that it is the
                                        callers responsibility to ensure there are no intersections between
                                        train and test dataset when data is loaded using this parameter.
        test_size                    -- Float between 0 and 1. The portion of the train dataset to allocate
                                        as the test dataset based if `test_data_asset` not given and
                                        `test_data_load_function` is None.
        validation_size              -- Float between 0 and 1. The portion of the train dataset to allocate
                                        as the validadtion dataset based if
                                        `validation_data_asset` not given.
        """
        # for backward compatiblity
        if len(kwargs) > 0:
            import warnings
            
            warnings.warn("train_dataset_file_path, test_dataset_file_path and validadtion_dataset_file_path are being deprecated."\
                               " Use train_data_asset, test_data_asset and valiadtion_data_asset instead", DeprecationWarning)
            if train_data_asset is None and "train_dataset_file_path" in kwargs:
                train_data_asset = kwargs["train_dataset_file_path"]
            if test_data_asset is None and "test_dataset_file_path" in kwargs:
                test_data_asset = kwargs["test_dataset_file_path"]
            if validation_data_asset is None and "validation_dataset_file_path" in kwargs:
                validation_data_asset = kwargs["validation_dataset_file_path"]
        assert train_data_load_function is not None or \
            test_data_load_function is not None or \
            validation_data_load_function is not None, \
            'all data load functions canot be None'

        self._params = EasyDict({"train_data_asset": train_data_asset,
                                 "test_data_asset": test_data_asset,
                                 "validation_data_asset": validation_data_asset,
                                 "class_encoding": class_encoding,
                                 "train_data_load_function": train_data_load_function,
                                 "test_data_load_function": test_data_load_function,
                                 "validation_data_load_function": validation_data_load_function,
                                 "test_size": test_size,
                                 "validation_size": validation_size})
        self._processed_datasets = False
        

    def _process_datasets(self):
        if self._processed_datasets:
            return

        log("Processing datasets", agent="Datasets")
        if self._params.train_data_asset is not None:
            log("Not setting train_dataset", agent="Datasets")
            self._train_dataset = self._load_data(self._params.train_data_asset,
                                                  self._params.train_data_load_function)
        else:
            self._train_dataset = []

        if self._params.test_data_asset is None:
            if self._params.test_data_load_function is not None:
                self._test_dataset = self._load_data(self._params.train_data_asset,
                                                     self._params.test_data_load_function)
            elif self._params.train_data_asset is not None:
                if self._params.test_size is None:
                    log("Using default 'test_size': 0.1", agent="Datasets")
                    test_size = 0.1
                assert 0 <= self._params.test_size <= 1
                train_size = round(len(self._train_dataset) * self._params.test_size)
                self._test_dataset = self._train_dataset[:train_size]
                self._train_dataset = self._train_dataset[train_size:]
            else:
                self._test_dataset = []
        else:
            if self._params.test_size is not None:
                log("Ignoring 'test_size'", agent="Datasets")
            test_data_load_function = self._params.test_data_load_function or self._params.train_data_load_function
            self._test_dataset = self._load_data(self._params.test_data_asset,
                                                 self._params.test_data_load_function)

        if self._params.validation_data_asset is None:
            if self._params.train_data_asset is not None:
                if self._params.validation_size is None:
                    log("Using default 'validation_size': 0.1", agent="Datasets")
                    self._params.validation_size = 0.1
                assert 0 <= self._params.validation_size <= 1
                train_size = round(len(self._train_dataset) * self._params.validation_size)
                self._validation_dataset = self._train_dataset[:train_size]
                self._train_dataset = self._train_dataset[train_size:]
            else:
                self._validation_dataset = []
        else:
            if self._params.validation_size is not None:
                log("Ignoring 'validation_size'", agent="Datasets")
            validation_data_load_function = self._params.validation_data_load_function or \
                self._params.test_data_load_function or self._params.train_data_load_function
            self._validation_dataset = self._load_data(self._params.validation_data_asset,
                                                       self._params.validation_data_load_function)

        if self._params.class_encoding is not None:
            assert isinstance(self._params.class_encoding, dict)
        self.class_encoding = self._params.class_encoding
        log("Train dataset size: {}".format(len(self._train_dataset)), agent="Datasets")
        log("Test dataset size: {}".format(len(self._test_dataset)), agent="Datasets")
        log("Validation dataset size: {}".format(len(self._validation_dataset)), agent="Datasets")

        self._processed_datasets = True

    def _load_data(self,
                   data_file_path,
                   data_load_function):
        """Helper function to load the data using the provided `data_load_function`"""
        data, used_labels = data_load_function(data_file_path)
        try:
            self._used_labels.update(used_labels)
        except AttributeError:
            self._used_labels = set(used_labels)

        # Cheap way of checking if slicing is supported
        try:
            data[0:2:2]
        except Exception:
            raise Exception("Check if the object returned by 'data_load_function' supports slicing!")
        return data

    @property
    def train_dataset(self):
        """The pandas dataframe representing the training dataset"""
        self._process_datasets()
        return self._train_dataset

    @train_dataset.setter
    def train_dataset(self, value):
        self._process_datasets()
        self._train_dataset = value

    @property
    def test_dataset(self):
        """The pandas dataframe representing the training dataset"""
        self._process_datasets()
        return self._test_dataset

    @test_dataset.setter
    def test_dataset(self, value):
        self._process_datasets()
        self._test_dataset = value

    @property
    def validation_dataset(self):
        """The pandas dataframe representing the validation dataset"""
        self._process_datasets()
        return self._validation_dataset

    @validation_dataset.setter
    def validation_dataset(self, value):
        self._process_datasets()
        self._validation_dataset = value

    @property
    def used_labels(self):
        """The labels used by the dataset"""
        self._process_datasets()
        return self._used_labels
