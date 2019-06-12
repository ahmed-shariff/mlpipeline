import string
import random
import itertools
import logging
import sys
import os
import re
import statistics
import shutil
from easydict import EasyDict
from inspect import getsourcefile
from itertools import product
from datetime import datetime

try:
    import mlflow
    use_mlflow = True
except ImportError:
    use_mlflow = False

LOGGER = None

class ExperimentModeKeys():
    '''
    Enum class that defines the keys to use to specify the execution mode of the experiment
'''
    RUN = 'Run'
    TEST = 'Test'
    EXPORT = 'Export'


class ExecutionModeKeys():
    '''
    Enum class that defines the keys to use to specify the execution mode the pipeline is currently at.
'''
    TRAIN = 'Train'
    TEST = 'Test'

class console_colors():
    RESET = "\033[0m"
    BOLD = "\033[1m"
    BLACK_FG = "\033[30m"
    RED_FG = "\033[31m"
    GREEN_FG = "\033[32m"
    YELLOW_FG = "\033[33m"
    BLUE_FG = "\033[34m"
    MEGENTA_FG = "\033[35m"
    CYAN_FG = "\033[36m"
    WHITE_FG = "\033[37m"
    BLACK_BG = "\033[40m"
    RED_BG = "\033[41m"
    GREEN_BG = "\033[42m"
    YELLOW_BG = "\033[43m"
    BLUE_BG = "\033[44m"
    MEGENTA_BG = "\033[45m"
    CYAN_BG = "\033[46m"
    WHITE_BG = "\033[47m"
  
class version_parameters():
    '''
    Enum class that defines eums for some of the parameters used in versions
'''
    NAME = "name"
    DATALOADER = "dataloader"
    BATCH_SIZE = "batch_size"
    EPOC_COUNT = "epoc_count"
    LEARNING_RATE = "learning_rate"
    EXPERIMENT_DIR_SUFFIX = "experiment_dir_suffix"
    ORDER = "order"
  
    #the rest are not needed for experiment is general, just mine 
    HOOKS = "hooks"
    CLASSES_COUNT = "classes_count"
    CLASSES_OFFSET = "classes_offset"
    USE_ALL_CLASSES = "use_all_classes"

        
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
	    modifier_1 = console_colors.BOLD,
	    modifier_2 = console_colors.GREEN_FG)

    @classmethod
    def log_mode_train(cls):
        log("Mode: {}{}".format(console_colors.RED_FG, log_special_tokens.MODE_RUNNING),
	    modifier_1 = console_colors.BOLD,
	    modifier_2 = console_colors.GREEN_FG)

    @classmethod
    def log_mode_exporting(cls):
        log("Mode: {}{}".format(console_colors.YELLOW_FG, log_special_tokens.MODE_EXPORTING),
	    modifier_1 = console_colors.BOLD,
	    modifier_2 = console_colors.MEGENTA_FG)
        
class Versions():
    '''
    The class that holds the paramter versions.
    Also prvodes helper functions to define and add new parameter versions.
    '''
    def __init__(self,
                 dataloader,
                 batch_size,
                 epoc_count,
                 **kwargs):
        self._default_values = EasyDict(dataloader = dataloader,
                                        batch_size = batch_size,
                                        epoc_count = epoc_count,
                                        **kwargs)
        self._order_index = 0
        self._versions = {}

    def add_version(self,
                    name,
                    dataloader = None,
                    batch_size = None,
                    epoc_count = None,
                    experiment_dir_suffix = None,
                    order = None,
                    custom_paramters = {},
                    **kwargs):
        v = EasyDict()
        v.update(kwargs)
        v.update(custom_paramters)
        v.dataloader = self._default_values.dataloader if dataloader is None else dataloader
        v.batch_size = self._default_values.batch_size if batch_size is None else batch_size
        v.epoc_count = self._default_values.epoc_count if epoc_count is None else epoc_count
        if order is None:
            v.order = self._order_index
            self._order_index += 1
        else:
            v.order = order
        v.experiment_dir_suffix = experiment_dir_suffix
        self._versions[name] = v

    def get_version(self, version_name):
        '''
        Returns the version with name `version_name`. 
        The return value is a EasyDict object which represents the version
        '''
        try:
            return self._versions[version_name]
        except KeyError:
            raise ValueError("Version name '{0}' not found".format(version_name))

    def rangeOnParameters(self,
                          names=None,
                          combining_parameters = [],
                          parameters = {}):
        '''
        **Experimental**
        Allows to deifine versions by providing a range(i.e. list) of values. The names of the paramters for which range is provided should be procided by combination_parmters. The values should be provided through the paramteres dictionary. The dictionaries keys are the same as that used for the versions as well as the combining_parameters parameter. Combinations of the values of the paramters specified in combining_parameters taken from the paramters dict will be used to generate versions. Parameters in the parameters dict which are not given in combining_paramters, will be used for all the combinations produced. Prameters not specified in paramteres dict will use the default values defined.

Example:
rangeOnParameters(combining_paramters = [version_parameters.LEARNING_RATE, 'experiment_specific_param1'],
                  paramters = {version_parameters.LEARNING_RATE = [0.005, 0.001], 
                               'experiment_specific_param1' = [1,2],
                               version_parameters.BATCH_SIZE = 100,
                               'experiment_specific_param2' = 0.1}
The combinations by the above call would be:
    {version_parameters.LEARNING_RATE = 0.005, 
     'experiment_specific_param1' = 1,
     version_parameters.BATCH_SIZE = 100,
     'experiment_specific_param2' = 0.1},
    {version_parameters.LEARNING_RATE = 0.005, 
     'experiment_specific_param1' = 2,
     version_parameters.BATCH_SIZE = 100,
     'experiment_specific_param2' = 0.1},
    {version_parameters.LEARNING_RATE = 0.001, 
     'experiment_specific_param1' = 1,
     version_parameters.BATCH_SIZE = 100,
     'experiment_specific_param2' = 0.1},
    {version_parameters.LEARNING_RATE = 0.001, 
     'experiment_specific_param1' = 2,
     version_parameters.BATCH_SIZE = 100,
     'experiment_specific_param2' = 0.1}
'''
        for key in combining_parameters:
            if not isinstance(parameters[key], list):
                parameters[key] = [parameters[key]]

        products = [parameters[parameter] if isinstance(parameters[parameter], list) else [parameters[parameter]] for paramter in combining_parameters]

        if names is None:
            names = [_genName() for _ in products]
        elif len(products) != len(names):
            raise ValueError("length of names shoul be {0}, to match the number of products generated".format(len(products)))
        
        for idx, combination in enumerate(product(*products)):
            self.add_version(names[idx])
            parameters_temp = parameters.copy()
            for idx, parameter in combining_parameters:
                parameters_temp[parameter] = combination[idx]
                
            version = self.get_version(names[idx])
            for k,v in parameters_temp.items():
                version[k] = v

    def get_versions(self):
        '''
        Returns the list of versions (which are each an EasyDict object) 
        sorted by the ORDER parameter of each version.
        '''
        return sorted(self._versions.items(), key=lambda x:x[1][version_parameters.ORDER])

    def get_version_names(self):
        '''
        Returns the list of name of the versions.
        '''
        return list(self._versions.keys())

    def filter_versions(self,*, blacklist_versions = None, whitelist_versions = None):
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
    #list of version names
    executed_versions=[]
  
    executing_version=None
    executing_v_time=0.0
    EXECUTED = 0
    EXECUTING = 1
    NOT_EXECUTED = 2
    def __init__(self):
        self.executed_versions=[]
        self.executing_version=None
        self.executing_v_time=0.0

    def executed(self, version):
        if version is self.executing_version:
            return self.EXECUTING
        else:
            #for n, t in self.exectued_versions:
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
        self.executed_versions=[]
        self.executing_version=None
        self.executing_v_time=0.0
        
def set_logger(experiment_mode = ExperimentModeKeys.TEST, no_log = True, log_file = None):
    global LOGGER
    global use_mlflow
    formatter = logging.Formatter(fmt= "%(asctime)s:{0}{1}%(levelname)s:{2}%(name)s{3}- %(message)s" \
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
    use_mlflow = experiment_mode != ExperimentModeKeys.TEST
    LOGGER.NO_LOG = no_log
    LOGGER.LOG_FILE = log_file
    return LOGGER

def _genName():
    return ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(5))

def log(message, level = logging.INFO, log_to_file=True, agent=None, modifier_1=None, modifier_2=None):
    # if level is not logging.INFO and level is not logging.ERROR:
    #   raise AttributeError("level cannot be other than logging.INFO or logging.ERROR, coz i am lazy to get others in here")
    # assert any(special_token in message for special_token in
    #            log_special_tokens.values()), \
    #            "`message` cannot contain special token (check utils.log_special_tokens)"

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
        log("'set_logger' not called. Setting up Logger with default settings. To override, call 'set_logger' before any calls to 'log'", level = logging.WARN, modifier_1 = console_colors.RED_FG)
    LOGGER.log(level, message)
    #EXPERIMENT_MODE and NO_LOG will be set in the pipline subprocess script
    if LOGGER.EXPERIMENT_MODE != ExperimentModeKeys.TEST and not LOGGER.NO_LOG and log_to_file:
        with open(LOGGER.LOG_FILE, 'a', encoding="utf-8") as log_file:
            level = ["INFO" if level is logging.INFO else "ERROR"]
            time = datetime.now().isoformat()
            cleaned_message = re.sub("\[[0-9;m]*", "", message.translate(str.maketrans({"\x1b":None})))
            log_file.write("[{0}]::{1} - {2}\n".format(time, level[0], cleaned_message))

def add_script_dir_to_PATH(current_dir = None):
    if current_dir is None:
        current_dir = os.path.dirname(getsourcefile(lambda:0))
    if current_dir not in sys.path:
        sys.path = [current_dir] + sys.path

    log("Added dir `{}` to PYTHOAPATH. New PYTHONPATH: {}".format(current_dir, sys.path))

def _collect_related_files(experiment, root, additional_files = []):
    assert isinstance(additional_files, list)
    modules_list = additional_files
    root = os.path.abspath(root)
    for module in sys.modules.values():
        try:
            file_name =  os.path.abspath(module.__file__)
            if root in file_name:
                modules_list.append(file_name)
            else:
                pass
        except:
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
        if LOGGER.EXPERIMENT_MODE == ExperimentModeKeys.TEST:
            log("Not copying in TEST mode: file - {}".format(file))
        else:
            shutil.copy(file, dst_dir)
            log("\tCopied {}".format(file))
            if use_mlflow:
                mlflow.log_artifact(file)
    
class Metric():
    def __init__(self,  track_average_epoc_count = 1):
        self.count = 0
        self.value = 0
        # self.global_count = 0
        # self.global_value = 0
        self.epoc_count = 0
        self.epoc_value = 0
        self.track_average_epoc_count = track_average_epoc_count
        if self.track_average_epoc_count < 1:
            raise ValueError("`track_average_count` should be more than or equal to 0")
        self.track_value_list = []
        #print(type(self.value))

    def update(self, value, count = 1):
        #value = value.item()
        if not isinstance(value, int) and not isinstance(value, float):
            #print(value, value.shape)
            raise Exception("Value should be int or float, but got {}".format(type(value)))
        self.count += count
        self.value += value
        self.epoc_count += count
        self.epoc_value += value
        # try:
        #     #print(value.data[0], self.value.data[0], type(self.value), type(value))
        # except:
        #     pass
        
    def reset(self):
        self.count = 0
        self.value = 0

    def reset_epoc(self):
        if len(self.track_value_list) == self.track_average_epoc_count:
            try:
                self.track_value_list = self.track_value_list[1:] + [self.epoc_value/self.epoc_count]
            except ZeroDivisionError:
                self.track_value_list = self.track_value_list[1:] + [0]
        else:
            try:
                self.track_value_list.append(self.epoc_value/self.epoc_count)
            except ZeroDivisionError:
                self.track_value_list.append(0)

        self.epoc_count = 0
        self.epoc_value = 0
        
    def avg(self):
        try:
            return self.value/self.count
        except ZeroDivisionError:
            return 0

    def avg_epoc(self):
        try:
            return self.epoc_value/self.epoc_count
        except ZeroDivisionError:
            return 0

    def get_tracking_average(self):
        if len(self.track_value_list) < self.track_average_epoc_count/2:
            return 0
        try:
            return statistics.mean(self.track_value_list)#sum(self.track_value_list)/len(self.track_value_list)
        except ZeroDivisionError:
            return 0

    def get_tracking_delta(self):
        if len(self.track_value_list) == self.track_average_epoc_count:
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
    
    def __init__(self, metrics = None, track_average_epoc_count = 1, **kwargs):
        metrics_dict = {}
        if metrics is not None:
            if not isinstance(metrics, list):
                raise TypeError("`metrics` must be a list")

            if isinstance(metrics[0], dict):
                for metrics_set in metrics:
                    for metric in metrics_set["metrics"]:
                        metric_value = Metric(track_average_epoc_count = track_average_epoc_count)
                        try:
                            metric_value.track_average_epoc_count = metrics_set['track_average_epoc_count']
                        except KeyError:
                            pass
                        metrics_dict[metric] = metric_value
            else:
                for metric in metrics:
                    #setattr(self, metric, Metric(track_average_epoc_count = track_average_epoc_count))
                    metrics_dict[metric] = Metric(track_average_epoc_count = track_average_epoc_count)


            for k,v in metrics_dict.items():
                setattr(self, k, v)
            
            for k in self.__class__.__dict__.keys():
                if not (k.startswith('__') and k.endswith('__')) and not k in ('update', 'pop', 'reset', 'log_metrics', 'reset_epoc', '_get_matrics_subset'):
                    setattr(self, k, getattr(self, k))
        
    def _get_matrics_subset(self, metrics, return_named_tuples = False):
        if metrics is None:
            if return_named_tuples:
                return self.items()
            else:
                return [v for k,v in self.items()]
        else:
            if return_named_tuples:
                return [(k,v) for k,v in self.items() if k in metrics]
            else:
                return [v for k,v in self.items() if k in metrics]

    def reset(self, metrics = None):
        for metric in self._get_matrics_subset(metrics):
            metric.reset()

    def log_metrics(self, metrics = None, log_to_file = True, complete_epoc = False, items_per_row = 3, charachters_per_row = 50, name_prefix = "", step=None):
        return_string = ""
        printable_string = ""
        row_item_count = 0
        row_char_count = 0
        for name, metric in self._get_matrics_subset(metrics, return_named_tuples = True):
            name = name_prefix + name
            if complete_epoc:
                value = metric.avg_epoc()
            else:
                value = metric.avg()
                
            s = "{}: {:.4f}    ".format(name, value)
            # EXPERIMENT_MODE is set in the pipeline subprocess script
            if use_mlflow and log_to_file and LOGGER.EXPERIMENT_MODE != ExperimentModeKeys.TEST:
                mlflow.log_metric(name, value, step=step)
            row_char_count += len(s)
            if row_char_count > charachters_per_row:
                log(message = printable_string, log_to_file = log_to_file)
                return_string += printable_string
                printable_string = s
                row_char_count = len(s)
                row_item_count = 0
            else:
                printable_string += s
                
            row_item_count +=1    
            if row_item_count % items_per_row == 0:
                log(message = printable_string, log_to_file = log_to_file)
                return_string += printable_string
                printable_string = ""
                row_char_count = 0
                row_item_count = 0
        if printable_string != "":
            log(message = printable_string, log_to_file = log_to_file)
            return_string +=printable_string
        return return_string
            
    def reset_epoc(self, metrics = None):
        for metric in self._get_matrics_subset(metrics):
            metric.reset_epoc()

#Implimented as a class with properties for clarity and safty of sanity
class _PipelineConfig():
    '''
    Used by pipeline to maintin the configurations across multiple functions
    '''

    def __init__(self):
        import mlpipeline._default_configurations as config
        self.experiments_dir = config.EXPERIMENTS_DIR
        self.output_file = config.OUTPUT_FILE
        self.history_file = config.HISTORY_FILE
        self.training_history_log_file = config.TRAINING_HISTORY_LOG_FILE
        self.no_log = config.NO_LOG
        self.executed_experiments = config.EXECUTED_EXPERIMENTS
        self.use_blacklist = config.USE_BLACKLIST
        self.listed_experiments = config.LISTED_EXPERIMENTS
        self.experiment_mode = config.EXPERIMENT_MODE
        self.logger = None
        self.cmd_mode = False

class ExperimentWrapper():
    '''
    To be used when using the programmetic interface to execute the pipeline
    '''
    def __init__(self, file_path, whitelist_versions = None, blacklist_versions = None):
        self.file_path = file_path
        if whitelist_versions is not None and blacklist_versions is not None:
            raise ValueError("Both `whitelist_versions` and `blacklist_versions` cannot be set!")
        self.whitelist_versions = whitelist_versions
        self.blacklist_versions = blacklist_versions
