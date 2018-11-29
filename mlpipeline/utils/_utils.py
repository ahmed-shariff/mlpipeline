import string
import random
import itertools
import logging
import sys
import os
import re
from easydict import EasyDict
from inspect import getsourcefile
from itertools import product
from datetime import datetime

LOGGER = None

class ModeKeys():
    '''
    Enum class that defines the keys to use in the models
    '''
    TRAIN = "Train"
    PREDICT = "Predict"
    EVALUATE = "Evaluate"


class ExecutionModeKeys():
    '''
    Enum class that defines the keys to use to specify the execution mode of the pipeline
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
    MODEL_DIR_SUFFIX = "model_dir_suffix"
    ORDER = "order"
  
    #the rest are not needed for model is general, just mine 
    HOOKS = "hooks"
    CLASSES_COUNT = "classes_count"
    CLASSES_OFFSET = "classes_offset"
    USE_ALL_CLASSES = "use_all_classes"
  
class Versions():
    '''
    The class that holds the paramter versions.
    Also prvodes helper functions to define and add new parameter versions.
    '''
    order_index = 0
    versions = {}
    versions_defaults = {}
    def __init__(self,
               learning_rate,
               dataloader,
               batch_size = None,
               epoc_count = None,
               model_dir_suffix = None,
               order = None,
               #
               hooks = None,
               #
               use_all_classes = None,
               classes_count = None,
               classes_offset = None
               ):
        self.versions = {}
        self.versions_defaults[version_parameters.LEARNING_RATE] = learning_rate
        self.versions_defaults[version_parameters.DATALOADER] = dataloader
        
        if batch_size is None:
            self.versions_defaults[version_parameters.BATCH_SIZE] = None
        else:
            self.versions_defaults[version_parameters.BATCH_SIZE] = batch_size
      
        if epoc_count is None:
            self.versions_defaults[version_parameters.EPOC_COUNT] = None
        else:
            self.versions_defaults[version_parameters.EPOC_COUNT] = epoc_count

        if model_dir_suffix is None:
            self.versions_defaults[version_parameters.MODEL_DIR_SUFFIX] = None
        else:
            self.versions_defaults[version_parameters.MODEL_DIR_SUFFIX] = model_dir_suffix
            
        self.versions_defaults[version_parameters.ORDER] = order
    #
    # if hooks is None:
    #   self.versions_defaults[version_parameters.HOOKS] = NoneHOOKS
    # else:
        self.versions_defaults[version_parameters.HOOKS] = hooks
            
            
        if use_all_classes is None:
            self.versions_defaults[version_parameters.USE_ALL_CLASSES] = None
        else:
            self.versions_defaults[version_parameters.USE_ALL_CLASSES] = use_all_classes

        if classes_offset is None:
            self.versions_defaults[version_parameters.CLASSES_OFFSET] = None
        else:
            self.versions_defaults[version_parameters.CLASSES_OFFSET] = classes_offset

        if classes_count is None:
            self.versions_defaults[version_parameters.CLASSES_COUNT] = None
        else:
            self.versions_defaults[version_parameters.CLASSES_COUNT] = classes_count

    def addV(self,
             name,
             dataloader = None,
             batch_size = None,
             epoc_count = None,
             learning_rate = None,
             model_dir_suffix = None,
             order = None,
             custom_paramters={},
             #
             hooks = None,
             use_all_classes = None,
             classes_count = None,
             classes_offset = None):

        if dataloader is None:
            dataloader = self.versions_defaults[version_parameters.DATALOADER]
        if batch_size is None:
            batch_size = self.versions_defaults[version_parameters.BATCH_SIZE]
        if epoc_count is None:
            epoc_count = self.versions_defaults[version_parameters.EPOC_COUNT]
        if learning_rate is None:
            learning_rate = self.versions_defaults[version_parameters.LEARNING_RATE]
        if model_dir_suffix is None:
            model_dir_suffix = self.versions_defaults[version_parameters.MODEL_DIR_SUFFIX]
        if order is None:
            order = self.versions_defaults[version_parameters.ORDER]
      
        
        if hooks is None:
            hooks = self.versions_defaults[version_parameters.HOOKS]
    
        if use_all_classes is None:
            use_all_classes = self.versions_defaults[version_parameters.USE_ALL_CLASSES]
        if classes_count is None:
            classes_count = self.versions_defaults[version_parameters.CLASSES_COUNT]
        if classes_offset is None:
            classes_offset = self.versions_defaults[version_parameters.CLASSES_OFFSET]

        self.versions[name] = {}
        self.versions[name][version_parameters.NAME] = name
        self.versions[name][version_parameters.DATALOADER] = dataloader
        self.versions[name][version_parameters.BATCH_SIZE] = batch_size
        self.versions[name][version_parameters.EPOC_COUNT] = epoc_count
        self.versions[name][version_parameters.LEARNING_RATE] = learning_rate
        self.versions[name][version_parameters.MODEL_DIR_SUFFIX] = model_dir_suffix
    
        self.versions[name][version_parameters.HOOKS] = hooks
    #
        self.versions[name][version_parameters.USE_ALL_CLASSES] = use_all_classes
        self.versions[name][version_parameters.CLASSES_COUNT] = classes_count
        self.versions[name][version_parameters.CLASSES_OFFSET] = classes_offset

        if order is None:
            self.versions[name][version_parameters.ORDER] = self.order_index
            self.order_index += 1
        else:
            self.versions[name][version_parameters.ORDER] = order
    
        for k,v in custom_paramters.items():
            self.versions[name][k] = v
      
    def rangeOnParameters(self,
                          names=None,
                          combining_parameters = [],
                          parameters = {}):
        '''
        Allows to deifine versions by providing a range(i.e. list) of values. The names of the paramters for which range is provided should be procided by combination_parmters. The values should be provided through the paramteres dictionary. The dictionaries keys are the same as that used for the versions as well as the combining_parameters parameter. Combinations of the values of the paramters specified in combining_parameters taken from the paramters dict will be used to generate versions. Parameters in the parameters dict which are not given in combining_paramters, will be used for all the combinations produced. Prameters not specified in paramteres dict will use the default values defined.

Example:
rangeOnParameters(combining_paramters = [version_parameters.LEARNING_RATE, 'model_specific_param1'],
                  paramters = {version_parameters.LEARNING_RATE = [0.005, 0.001], 
                               'model_specific_param1' = [1,2],
                               version_parameters.BATCH_SIZE = 100,
                               'model_specific_param2' = 0.1}
The combinations by the above call would be:
    {version_parameters.LEARNING_RATE = 0.005, 
     'model_specific_param1' = 1,
     version_parameters.BATCH_SIZE = 100,
     'model_specific_param2' = 0.1},
    {version_parameters.LEARNING_RATE = 0.005, 
     'model_specific_param1' = 2,
     version_parameters.BATCH_SIZE = 100,
     'model_specific_param2' = 0.1},
    {version_parameters.LEARNING_RATE = 0.001, 
     'model_specific_param1' = 1,
     version_parameters.BATCH_SIZE = 100,
     'model_specific_param2' = 0.1},
    {version_parameters.LEARNING_RATE = 0.001, 
     'model_specific_param1' = 2,
     version_parameters.BATCH_SIZE = 100,
     'model_specific_param2' = 0.1}
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
            self.addV(names[idx])
            parameters_temp = parameters.copy()
            for idx, parameter in combining_parameters:
                parameters_temp[parameter] = combination[idx]
            version = self.getVersion(names[idx])
            for k,v in parameters_temp.items():
                version.parameters[k] = v
          
    def getVersion(self, version_name):
        #for v in self.versions:
        #  if v.name == version_name:
        try:
            return self.versions[version_name]
        except KeyError:
            raise ValueError("Version name '{0}' not found".format(version_name))

class VersionLog():
    '''
    used to maintain model version information.
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


def set_logger(test_mode = True, no_log = True, log_file = None):
    global LOGGER
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
    LOGGER.TEST_MODE = test_mode
    LOGGER.NO_LOG = no_log
    LOGGER.LOG_FILE = log_file
    return LOGGER

def _genName():
    return ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(5))

def log(message, level = logging.INFO, log_to_file=True, modifier_1=None, modifier_2=None):
    # if level is not logging.INFO and level is not logging.ERROR:
    #   raise AttributeError("level cannot be other than logging.INFO or logging.ERROR, coz i am lazy to get others in here")
    if modifier_1 is None and modifier_2 is None:
        reset_string = ""
    else:
        reset_string = console_colors.RESET
    
    if modifier_1 is None:
        modifier_1 = ""
    if modifier_2 is None:
        modifier_2 = ""

    message = "{0}{1}{2}{3}".format(modifier_1, modifier_2, message, reset_string)
  
    LOGGER.log(level, message)
    #TEST_MODE and NO_LOG will be set in the pipline script
    if not LOGGER.TEST_MODE and not LOGGER.NO_LOG and log_to_file:
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
        if len(self.track_value_list) > self.track_average_epoc_count:
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
        if name not in self.__class__.__dict__ and not isinstance(value, Metric):
            raise TypeError("Value set must be type of `Metric`. Better yet, avoid maually setting a value.")        
        super(EasyDict, self).__setattr__(name, value)
        super(EasyDict, self).__setitem__(name, value)

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

    def log_metrics(self, metrics = None, log_to_file = True, complete_epoc = False):
        return_string = ""
        printable_string = ""
        for idx, (name, metric) in enumerate(self._get_matrics_subset(metrics, return_named_tuples = True)):
            if complete_epoc:
                printable_string += "{}: {:.4f}    ".format(name, metric.avg_epoc())
            else:
                printable_string += "{}: {:.4f}    ".format(name, metric.avg())
            if idx % 3 == 0 and idx > 0:
                log(message = printable_string, log_to_file = log_to_file)
                return_string += printable_string
                printable_string = ""
        if printable_string != "":
            log(message = printable_string, log_to_file = log_to_file)
            return_string +=printable_string
        return return_string
            
    def reset_epoc(self, metrics = None):
        for metric in self._get_matrics_subset(metrics):
            metric.reset_epoc()
