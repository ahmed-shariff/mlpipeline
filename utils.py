import string
import random
import itertools
from itertools import product
import global_values as G

class ModeKeys():
  '''
Enum class that defines the keys to use in the models
'''
  TRAIN = "Train"
  PREDICT = "Predict"
  EVALUATE = "Evaluate"

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
  HOOKS = "hooks"
  #the rest are not needed for model is general, just mine 
  CLASSES_COUNT = "classes_count"
  CLASSES_OFFSET = "classes_offset"
  USE_ALL_CLASSES = "use_all_classes"
  
class VersionS():
  '''
The class that holds the paramter versions.
Also prvodes helper functions to define and add new parameter versions.
'''
  versions = {}
  versions_defaults = {}
  def __init__(self,
               learning_rate,
               dataloader,
               batch_size = None,
               epoc_count = None,
               model_dir_suffix = None,
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
      self.versions_defaults[version_parameters.BATCH_SIZE] = G.BATCH_SIZE
    else:
      self.versions_defaults[version_parameters.BATCH_SIZE] = batch_size
      
    if epoc_count is None:
      self.versions_defaults[version_parameters.EPOC_COUNT] = G.EPOC_COUNT
    else:
      self.versions_defaults[version_parameters.EPOC_COUNT] = epoc_count

    if model_dir_suffix is None:
      self.versions_defaults[version_parameters.MODEL_DIR_SUFFIX] = G.MODEL_DIR_SUFFIX
    else:
      self.versions_defaults[version_parameters.MODEL_DIR_SUFFIX] = model_dir_suffix

    if hooks is None:
      self.versions_defaults[version_parameters.HOOKS] = G.HOOKS
    else:
      self.versions_defaults[version_parameters.HOOKS] = hooks

    #
    if use_all_classes is None:
      self.versions_defaults[version_parameters.USE_ALL_CLASSES] = G.USE_ALL_CLASSES
    else:
      self.versions_defaults[version_parameters.USE_ALL_CLASSES] = use_all_classes

    if classes_offset is None:
      self.versions_defaults[version_parameters.CLASSES_OFFSET] = G.CLASSES_OFFSET
    else:
      self.versions_defaults[version_parameters.CLASSES_OFFSET] = classes_offset

    if classes_count is None:
      self.versions_defaults[version_parameters.CLASSES_COUNT] = G.CLASSES_COUNT
    else:
      self.versions_defaults[version_parameters.CLASSES_COUNT] = classes_count

  def addV(self,
           name,
           dataloader = None,
           batch_size = None,
           epoc_count = None,
           learning_rate = None,
           model_dir_suffix = None,
           hooks = None,
           custom_paramters={},
           #
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
    if hooks is None:
      hooks = self.versions_defaults[version_parameters.HOOKS]
    #
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
      names = [genName() for _ in products]
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
  exectued_versions=[]
  
  executing_version=None
  executing_v_time=0.0
  EXECUTED = 0
  EXECUTING = 1
  NOT_EXECUTED = 2
  def __init__(self):
    self.exectued_versions=[]
    self.executing_version=None
    self.executing_v_time=0.0

  def executed(self, version):
    if version is self.executing_version:
      return self.EXECUTING
    else:
      #for n, t in self.exectued_versions:
      if version in self.exectued_versions:
        return self.EXECUTED
      return self.NOT_EXECUTED

  def addExecutedVersion(self, version_name):
    self.exectued_versions.append(version_name)

  def moveExecutingToExecuted(self):
    self.addExecutedVersion(self.executing_version)
    self.executing_version = None
    self.executing_v_time = 0.0

  def addExecutingVersion(self, version_name, train_start_time):
    self.executing_version = version_name
    self.executing_v_time = train_start_time
    
  def clean(self):
    self.exectued_versions=[]
    self.executing_version=None
    self.executing_v_time=0.0
    
def genName():
  return ''.join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(5))


class Model():
  '''
each model script should have a global variable `MODEL` set with an instance of this class. Refer to the methods for more details.
'''
  versions = None
  def __init__(versions):
    self.versions = versions
    raise NotImplementedError

  def set_current_version(version, model_dir):
    '''
During execution, this method will be called to set the version obtained from `self.versions`. Also `model_dir` will provide the destination to save the model in as specified in the config file 
'''
    raise NotImplementedError
  
  def train_model(input_fn, steps):
    '''
This will be called when the model is entering the traning phase. Ideally, what needs to happen in this function is to use the input_fn to obtain the inputs and train the model for a given number of steps. In addition to that other functionalities can be included here as well, such as saving the model parameters during training, etc.
'''
    raise NotImplementedError

  def evaluate_model(input_fn, steps):
    '''
This will be called when the model is entering the testing phase. Ideally, what needs to happen in this function is to use the input_fn to obtain the inputs and test the model for a given number of steps. In addition to that other functionalities can be included here as well, such as saving the model parameters, producing additional statistics etc.
'''
    raise NotImplementedError


class DataLoader():
    def __init__(self, **kargs):
      raise NotImplementedError
    
    #TODO: remove this method? as each version will be given it's own dataloader....
#     def set_classes(self, use_all_classes, classes_count):
#       '''
# This function will be called before the execution of a specific verion of a model. This function can be used to modify the data provided by dataloader based in the needs of the version of the model being executed. 
# '''
#       raise NotImplementedError
    
    def get_train_input_fn(self, mode= ModeKeys.TRAIN, **kargs):
      '''
This function returns a function which will be called when executing the training function of the model, the same function will be used to evaluate the model following training. The return value(s) of the function returned would depend on the how the return function will be used in the model.
'''
      raise NotImplementedError

    def get_test_input_fn(self,**kargs):
      '''
This function returns a function which will be called when calling the testing function of the model. The return value(s) of the function returned would depend on the how the return function will be used in the model.
'''
      raise NotImplementedError

    def get_dataloader_summery(self, **kargs):
      '''
This function will be called to log a summery of the dataloader when logging the results of a model
'''
      raise NotImplementedError

    
