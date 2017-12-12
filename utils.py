import numpy as np
import tensorflow as tf
import string
import random
import itertools
from itertools import product
import global_values as cnn

class VersionContainer():
  def __init__(self,
               name,
               use_all_classes,
               classes_count,
               batch_size,
               epoc_count,
               classes_offset,
               learning_rate,
               model_dir_suffix,
               hooks):
    self.name = name
    self.use_all_classes = use_all_classes
    self.classes_count = classes_count
    self.batch_size = batch_size
    self.epoc_count = epoc_count
    self.classes_offset = classes_offset
    self.learning_rate = learning_rate
    self.model_dir_suffix=model_dir_suffix
    self.hooks=hooks

class Version():
  versions = []
  def __init__(self,
               learning_rate,
               use_all_classes = None,
               classes_count = None,
               batch_size = None,
               epoc_count = None,
               classes_offset = None,
               model_dir_suffix = None,
               hooks = None
               ):
    self.versions = []
    self.learning_rate = learning_rate
    if use_all_classes is None:
      self.use_all_classes = cnn.USE_ALL_CLASSES
    else:
      self.use_all_classes = use_all_classes

    if classes_count is None:
      self.classes_count = cnn.CLASSES_COUNT
    else:
      self.classes_count = classes_count

    if batch_size is None:
      self.batch_size = cnn.BATCH_SIZE
    else:
      self.batch_size = batch_size
      
    if epoc_count is None:
      self.epoc_count = cnn.EPOC_COUNT
    else:
      self.epoc_count = epoc_count

    if classes_offset is None:
      self.classes_offset = cnn.CLASSES_OFFSET
    else:
      self.classes_offset = classes_offset

    if model_dir_suffix is None:
      self.model_dir_suffix = cnn.MODEL_DIR_SUFFIX
    else:
      self.classes_offset = model_dir_suffix

    if hooks is None:
      self.hooks = cnn.VERSION_HOOKS
    else:
      self.hooks = hooks


  def addV(self,
           name,
           use_all_classes = None,
           classes_count = None,
           batch_size = None,
           epoc_count = None,
           classes_offset = None,
           learning_rate = None,
           model_dir_suffix = None,
           hooks = None):

    if use_all_classes is None:
      use_all_classes = self.use_all_classes
    if classes_count is None:
      classes_count = self.classes_count
    if batch_size is None:
      batch_size = self.batch_size
    if epoc_count is None:
      epoc_count = self.epoc_count
    if classes_offset is None:
      classes_offset = self.classes_offset
    if learning_rate is None:
      learning_rate = self.learning_rate
    if model_dir_suffix is None:
      model_dir_suffix = self.model_dir_suffix
    if hooks is None:
      hooks = self.hooks
      
    self.versions.append(VersionContainer(name=name,
                                          use_all_classes=use_all_classes,
                                          classes_count=classes_count,
                                          batch_size=batch_size,
                                          epoc_count=epoc_count,
                                          classes_offset=classes_offset,
                                          learning_rate=learning_rate,
                                          model_dir_suffix=model_dir_suffix,
                                          hooks=hooks))
                         
                         
  def rangeOnParameter(self,
                       names,
                       use_all_classes = [],
                       classes_count= [],
                       batch_size= [],
                       epoc_count = [],
                       classes_offset = [],
                       learning_rate = [],
                       model_dir_suffix = [],
                       hooks = []):

    count=itertools.count(0,1)
    getVal = lambda var, dval: (next(count), var) if len(var) is not 0 else (None, dval)
    productWrapper = lambda args: product(*args)
    indexes={"b": getVal(use_all_classes, self.use_all_classes),
             "c": getVal(classes_count, self.classes_count),
             "d": getVal(batch_size, self.batch_size),
             "e": getVal(epoc_count, self.epoc_count),
             "f": getVal(classes_offset, self.classes_count),
             "g": getVal(learning_rate, self.learning_rate),
             "h": getVal(model_dir_suffix, self.model_dir_suffix),
             "i": getVal(hooks, self.hooks)
            }
    args = []
    args = [v[1] for k,v in indexes.items() if v[0] is not None]
    if len(names) != len(args):
      raise ValueError("lenght of names shoul be {0}, to match the number of products generated".format(len(args)))
    count=itertools.count(0,1)
    for params in productWrapper(args):
      self.addV(names[next(count)],
                indexes["b"][1] if indexes["b"][0] is None else params[indexes["b"][0]],
                indexes["c"][1] if indexes["c"][0] is None else params[indexes["c"][0]],
                indexes["d"][1] if indexes["d"][0] is None else params[indexes["d"][0]],
                indexes["e"][1] if indexes["e"][0] is None else params[indexes["e"][0]],
                indexes["f"][1] if indexes["f"][0] is None else params[indexes["f"][0]],
                indexes["g"][1] if indexes["g"][0] is None else params[indexes["g"][0]],
                indexes["h"][1] if indexes["h"][0] is None else params[indexes["h"][0]],
                indexes["i"][1] if indexes["i"][0] is None else params[indexes["i"][0]])
  def getVersion(self, version_name):
    for v in self.versions:
      if v.name == version_name:
        return v
    raise ValueError("Version name '{0}' not found".format(version_name))

class VersionLog():
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
