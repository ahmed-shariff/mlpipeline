import math
import itertools

import os
from inspect import getsourcefile

import sys
print(sys.path)

from mlp_utils import add_script_dir_to_PATH
from mlp_utils import ExecutionModeKeys
from mlp_utils import Versions
from mlp_helper import Model
from mlp_helper import DataLoader

class An_ML_Model():
  def __init__(self, hyperparameter="default value"):
    self.hyperparameter = hyperparameter

  def train(self):
    return "Trained using {}".format(self.hyperparameter)

class TestingDataLoader(DataLoader):
  def __init__(self):
    self.log("creating dataloader")

  def get_train_sample_count(self):
    return 1000

  def get_test_sample_count(self):
    return 1000

  def get_train_input(self, **kargs):
    return lambda:"got input form train input function"

  def get_test_input(self):
    return lambda:"got input form test input function"
  
  
class TestingModel(Model):
  def __init__(self, versions, **args):
    super().__init__(versions, **args)
    self.model = An_ML_Model()
    

  def pre_execution_hook(self, version, model_dir, exec_mode=ExecutionModeKeys.TEST):
    self.log("Pre execution")
    self.log("Version spec: {}".format(version))
    self.model.hyperparameter = version["hyperparameter"]
    self.current_version = version

  def get_current_version(self):
    return self.current_version

  def get_trained_step_count(self):
    return 10

  def train_model(self, input_fn, steps):
    self.log("steps: ", steps)
    self.log("calling input fn")
    input_fn()
    self.log("trained: {}".format(self.model.train()))

  def evaluate_model(self, input_fn, steps):
    self.log("steps: ", steps)
    self.log("calling input fn")
    input_fn()

dl = TestingDataLoader()
v = Versions(0.01, dl, 1, 1)
v.addV("version1", custom_paramters = {"hyperparameter": "a hyperparameter"})
v.addV("version2", custom_paramters = {"hyperparameter": None})
MODEL = TestingModel(versions = v)
