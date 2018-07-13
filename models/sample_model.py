import math
import itertools

import os
from inspect import getsourcefile

from utils import add_script_dir_to_PATH
from utils import ExecutionModeKeys
from utils import Versions
from helper import Model
from helper import DataLoader

#This section is sepcially needed if the model scripts are not in the same directory from which the pipline is being executed
#add_script_dir_to_PATH(os.path.abspath(os.path.dirname(getsourcefile(lambda:0))))


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

  def pre_execution_hook(self, version, model_dir, exec_mode=ExecutionModeKeys.TEST):
    self.log("Pre execution")
    self.log("Version spec: ", version)
    self.current_version = version

  def get_current_version(self):
    return self.current_version

  def get_trained_step_count(self):
    return 10

  def train_model(self, input_fn, steps):
    self.log("steps: ", steps)
    self.log("calling input fn")
    input_fn()

  def evaluate_model(self, input_fn, steps):
    self.log("steps: ", steps)
    self.log("calling input fn")
    input_fn()

dl = TestingDataLoader()
v = Versions(0.01, dl)
v.addV("version1")
v.addV("version2")
MODEL = TestingModel(versions = v)
