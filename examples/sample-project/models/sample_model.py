import math
import itertools

import os
from inspect import getsourcefile

import sys
print(sys.path)

from mlpipeline.utils import add_script_dir_to_PATH
from mlpipeline.utils import ExecutionModeKeys
from mlpipeline.utils import Versions
from mlpipeline.utils import MetricContainer
from mlpipeline.helper import Model
from mlpipeline.helper import DataLoader


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
        metric_container = MetricContainer(metrics =  ['1', 'b', 'c'], track_average_epoc_count = 5)
        metric_container = MetricContainer(metrics = [{'metrics': ['a', 'b', 'c']}, {'metrics': ['2', 'd', 'e'], 'track_average_epoc_count': 10}], track_average_epoc_count = 5)
        self.log("steps: {}".format(steps))
        self.log("calling input fn")
        input_fn()
        for epoc in range(3):
            for idx in range(6):
                metric_container.a.update(idx)
                metric_container.b.update(idx*2)
                self.log("Epoc: {}   step: {}".format(epoc, idx))
                self.log("a {}".format(metric_container.a.avg()))
                self.log("b {}".format(metric_container.b.avg()))
                
                if idx%3 == 0:
                    metric_container.reset()
            metric_container.log_metrics(['a', '2'])
            metric_container.reset_epoc()
        metric_container.log_metrics()
        self.log("trained: {}".format(self.model.train()))
        self.copy_related_files("models/exports")

    def evaluate_model(self, input_fn, steps):
        self.log("steps: {}".format(steps))
        self.log("calling input fn")
        input_fn()
        metrics = MetricContainer(['a', 'b'])
        metrics.a.update(10,1)
        metrics.b.update(2,1)
        return metrics

dl = TestingDataLoader()
v = Versions(0.01, dl, 1, 10)
v.addV("version1", custom_paramters = {"hyperparameter": "a hyperparameter"})
v.addV("version2", custom_paramters = {"hyperparameter": None})
MODEL = TestingModel(versions = v)
