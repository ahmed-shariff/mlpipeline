from mlpipeline.entities import ExecutionModeKeys
from mlpipeline import (Versions,
                        MetricContainer,
                        iterator)
from mlpipeline.base import (ExperimentABC,
                             DataLoaderABC,
                             DataLoaderCallableWrapper)


class An_ML_Model():
    def __init__(self, hyperparameter="default value"):
        self.hyperparameter = hyperparameter

    def train(self):
        return "Trained using {}".format(self.hyperparameter)


class TestingDataLoader(DataLoaderABC):
    def __init__(self):
        self.log("creating dataloader")

    def get_train_sample_count(self):
        return 1000

    def get_test_sample_count(self):
        return 1000

    def get_train_input(self, **kargs):
        return lambda: "got input form train input function"

    def get_test_input(self):
        return lambda: "got input form test input function"


class TestingExperiment(ExperimentABC):
    def __init__(self, versions, **args):
        super().__init__(versions, **args)

    def setup_model(self, ):
        self.model = An_ML_Model()
        self.model.hyperparameter = self.current_version["hyperparameter"]

    def pre_execution_hook(self, mode=ExecutionModeKeys.TEST):
        self.log("Pre execution")
        self.log("Version spec: {}".format(self.current_version))
        self.log(f"Experiment dir: {self.experiment_dir}")
        self.log(f"Dataloader: {self.dataloader}")
        self.current_version = self.current_version

    def train_loop(self, input_fn):
        metric_container = MetricContainer(metrics=['1', 'b', 'c'], track_average_epoch_count=5)
        metric_container = MetricContainer(metrics=[{'metrics': ['a', 'b', 'c']},
                                                    {'metrics': ['2', 'd', 'e'],
                                                     'track_average_epoch_count': 10}],
                                           track_average_epoch_count=5)
        self.log("calling input fn")
        input_fn()
        for epoch in iterator(range(6)):
            for idx in iterator(range(6), 2):
                metric_container.a.update(idx)
                metric_container.b.update(idx*2)
                self.log("Epoch: {}   step: {}".format(epoch, idx))
                self.log("a {}".format(metric_container.a.avg()))
                self.log("b {}".format(metric_container.b.avg()))

                if idx % 3 == 0:
                    metric_container.reset()
            metric_container.log_metrics(['a', '2'])
            metric_container.reset_epoch()
        metric_container.log_metrics()
        self.log("trained: {}".format(self.model.train()))
        self.copy_related_files("experiments/exports")

    def evaluate_loop(self, input_fn):
        self.log("calling input fn")
        input_fn()
        metrics = MetricContainer(['a', 'b'])
        metrics.a.update(10, 1)
        metrics.b.update(2, 1)
        return metrics

    def export_model(self):
        self.log("YAY! Exported!")


dl = DataLoaderCallableWrapper(TestingDataLoader)
v = Versions(dl, 1, 10, learning_rate=0.01)
v.add_version("version1", hyperparameter="a hyperparameter")
v.add_version("version2", custom_paramters={"hyperparameter": None})
v.add_version("version3", custom_paramters={"hyperparameter": None})
v.add_version("version4", custom_paramters={"hyperparameter": None})
v.filter_versions(blacklist_versions=["version3"])
v.filter_versions(whitelist_versions=["version1", "version2"])
v.add_version("version5", custom_paramters={"hyperparameter": None})
EXPERIMENT = TestingExperiment(versions=v)
