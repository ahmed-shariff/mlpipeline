import subprocess
from mlpipeline._pipeline_subprocess import mlpipeline_execute_exeperiment
from mlpipeline.utils import ExperimentModeKeys

def train_versions_with_whitelist():
    subprocess.run(["rm", "-rf", "experiments/outputs"])
    while mlpipeline_execute_exeperiment("experiments/sample_experiment.py",
                                         "experiments",
                                         ExperimentModeKeys.RUN,
                                         whitelist_versions = ["version5"]):
        pass

def train_versions_with_blacklist():
    subprocess.run(["rm", "-rf", "experiments/outputs"])
    while mlpipeline_execute_exeperiment("experiments/sample_experiment.py",
                                         "experiments",
                                         ExperimentModeKeys.RUN,
                                         blacklist_versions = ["version2, version5"]):
        pass
    
if __name__ == "__main__":
    train_versions_with_whitelist()
    train_versions_with_blacklist()
