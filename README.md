**Note that the this documentation is a tad bit outdated. Will be updating as soon as I can**
# ml-pipeline
I use this pipeline to simplify my life when working on ML projects. 

## Installation
This can be installed using pip
```
pip install mlpipeline
```

## Usage (tl;dr version)
1. Extend `mlpipeline.helper.Experiment` and `mlpipeline.helper.Dataloader` to suit your needs.
2. Define the versions using the interface provided by `mlpipeline.utils.Versions`.
   - Version parameters that must be defined: 
	 - `mlpipeline.utils.version_parameters.NAME`
	 - `mlpipeline.utils.version_parameters.DATALOADER`
	 - `mlpipeline.utils.version_parameters.BATCH_SIZE`
	 - `mlpipeline.utils.version_parameters.EPOC_COUNT`
3. Place the script(s) containing above in a specified directory.
4. Add the directory to `mlp.config`
5. Add the name of the script to the `experiments.config`
6. (optional) Add the name of the script to the `experiments_test.config`
7. (optional) Run the experiment in test mode to ensure the safety of your sanity.

``` bash
mlpipeline
```
8. Execute the pipeline

``` bash
mlpipeline -r -u
```
9. Anything saved to the `experiment_dir` passed through the `mlpipeline.utils.Experiment.train_loop` and `mlpipeline.utils.Experiment.evaluate_loop` will be available to access. The output and logs can be found in `outputs/log-<hostname>` and `outputs/output-<hostname>` files relative to the directory in 3. above.

## Usage (Long version)
### Experiment scripts
The experiment script is a python script that contain a global variable `EXPERIMENT` which holds an `mlpipeline.helper.Experiment` object. Ideally, one would extend the `mlpipeline.helper.Experiment` class and implement it's methods to perform the intended tasks (Refer documentation in [mlpipeline.helper](mlpipeline.helper.py) for more details). 

Place experiment scripts in a separate folder. Note that this folder can be anywhere in your system. Add the path to the folder in which the code is placed in the `mlp.config` file.
The directory structure recommended to use in this case would be as follows:
```
/<project>
  /experiment
    <experimentscripts>
  mlp.config
  experiments.config
  experiments_test.config
```

The `mlpipeline` will be executed from the <projects> directory.

For example: A sample experiment can be seen in [examples/sample-project/experiments/sample_experiment.py](examples/sample-project/experiments/sample_experiment.py). The default [mlp.config](mlp.config) file has points to the [experiments](experiments) folder. The [examples/sample-project/](examples/sample-project/) is a sample directory structure for a project.


### Versions (I should choose a better term for this)
* `mlpipeline.utils.version_parameters.NAME`: This is a string used to keep track of the training and history and this name will be appended to the logs and outputs. This parameters must be set for each version.
* `mlpipeline.utils.version_parameters.DATALOADER`: An `mlpipeline.helper.DataLoader` object. Simply put, it is a wrapper for a dataset. You'll have extend the `mlpipeline.helper.DataLoader` class to fit your needs. This object will be used by the pipeline to infer details about a training process, such as the number of steps (Refer documentation in [mlpipeline.helper](mlpipeline.helper.py) for more details). As of the current version of the pipeline, this parameter is mandatory.
* `mlpipeline.utils.version_parameters.EXPERIMENT_DIR_SUFFIX`: Each version of the experiment that's completed the training loop will be allocated a directory which can be used to save outputs (e.g. checkpoint files). When a experiment is being trained with a different set of versions if `allow_delete_experiment_dir` is set to `True` in the `EXPERIMENT`, the directory will be cleared as defined in `mlpipeline.helper.Experiment.clean_experiment_dir` (Note that the behaviour of this function is not implemented by default to avoid a disaster). Some times you may want to have different directories to for each version of the experiment, in such a case, pass a string to this parameter, which will be appended to the directory name.
* `mlpipeline.utils.version_parameters.BATCH_SIZE`: The batch size used in the experiment's training loop. As of the current version of the pipeline, this parameter is mandatory.
* `mlpipeline.utils.version_parameters.EPOC_COUNT`: The number of epocs that will be used. As of the current version of the pipeline, this parameter is mandatory.
* `mlpipeline.utils.version_parameters.ORDER`: This is set to ensure the versions are loaded in the order they are defined. This value can be passed to a version to override this behaviour.

### Executing experiments
You can have any number of experiments in the `experiments` folder. Add the names of the scripts to the `experiments.config` file. If the `use_blacklist` is false, only the scripts whose names are under `[WHITELISTED_EXPERIMENTS]` will be executed. if it is set to true all scripts except the ones under the `[BLACKLISTED_EXPERIMENTS]` will be executed. Note that experiments can be added or removed (assuming it has not been executed) to the execution queue while the pipeline is running. That is after each experiment is executed, the pipeline will re-load the config file.

You can execute the pipeline by running the python script:

``` bash
python pipeline.py
```
Note: this will run the pipeline in test mode (Read [The two modes](#the-two-modes) for more information)
#### Outputs
The outputs and logs will be saved in files in a folder named `outputs` in the `experiments` folder. There are two files the user would want to keep track of (note that the \<hostname\> is the host name of the system on which the pipeline is being executed):
- `log-<hostname>`: This file contains the logs
- `output-<hostname>`: This file contains the output results of each "incarnation" of a experiment.

Note that the other files are used by the pipeline to keep track of training sessions previously launched.

#### The two modes
The pipeline can be executed in two modes: **test mode** and **execution mode**. When you are developing a experiment, you'd want to use the test mode. The pipeline when executed without any additional arguments will be executed in the test mode. Note that the test mode uses it's own config file `experiments_test.config`, that functions similar to the `experiments.config` file. To execute in execution mode, pass `-r` to the above command:

``` bash
python pipeline.py -r
```
Differences between test mode and execution mode (default behaviour):

Test mode | Execution mode
----------|---------------
Uses `experiments_test.config` | Uses `experiments.config`
The experiment directory is a temporary directory which will be cleared each time the experiment is executed | The experiment directory is a directory defined by the name of the experiment and versions `EXPERIMENT_DIR_SUFFIX`
If an exception is raised, the pipeline will halt is execution by raising the exception to the top level | Any exception raised will not stop the pipeline, the error will be logged and the pipeline will continue process with other versions and experiments
No results or logs will be recorded in the output files | All logs and outputs will be recorded in the output files

## Extra
I use an experiment log to maintain the experiments, which kinda ties into how I use the pipeline. For more info on that: [Experiment log - How I keep track of my ML experiments](https://ahmed-shariff.github.io/2018/06/11/Experiment-log/)

The practices I have grown to follow are described in this post: [Practices I follow with the machine learning pipeline](https://ahmed-shariff.github.io/2018/08/01/mlp_file_structure)

Other projects that address similar problems (I'd be trying to combine them in the future iterations of the pipeline):
- [DVC](https://github.com/iterative/dvc)
- [sacred](https://github.com/IDSIA/sacred)
- [MLflow](https://github.com/databricks/mlflow)


