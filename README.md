# ml-pipeline
I use this pipeline to simplify my life when working on ML projects. 

## Installation
This can be installed using pip
```
pip install mlpipeline
```

## Usage (tl;dr version)
1. Extend `mlpipeline.helper.Model` and `mlpipeline.helper.Dataloader` to suit your needs.
2. Define the versions using the interface provided by `mlpipeline.utils.Versions`.
   - Version parameters that must be defined: 
	 - `mlpipeline.utils.version_parameters.NAME`
	 - `mlpipeline.utils.version_parameters.DATALOADER`
	 - `mlpipeline.utils.version_parameters.BATCH_SIZE`
	 - `mlpipeline.utils.version_parameters.EPOC_COUNT`
3. Place the script(s) containing above in a specified directory.
4. Add the directory to `mlp.config`
5. Add the name of the script to the `models.config`
6. (optional) Add the name of the script to the `models_test.config`
7. (optional) Run the model in test mode to ensure the safety of your sanity.

``` bash
mlpipeline
```
8. Execute the pipeline

``` bash
mlpipeline -r -u
```
9. Anything saved to the `model_dir` passed through the `mlpipeline.utils.Model.train_model` and `mlpipeline.utils.Model.evaluate_model` will be available to access. The output and logs can be found in `outputs/log-<hostname>` and `outputs/output-<hostname>` files relative to the directory in 3. above.

## Usage (Long version)
### Model scripts
The model script is a python script that contain a global variable `MODEL` which holds an `mlpipeline.helper.Model` object. Ideally, one would extend the `mlpipeline.helper.Model` class and implement it's methods to perform the intended tasks (Refer documentation in [mlpipeline.helper](mlpipeline.helper.py) for more details). 

Place model scripts in a separate folder. Note that this folder can be anywhere in your system. Add the path to the folder in which the code is placed in the `mlp.config` file.
The directory structure recommended to use in this case would be as follows:
```
/<project>
  /model
    <modelscripts>
  mlp.config
  models.config
  models_test.config
```

The `mlpipeline` will be executed from the <projects> directory.

For example: A sample model can be seen in [examples/sample-project/models/sample_model.py](examples/sample-project/models/sample_model.py). The default [mlp.config](mlp.config) file has points to the [models](models) folder. The [examples/sample-project/](examples/sample-project/) is a sample directory structure for a project.


### Versions (I should choose a better term for this)
* `mlpipeline.utils.version_parameters.NAME`: This is a string used to keep track of the training and history and this name will be appended to the logs and outputs. This parameters must be set for each version.
* `mlpipeline.utils.version_parameters.DATALOADER`: An `mlpipeline.helper.DataLoader` object. Simply put, it is a wrapper for a dataset. You'll have extend the `mlpipeline.helper.DataLoader` class to fit your needs. This object will be used by the pipeline to infer details about a training process, such as the number of steps (Refer documentation in [mlpipeline.helper](mlpipeline.helper.py) for more details). As of the current version of the pipeline, this parameter is mandatory.
* `mlpipeline.utils.version_parameters.MODEL_DIR_SUFFIX`: Each model trained will be allocated a directory which can be used to save outputs (e.g. checkpoint files). When a model is being trained with a different set of versions if `allow_delete_model_dir` is set to `True` in the `MODEL`, the directory will be cleared as defined in `mlpipeline.helper.Model.clean_model_dir` (Note that the behaviour of this function is not implemented by default to avoid a disaster). Some times you may want to have different directories to for each version of the model, in such a case, pass a string to this parameter, which will be appended to the directory name.
* `mlpipeline.utils.version_parameters.BATCH_SIZE`: The batch size used in the model training. As of the current version of the pipeline, this parameter is mandatory.
* `mlpipeline.utils.version_parameters.EPOC_COUNT`: The number of epocs that will be used. As of the current version of the pipeline, this parameter is mandatory.
* `mlpipeline.utils.version_parameters.ORDER`: This is set to ensure the versions are loaded in the order they are defined. This value can be passed to a version to override this behaviour.

### Executing models
You can have any number of models in the `models` folder. Add the names of the scripts to the `models.config` file. If the `use_blacklist` is false, only the scripts whose names are under `[WHITELISTED_MODELS]` will be executed. if it is set to true all scripts except the ones under the `[BLACKLISTED_MODELS]` will be executed. Note that models can be added or removed (assuming it has not been executed) to the execution queue while the pipeline is running. That is after each model is executed, the pipeline will re-load the config file.

You can execute the pipeline by running the python script:

``` bash
python pipeline.py
```
Note: this will run the pipeline in test mode (Read [The two modes](#the-two-modes) for more information)
#### Outputs
The outputs and logs will be saved in files in a folder named `outputs` in the `models` folder. There are two files the user would want to keep track of (note that the \<hostname\> is the host name of the system on which the pipeline is being executed):
- `log-<hostname>`: This file contains the logs
- `output-<hostname>`: This file contains the output results of each "incarnation" of a model.

Note that the other files are used by the pipeline to keep track of training sessions previously launched.

#### The two modes
The pipeline can be executed in two modes: **test mode** and **execution mode**. When you are developing a model, you'd want to use the test mode. The pipeline when executed without any additional arguments will be executed in the test mode. Note that the test mode uses it's own config file `models_test.config`, that functions similar to the `models.config` file. To execute in execution mode, pass `-r` to the above command:

``` bash
python pipeline.py -r
```
Differences between test mode and execution mode (default behaviour):

Test mode | Execution mode
----------|---------------
Uses `models_test.config` | Uses `models.config`
The model directory is a temporary directory which will be cleared each time the model is executed | The model directory is a directory defined by the name of the model and versions `MODEL_DIR_SUFFIX`
If an exception is raised, the pipeline will halt is execution by raising the exception to the top level | Any exception raised will not stop the pipeline, the error will be logged and the pipeline will continue process with other versions and models
No results or logs will be recorded in the output files | All logs and outputs will be recorded in the output files

## Extra
I use an experiment log to maintain the experiments, which kinda ties into how I use the pipeline. For more info on that: [Experiment log - How I keep track of my ML experiments](https://ahmed-shariff.github.io/2018/06/11/Experiment-log/)

The practices I have grown to follow are described in this post: [Practices I follow with the machine learning pipeline](https://ahmed-shariff.github.io/2018/08/01/mlp_file_structure)

Other projects that address similar problems (I'd be trying to combine them in the future iterations of the pipeline):
- [DVC](https://github.com/iterative/dvc)
- [sacred](https://github.com/IDSIA/sacred)
- [MLflow](https://github.com/databricks/mlflow)


