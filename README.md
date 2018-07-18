# ml-pipeline
I use this pipeline to simplify my life when working on ML projects. 

## Usage (Long version)
### Model scripts
The model script is a python script that contain a global variable `MODEL` which holds an `mlp_helper.Model` object. Ideally, one would extend the `mlp_helper.Model` class and implement it's methods to perform the intended tasks (Refer documentation in [mlp_helper](mlp_helper.py) for more details). 

Place model scripts in a separate folder. Note that this folder can be anywhere in your system. Add the path to the folder in which the code is placed in the `mlp.config` file.

For example: A sample model can be seen in [models/sample_model.py](models/sample_model.py). The default [mlp.config](mlp.config) file has points to the [models](models) folder. 


### Versions (I should choose a better term for this)
Almost always you'd have to run the same model with different hyper-parameters. Here I group a set of values for a hyper-parameter and call them a version of a model. Use the `mlp_utils.Versions` class to define groups of parameters to be passed during each incarnation of the model. During each "incarnation", the model will be passed a dictionary which contains the values for each parameter set. For convenience I have defined a set of parameters as default and provided keys you can use in `mlp_utils.version_parameters`. While most of the parameters used will have no consequence out of the model, the parameters represented by the following keys of a version have side-effect:
* `mlp_utils.version_parameters.NAME`: This is a string used to keep track of the training and history and this name will be appended to the logs and outputs. This parameters must be set for each version.
* `mlp_utils.version_parameters.DATALOADER`: An `mlp_helper.DataLoader` object. Simply put, it is a wrapper for a dataset. You'll have extend the `mlp_helper.DataLoader` class to fit your needs. This object will be used by the pipeline to infer details about a training process, such as the number of steps (Refer documentation in [mlp_helper](mlp_helper.py) for more details). As of the current version of the pipeline, this parameter is mandatory.
* `mlp_utils.version_parameters.MODEL_DIR_SUFFIX`: Each model trained will be allocated a directory which can be used to save outputs (e.g. checkpoint files). When a model is being trained with a different set of versions if `allow_delete_model_dir` is set to `True` in the `MODEL`, the directory will be cleared as defined in `mlp_helper.Model.clean_model_dir` (Note that the behaviour of this function is not implemented by default to avoid a disaster). Some times you may want to have different directories to for each version of the model, in such a case, pass a string to this parameter, which will be appended to the directory name.
* `mlp_utils.version_parameters.BATCH_SIZE`: The batch size used in the model training. As of the current version of the pipeline, this parameter is mandatory.
* `mlp_utils.version_parameters.EPOC_COUNT`: The number of epocs that will be used. As of the current version of the pipeline, this parameter is mandatory.
* `mlp_utils.version_parameters.ORDER`: This is set to ensure the versions are loaded in the order they are defined. This value can be passed to a version to override this behaviour.

