mlpipeline
==========
This is a simple framework to organize you machine learning workflow. It automates most of the basic functionalities such as logging, a framework for testing models and gluing together different steps at different stages. This project came about as a result of me abstracting the boilerplate code and automating different parts of the process.

The aim of this simple framework is to consolidate the different sub-problems (such as loading data, model configurations, training process, evaluation process, exporting trained models, etc.) when working/researching with machine learning models. This allows the user to define how the different sub-problems are to be solved using their choice of tools and mlpipeline would handle piecing them together.

Core operations
---------------
This framework chains the different operations (sub-problems) depending on the mode it is executed in. mlpipeline currently has 3 modes:

* TEST mode: When in TEST mode, it doesn't perform any logging or tracking. It creates a temporary empty directory for the experiment to store the artifacts of an experiment in. When developing and testing the different operations, this mode can be used.
* RUN mode: In this mode, logging and tracking is performed. In addition, for each experiment run (referred to as a experiment version in mlpipeline) a directory is created for artifacts to be stored.
* EXPORT mode: In this mode, the exporting related operations will be executed instead of the training/evaluation related operations.

In addition to providing different modes, the pipeline also supports logging and recording various details. Currently mlpipeline records all logs, metrics and artifacts using a basic log files as well using `mlflow <https://github.com/databricks/mlflow>`_.

The following information is recorded:

* The scripts that were executed/imported in relation to an experiment.
* The any output results
* The metrics and parameters

Documentation
-------------
The documentation is hosted at `ReadTheDocs <https://mlpipeline.readthedocs.io/>`_.

Installing
----------
Can be installed directly using the Python Package Index using pip::
  
  pip install mlpipeline

Usage
-----
*work in progress*
