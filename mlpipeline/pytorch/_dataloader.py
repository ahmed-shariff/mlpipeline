import torch
import json
from mlpipeline.base import DataLoaderABC
from mlpipeline.utils import (ExecutionModeKeys,
                              log)


class Datasets():
    """Class to store the datasets"""
    # pylint disable:too-many-arguments

    def __init__(self,
                 train_dataset_file_path,
                 test_dataset_file_path=None,
                 validation_dataset_file_path=None,
                 class_encoding=None,
                 train_data_load_function=lambda file_path: json.load(open(file_path, "r")),
                 test_data_load_function=None,
                 test_size=None,
                 # use_cache=True  # Need to implementate this one?
                 validation_size=None):
        """
        Keyword arguments:
        train_dataset_file_path      -- The path to the file containing the train dataset
        test_dataset_file_path       -- The path to the file containing the test dataset.
                                        If this is None, a portion of the train dataset will be allocated as
                                        the test dataset based on the `test_size`.
        validation_dataset_file_path -- The path to the file containing the validation dataset.
                                        If this is None, a portion of the train dataset will be allocated as
                                        the validation dataset based on the `validation_size` after
                                        allocating the test dataset.
        class_encoding               -- Dict. The index to class name mapping of the dataset. Will be logged.
        train_data_load_function     -- The function that will be used the content of the files passed above.
                                        This is a callable, that takes the file path and return the dataset.
                                        The returned value should allow selecting rows using python's slicing
                                        (eg: pandas.DataFrame, python lists, numpy.array). Will be used to
                                        load the file_passed through `train_daset_file_path`,
                                        `validation_dataset_file_path`. Also will be used to load the
                                        `test_dataset_file_path` if `test_data_load_function` is None.
        test_data_load_function      -- Similar to `train_data_load_function`. This parameter can be used to
                                        define a seperate loading process for the test_dataset. If
                                        `test_dataset_file_path` is not None, this callable will be used to
                                        load the file's content. Also, if this parameter is set and
                                        `test_dataset_file_path` is None, instead of allocating a portion of
                                        the train_dataset as test_dataset, the files`train_dataset_file_path`
                                        passed will be loaded using this callable. Note that it is the
                                        callers responsibility to ensure there are no intersections between
                                        train and test dataset when data is loaded using this parameter.
        test_size                    -- Float between 0 and 1. The portion of the train dataset to allocate
                                        as the test dataset based if `test_dataset_file_path` not given and
                                        `test_data_load_function` is None.
        validation_size              -- Float between 0 and 1. The portion of the train dataset to allocate
                                        as the validadtion dataset based if
                                        `validation_dataset_file_path` not given.
        """
        self._train_dataset = self._load_data(train_dataset_file_path,
                                              train_data_load_function)
        if test_dataset_file_path is None:
            if test_data_load_function is not None:
                self._test_dataset = self._load_data(train_dataset_file_path,
                                                     test_data_load_function)
            else:
                if test_size is None:
                    log("Using default 'test_size': 0.1", agent="Datasets")
                    test_size = 0.1
                assert 0 <= test_size <= 1
                train_size = round(len(self._train_dataset) * test_size)
                self._test_dataset = self._train_dataset[:train_size]
                self._train_dataset = self._train_dataset[train_size:]
        else:
            if test_size is not None:
                log("Ignoring 'test_size'", agent="Datasets")
            if test_data_load_function is None:
                self._test_dataset = self._load_data(test_dataset_file_path,
                                                     train_data_load_function)
            else:
                self._test_dataset = self._load_data(test_dataset_file_path,
                                                     test_data_load_function)

        if validation_dataset_file_path is None:
            if validation_size is None:
                log("Using default 'validation_size': 0.1", agent="Datasets")
                validation_size = 0.1
            assert 0 <= validation_size <= 1
            train_size = round(len(self._train_dataset) * validation_size)
            self._validation_dataset = self._train_dataset[:train_size]
            self._train_dataset = self._train_dataset[train_size:]
        else:
            if validation_size is not None:
                log("Ignoring 'validation_size'", agent="Datasets")
            self._validation_dataset = self._load_data(validation_dataset_file_path,
                                                       train_data_load_function)

        if class_encoding is not None:
            assert isinstance(class_encoding, dict)
        self.class_encoding = class_encoding
        log("Train dataset size: {}".format(len(self._train_dataset)), agent="Datasets")
        log("Test dataset size: {}".format(len(self._test_dataset)), agent="Datasets")
        log("Validation dataset size: {}".format(len(self._validation_dataset)), agent="Datasets")

    def _load_data(self,
                   data_file_path,
                   data_load_function):
        """Helper function to load the data using the provided `data_load_function`"""
        data, used_labels = data_load_function(data_file_path)
        try:
            self.used_labels.update(used_labels)
        except AttributeError:
            self.used_labels = set(used_labels)

        # Cheap way of checking if slicing is supported
        try:
            data[0:2:2]
        except Exception:
            raise Exception("Check if the object returned by 'data_load_function' supports slicing!")
        return data

    @property
    def train_dataset(self):
        """The pandas dataframe representing the training dataset"""
        return self._train_dataset

    @train_dataset.setter
    def train_dataset(self, value):
        self._train_dataset = value

    @property
    def test_dataset(self):
        """The pandas dataframe representing the training dataset"""
        return self._test_dataset

    @test_dataset.setter
    def test_dataset(self, value):
        self._test_dataset = value

    @property
    def validation_dataset(self):
        """The pandas dataframe representing the validation dataset"""
        return self._validation_dataset

    @validation_dataset.setter
    def validation_dataset(self, value):
        self._validation_dataset = value


class BaseTorchDataLoader(DataLoaderABC):
    """Base DataLoader implementation for using with pytoch"""

    def __init__(self,
                 datasets,
                 pytorch_dataset_factory,
                 batch_size,
                 train_transforms=[],
                 test_transforms=[]):
        super().__init__()
        # assert isinstance(datasets, Datasets)
        # assert isinstance(pytorch_dataset_factory, DatasetFactory)
        self.datasets = datasets
        self.pytorch_dataset_factory = pytorch_dataset_factory
        self.batch_size = batch_size
        self.train_transforms = train_transforms
        self.test_transforms = test_transforms

    def get_dataloader_summery(self, **kargs):
        # This is set in the base class DataLoaderABC
        return self.summery

    def get_train_sample_count(self):
        return len(self.datasets.train_dataset)

    def get_test_sample_count(self):
        """Retruns the number of test samples"""
        return len(self.datasets.test_dataset)

    def get_train_input(self, mode, **kwargs):
        self.log("batch size: {}, mode: {}".format(self.batch_size, mode))

        if mode == ExecutionModeKeys.TRAIN:
            dataset_class = self.pytorch_dataset_factory.create_instance(
                current_data=self.datasets.train_dataset,
                transform=self.train_transforms,
                mode=ExecutionModeKeys.TRAIN)
        else:
            dataset_class = self.pytorch_dataset_factory.create_instance(
                current_data=self.datasets.train_dataset,
                transform=self.test_transforms,
                mode=ExecutionModeKeys.TEST)
        if mode == ExecutionModeKeys.TRAIN:
            batch_size = self.batch_size
        else:
            batch_size = 1
        dl = torch.utils.data.DataLoader(dataset_class,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         collate_fn=dataset_class.collate_fn)

        return dl

    def get_test_input(self, data=None, **kargs):
        if data is None:
            data = self.datasets.test_dataset
        dataset_class = self.pytorch_dataset_factory.create_instance(
            current_data=data,
            transform=self.test_transforms,
            mode=ExecutionModeKeys.TEST)
        dl = torch.utils.data.DataLoader(dataset_class,
                                         batch_size=1,
                                         collate_fn=dataset_class.collate_fn)
        return dl

    def get_validation_input(self, **kwargs):
        return self.get_test_input(self.datasets.validation_dataset)


class DatasetFactory():
    """
    This class will be used to create the dataset objects to be used by the different
    stages in the DataLoader.
    """
    def __init__(self, dataset_class, **args):
        self.dataset_class = dataset_class
        self.args = args

    def create_instance(self, current_data, mode, transform):
        obj = self.dataset_class(**self.args)
        obj._inject_params(current_data, mode, transform)
        obj.collate_fn = getattr(obj, 'collate_fn', torch.utils.data.dataloader.default_collate)
        return obj


class DatasetBasicABC(torch.utils.data.Dataset):
    """The Base dataset class."""

    def __init__(self, *args, **kwargs):
        self._current_data = None
        self._transform = None
        self._mode = None

    def _set_mode(self, value):
        self._mode = value

    def _get_mode(self):
        return self._mode

    def _set_transform(self, value):
        self._transform = value

    def _get_transform(self):
        return self._transform

    def _set_current_data(self, value):
        self._current_data = value

    def _get_current_data(self):
        return self._current_data

    current_data = property(fget=_get_current_data,
                            fset=_set_current_data,
                            doc="The data currently being used by the dataset")
    transform = property(fget=_get_transform,
                         fset=_set_transform,
                         doc="The transforms to be applied")
    mode = property(fget=_get_mode,
                    fset=_set_mode,
                    doc="The current mode of the experiment (ExecutionModeKeys)")

    def _inject_params(self, current_data, mode, transform=None):
        self._set_current_data(current_data)
        self._set_transform(transform)
        self._set_mode(mode)

    # def pre_process(self):
    #     raise NotImplementedError()

    def __len__(self):
        if self.current_data is None:
            return 0
        return len(self.current_data)

    def __getitem__(self, idx):
        raise NotImplementedError()
