import torch.utils.data
from mlpipeline.base import DataLoaderABC
from mlpipeline.entities import ExecutionModeKeys


class BaseTorchDataLoader(DataLoaderABC):
    """Base DataLoader implementation for using with pytoch"""

    def __init__(self,
                 datasets,
                 pytorch_dataset_factory,
                 batch_size=1,
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
