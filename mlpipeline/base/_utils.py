from mlpipeline.base import DataLoaderABC


class DummyDataloader(DataLoaderABC):
    summery = "Not specified"

    def get_train_input(self, **kargs):
        return None

    def get_test_input(self, **kargs):
        return None

    def get_dataloader_summery(self, **kargs):
        return self.summery


class DataLoaderCallableWrapper():
    def __init__(self, dataloader_class, *args, **kwargs):
        self.dataloader_class = dataloader_class
        self.args = args
        self.kwargs = kwargs
        self._dataloader = None

    def __call__(self):
        if self._dataloader is None:
            self._dataloader = self.dataloader_class(*self.args, **self.kwargs)
        return self._dataloader
