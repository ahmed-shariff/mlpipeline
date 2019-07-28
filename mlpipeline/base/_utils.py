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


class ExperimentWrapper():
    '''
    To be used when using the programmetic interface to execute the pipeline
    '''
    def __init__(self, file_path, whitelist_versions=None, blacklist_versions=None):
        self.file_path = file_path
        if whitelist_versions and blacklist_versions:
            raise ValueError("Both `whitelist_versions` and `blacklist_versions` cannot be set!")
        self.whitelist_versions = whitelist_versions
        self.blacklist_versions = blacklist_versions
