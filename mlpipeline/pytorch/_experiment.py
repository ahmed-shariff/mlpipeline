import os
import torch
from mlpipeline.base import ExperimentABC
from mlpipeline.entities import ExecutionModeKeys


class BaseTorchExperimentABC(ExperimentABC):
    def __init__(self, versions, allow_delete_experiment_dir=False, **args):
        super().__init__(versions, allow_delete_experiment_dir, **args)
        self.model = None
        self.topk_k = None
        self.logging_iteration = None
        self.criterion = None
        self.optimizer = None
        self.checkpoint_saving_per_epoch = None
        self.use_cuda = None
        self.save_history_checkpoints_count = None

    def setup_model(self):
        self.history_file_name = "{}/model_params{}.tch".format(self.experiment_dir.rstrip("/"), "{}")
        self.file_name = self.history_file_name.format(0)

    def pre_execution_hook(self, mode=ExecutionModeKeys.TEST):
        print("Version spec: ", self.current_version)
        self.current_version = self.current_version
        self.logging_iteration = 10
        self.save_history_checkpoints_count = 10
        if os.path.isfile(self.file_name):
            self.log("Loading parameters from: {}".format(self.file_name))
            self.load_history_checkpoint(self.file_name)
        else:
            self.epochs_params = 0
            self.log("No checkpoint")

    def get_trained_step_count(self):
        ret_val = self.epochs_params
        self.log("epochs_trained: {}".format(ret_val))
        return ret_val

    def save_checkpoint(self, epoch):
        directory = os.path.dirname(self.file_name)
        if not os.path.exists(directory):
            os.makedirs(directory)

        if self.save_history_checkpoints_count is not None:
            if self.save_history_checkpoints_count < 1:
                raise ValueError("save_history_checkpoints_count should be 1 or higher. "
                                 "Else set it to None to completely disable this feature.")
            for history_idx in range(self.save_history_checkpoints_count - 1, -1, -1):
                history_file_name = self.history_file_name.format(history_idx)
                if os.path.exists(history_file_name):
                    os.replace(history_file_name, self.history_file_name.format(history_idx + 1))
            self.log("History checkpoints: {}".format(self.save_history_checkpoints_count))
        torch.save({
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': None if self.optimizer is None else self.optimizer.state_dict(),
            'validation': self.dataloader.datasets.validation_dataset,
            'lr_scheduler': None if self.lr_scheduler is None else self.lr_scheduler.state_dict()
        }, self.file_name)
        self.log("Saved checkpoint for epoch: {} at {}".format(epoch + 1, self.file_name))

    def load_history_checkpoint(self, checkpoint_file_name, load_optimizer=True, export_mode=False):
        self.log("Loading: {}".format(checkpoint_file_name), log_to_file=True)
        checkpoint = torch.load(checkpoint_file_name)
        self.epochs_params = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['state_dict'])
        if export_mode:
            return

        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if checkpoint['lr_scheduler'] is not None:
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        # if checkpoint['validation'] is not None:
        #     self.dataloader.set_validation_set(checkpoint['validation'])

    def get_ancient_checkpoint_file_name(self, epoch_from_last=None):
        if epoch_from_last is None:
            epoch_from_last = self.save_history_checkpoints_count
        elif epoch_from_last == 0:
            history_file_name = self.history_file_name.format(0)
            if os.path.exists(history_file_name):
                return history_file_name
        elif epoch_from_last > self.save_history_checkpoints_count:
            raise ValueError("`epoch_from_last` should be less than or equal "
                             "`self.save_history_checkpoints_count`.")

        if self.save_history_checkpoints_count < 1:
            raise ValueError("save_history_checkpoints_count should be 1 or higher. "
                             "Else set it to None to completely disable this feature.")
        for history_idx in range(epoch_from_last, 0, -1):
            history_file_name = self.history_file_name.format(history_idx)
            if os.path.exists(history_file_name):
                return history_file_name
