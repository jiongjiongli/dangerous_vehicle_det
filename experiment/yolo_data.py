import sys

remove_path = '/project/train/src_repo'
if remove_path in sys.path:
    index = sys.path.index(remove_path)
    del sys.path[index]

append_path = '/project/train/src_repo/ultralytics'
if append_path not in sys.path:
    sys.path.append(append_path)

from pathlib import Path
import torch
from torch.utils.data import Dataset, IterableDataset, _DatasetKind
from torch.utils.data import dataloader, distributed
from ultralytics import YOLO as BaseYOLO
from ultralytics.utils import checks, RANK
from ultralytics.cfg import TASK2DATA


class YOLO(BaseYOLO):
    def __init__(self, model='yolov8n.pt', task=None):
        super().__init__(model=model, task=task)

    def train(self, trainer=None, **kwargs):
        self._check_is_pytorch_model()
        if self.session:  # Ultralytics HUB session
            if any(kwargs):
                LOGGER.warning('WARNING ⚠️ using HUB training arguments, ignoring local training arguments.')
            kwargs = self.session.train_args
        checks.check_pip_update_available()

        overrides = yaml_load(checks.check_yaml(kwargs['cfg'])) if kwargs.get('cfg') else self.overrides
        custom = {'data': TASK2DATA[self.task]}  # method defaults
        args = {**overrides, **custom, **kwargs, 'mode': 'train'}  # highest priority args on the right
        if args.get('resume'):
            args['resume'] = self.ckpt_path

        self.trainer = (trainer or self._smart_load('trainer'))(overrides=args, _callbacks=self.callbacks)
        if not args.get('resume'):  # manually set model only if not resuming
            self.trainer.model = self.trainer.get_model(weights=self.model if self.ckpt else None, cfg=self.model.yaml)
            self.model = self.trainer.model
        self.trainer.hub_session = self.session  # attach optional HUB session
        # self.trainer.train()
        # -> class DetectionTrainer(BaseTrainer):


class TrainerHelper:
    def __init__(self, trainer):
        self.args = trainer.args

    def get_world_size(self):
        if isinstance(self.args.device, str) and len(self.args.device):  # i.e. device='0' or device='0,1,2,3'
            world_size = len(self.args.device.split(','))
        elif isinstance(self.args.device, (tuple, list)):  # i.e. device=[0, 1, 2, 3] (multi-GPU from CLI is list)
            world_size = len(self.args.device)
        elif torch.cuda.is_available():  # i.e. device=None or device='' or device=number
            world_size = 1  # default to device 0
        else:  # i.e. device='cpu' or 'mps'
            world_size = 0

        return world_size


def main():
    repo_dir_path = Path('/project/train/src_repo')
    model_file_path = repo_dir_path / 'yolov8n.pt'
    data_root_path = Path(r'/home/data')
    dataset_config_file_path = data_root_path / 'custom_dataset.yaml'

    yolo_model = YOLO(model_file_path.as_posix())
    yolo_model.train(data=dataset_config_file_path.as_posix())
    trainer_helper = TrainerHelper(yolo_model.trainer)
    world_size = trainer_helper.get_world_size()

    assert world_size == 1, world_size

    # self._do_train(world_size)
    # ->
    # if world_size > 1:
    #     self._setup_ddp(world_size)
    # self._setup_train(world_size)

    yolo_model.trainer._setup_train(world_size)
    dataset = yolo_model.trainer.train_loader.dataset
    data_loader = yolo_model.trainer.train_loader
    data_loader._dataset_kind == _DatasetKind.Map

    # self.iterator = _MultiProcessingDataLoaderIter(self)

if __name__ == '__main__':
    main()
