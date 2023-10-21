import logging
from pathlib import Path
import re
import shutil

import distutils.version
from torch.utils.tensorboard import SummaryWriter
from ultralytics.utils import USER_CONFIG_DIR, LOGGER as logger, colorstr
from ultralytics.utils.callbacks.tensorboard import callbacks as tb_callbacks
from ultralytics import YOLO


def find_model_file_path(default_model_file_path, is_best=False):
    model_save_dir_path = Path('/project/train/models')
    child_paths = list(model_save_dir_path.glob('train*'))

    model_file_name = 'best.pt' if is_best else 'last.pt'

    model_file_infos = []

    for child_path in child_paths:
        if not child_path.is_dir():
            continue

        dir_name = child_path.name
        model_file_path = child_path  / 'weights' / model_file_name

        if not model_file_path.exists():
            continue

        model_file_info = {
            'model_file_path': model_file_path,
            'dir_name': dir_name
        }

        model_file_infos.append(model_file_info)

    if not model_file_infos:
        return default_model_file_path

    best_model_file_path = None
    best_dir_index = -1

    for model_file_info in model_file_infos:
        dir_name = model_file_info['dir_name']
        match_results = re.match(r'^train([0-9]+$)', dir_name)

        if match_results:
            dir_index = int(match_results.group(1))
        else:
            dir_index = 0

        if dir_index > best_dir_index:
            best_dir_index = dir_index
            best_model_file_path = model_file_info['model_file_path']

    return best_model_file_path


def main():
    repo_dir_path = Path('/project/train/src_repo')
    default_model_file_path = repo_dir_path / 'yolov8s.pt'
    model_save_dir_path = Path('/project/train/models')
    data_root_path = Path(r'/home/data')
    dataset_config_file_path = data_root_path / 'custom_dataset.yaml'

    model_file_path = find_model_file_path(default_model_file_path,
                                           is_best=True)
    result_graphs_dir_path = Path('/project/train/result-graphs')
    font_file_names = ['Arial.ttf']
    log_file_path = Path('/project/train/log/log.txt')

    file_handler = logging.FileHandler(log_file_path.as_posix(), mode='a')
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(module)s %(funcName)s %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    result_graphs_dir_path.mkdir(parents=True, exist_ok=True)

    for font_file_name in font_file_names:
        font_file_path = repo_dir_path / font_file_name
        dest_file_path = USER_CONFIG_DIR / font_file_name
        shutil.copyfile(font_file_path, dest_file_path)

    logger.info(r'model_file_path: {}'.format(model_file_path))

    model = YOLO(model_file_path.as_posix())

    def get_value_list(start_value, end_value, step=0.05):

        value_list= []
        value = start_value

        while value <= end_value:
            value_list.append(value)
            value += step

        return value_list

    results = []

    for conf in get_value_list(0.1, 0.4):
        for iou in get_value_list(0.5, 0.9):
            logger.info(r'conf: {}, iou: {}'.format(conf, iou))
            metrics = model.val(
                data=dataset_config_file_path.as_posix(),
                conf=conf,
                iou=iou,
                project=model_save_dir_path.as_posix())

            results.append((conf, iou, metrics.results_dict))

    results.sort(key=lambda result: result[-1]['metrics/mAP50(B)'],
                 reverse=True)

    for result in results:
        logger.info(result)

if __name__ == '__main__':
    main()
