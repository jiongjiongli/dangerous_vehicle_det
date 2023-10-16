import json
from pathlib import Path
import logging
from xml.etree import ElementTree as ET
import cv2
import sys

sys.path.append('/project/train/src_repo/ultralytics')

from ultralytics.utils import LOGGER as logger
from ultralytics import YOLO


def set_logging(log_file_path):
    file_handler = logging.FileHandler(Path(log_file_path).as_posix())
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def init():
    log_file_path = r'/project/train/log/log.txt'
    set_logging(log_file_path)

    model_save_dir_path = Path('/project/train/models')
    model_file_path = model_save_dir_path  / 'train/weights' / 'best.pt'

    model = YOLO(model_file_path.as_posix())
    return model


def main():
    model = init()
    model.export(format='onnx')


if __name__ == '__main__':
    main()
