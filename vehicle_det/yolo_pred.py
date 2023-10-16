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
    model_file_path = model_save_dir_path  / 'train/weights' / 'best.onnx'

    model = YOLO(model_file_path.as_posix())
    return model


def process_image(model, input_image=None, args=None, **kwargs):
    fake_result = {
        'algorithm_data': {
            'is_alert': False,
            'target_count': 0,
            'target_info': []
        },
        'model_data': {'objects': []}
    }

    conf_thresh = 0.25
    iou_thresh = 0.7

    results = model(input_image, conf=conf_thresh, iou=iou_thresh)

    target_count = 0

    for result in results:
        class_names = result.names
        boxes = result.boxes
        xyxy_tensor = boxes.xyxy
        conf_tensor = boxes.conf
        cls_tensor  = boxes.cls

        for xyxy, conf, class_index in zip(xyxy_tensor, conf_tensor, cls_tensor):
            target_count += 1

            target_info = {
                'x':int(xyxy[0]),
                'y':int(xyxy[1]),
                'width':int(xyxy[2]-xyxy[0]),
                'height':int(xyxy[3]-xyxy[1]),
                'confidence':float(conf),
                'name':class_names[int(class_index)]
            }

            fake_result['algorithm_data']['target_info'].append(target_info)

            object_info = {
                'x':int(xyxy[0]),
                'y':int(xyxy[1]),
                'width':int(xyxy[2]-xyxy[0]),
                'height':int(xyxy[3]-xyxy[1]),
                'confidence':float(conf),
                'name':class_names[int(class_index)]
            }

            fake_result['model_data']['objects'].append(object_info)

    if target_count > 0:
        fake_result['algorithm_data']['is_alert'] = True
        fake_result['algorithm_data']['target_count'] = target_count

    result_str = json.dumps(fake_result, indent = 4)
    return result_str


def main():
    data_root_path = Path(r'/home/data')

    anno_file_paths = list(data_root_path.rglob('*.xml'))
    anno_file_paths = anno_file_paths[:2]

    model = init()

    for anno_file_path in anno_file_paths:
        xml_tree = ET.parse(anno_file_path.as_posix())
        root = xml_tree.getroot()

        filename = root.find('filename').text
        image_file_path = anno_file_path.parent / filename
        img = cv2.imread(image_file_path.as_posix())
        result_str = process_image(model, img)

        logger.info(r'image_file_path: {}'.format(image_file_path))

        logger.info(r'result_str: {}'.format(result_str))

if __name__ == '__main__':
    main()
