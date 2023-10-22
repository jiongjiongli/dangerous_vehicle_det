import json
from pathlib import Path
import random
import cv2
import pandas as pd
from xml.etree import ElementTree as ET
import logging
import yaml
import numpy as np
import torch


def set_logging(log_file_path):
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(levelname)s %(module)s %(funcName)s %(message)s',
        handlers=[logging.FileHandler(log_file_path, mode='a'),
            logging.StreamHandler()])


def set_random_seed(seed=0, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)


class DataConfigManager:
    def __init__(self, config_file_path_dict):
        self.config_file_path_dict = config_file_path_dict

    def generate(self):
        logging.info('Start parse_anno_info...')
        anno_info_list = self.parse_anno_info()

        logging.info('Start generate_yolo_configs...')
        self.generate_yolo_configs(anno_info_list,
                                   expand_data=True)

    def parse_anno_info(self):
        anno_info_list = []

        data_root_path = self.config_file_path_dict['path']
        anno_file_paths = list(data_root_path.rglob('*.xml'))

        for anno_file_path in anno_file_paths:
            xml_tree = ET.parse(anno_file_path.as_posix())
            root = xml_tree.getroot()

            filename = root.find('filename').text
            image_file_path = anno_file_path.parent / filename
            size = root.find('size')
            size_dict = self.parse_size(size)
            anno_info = {
                'image_file_path': image_file_path,
                'size': size_dict,
                'bnd_box_list': []
            }

            for object_iter in root.findall('object'):
                name = object_iter.find('name').text
                bnd_box = object_iter.find('bndbox')

                bnd_box_dict = self.parse_bnd_box(bnd_box)
                bnd_box_dict['class_name'] = name
                anno_info['bnd_box_list'].append(bnd_box_dict)

            anno_info_list.append(anno_info)

        return anno_info_list

    def analyze_anno_infos(self, anno_info_list):
        class_image_dict = {}
        class_obj_dict = {}
        image_class_objs = []
        num_gt_dict = {}

        for anno_info in anno_info_list:
            bnd_box_list = anno_info['bnd_box_list']
            class_names = []

            for bnd_box_dict in bnd_box_list:
                class_name = bnd_box_dict['class_name']
                class_obj_dict.setdefault(class_name, 0)
                class_obj_dict[class_name] += 1

                if class_name not in class_names:
                    class_names.append(class_name)

            for class_name in class_names:
                class_image_dict.setdefault(class_name, 0)
                class_image_dict[class_name] += 1

            num_gt = len(bnd_box_list)
            num_gt_dict.setdefault(num_gt, 0)
            num_gt_dict[num_gt] += 1

        logging.info(r'class_image_dict: {}'.format(class_image_dict))
        logging.info(r'class_obj_dict: {}'.format(class_obj_dict))
        logging.info(r'num_gt_dict: {}'.format(num_gt_dict))

        dataset_info_dict = {
            'class_image_dict': class_image_dict,
            'class_obj_dict': class_obj_dict,
            'num_gt_dict': num_gt_dict
        }

        return dataset_info_dict

    def serialize_dataset_info_dict(self, dataset_info_dict):
        dataset_info_file_path = self.config_file_path_dict['dataset_info']
        with open(dataset_info_file_path.as_posix(), 'w') as file_stream:
            json.dump(dataset_info_dict,
                      file_stream,
                      indent=4)

    def select_balanced_anno_infos(self,
                                   anno_info_list,
                                   class_obj_dict):
        image_class_objs = []

        for anno_info in anno_info_list:
            bnd_box_list = anno_info['bnd_box_list']
            image_class_obj_dict = {}

            for bnd_box_dict in bnd_box_list:
                class_name = bnd_box_dict['class_name']

                image_class_obj_dict.setdefault(class_name, 0)
                image_class_obj_dict[class_name] += 1

            image_class_obj_info = {
                'anno_info': anno_info,
                'image_class_obj_dict':image_class_obj_dict,
                'selected': False
            }

            image_class_objs.append(image_class_obj_info)

        class_obj_list = sorted(class_obj_dict.items(), key=lambda item: item[1], reverse=True)
        min_num_class_obj = min(class_obj_dict.values())

        selected_class_obj_dict = {}

        for class_name, _ in class_obj_list:
            selected_class_obj_dict.setdefault(class_name, 0)

            for image_class_obj_info in image_class_objs:
                if selected_class_obj_dict[class_name] >= min_num_class_obj:
                    break

                if image_class_obj_info['selected']:
                    continue

                image_class_obj_dict = image_class_obj_info['image_class_obj_dict']
                num_obj = image_class_obj_dict.get(class_name, 0)

                if num_obj == 0:
                    continue

                image_class_obj_info['selected'] = True

                for curr_class_name, curr_num_obj in image_class_obj_dict.items():
                    selected_class_obj_dict.setdefault(curr_class_name, 0)
                    selected_class_obj_dict[curr_class_name] += curr_num_obj

        selected_anno_info_list = []

        for image_class_obj_info in image_class_objs:
            if image_class_obj_info['selected']:
                anno_info = image_class_obj_info['anno_info']
                selected_anno_info_list.append(anno_info)

        return selected_anno_info_list


    def generate_yolo_configs(self,
                              anno_info_list,
                              max_num_val_data=1000,
                              max_val_percent=0.2,
                              seed=7,
                              expand_data=False):
        config_file_path_dict = self.config_file_path_dict
        class_name_dict = {}

        for anno_info in anno_info_list:
            bnd_box_list = anno_info['bnd_box_list']

            for bnd_box_dict in bnd_box_list:
                class_name = bnd_box_dict['class_name']
                class_name_dict.setdefault(class_name, 0)
                class_name_dict[class_name] += 1

        class_names = list(class_name_dict.keys())

        for anno_info in anno_info_list:
            anno_contents = []

            size_dict = anno_info['size']
            image_width = size_dict['width']
            image_height = size_dict['height']

            bnd_box_list = anno_info['bnd_box_list']

            for bnd_box_dict in bnd_box_list:
                class_name = bnd_box_dict['class_name']
                class_index = class_names.index(class_name)

                x_min = bnd_box_dict['xmin']
                y_min = bnd_box_dict['ymin']
                x_max = bnd_box_dict['xmax']
                y_max = bnd_box_dict['ymax']

                normed_center_x = (x_min + x_max) / 2 / image_width
                normed_center_y = (y_min + y_max) / 2 / image_height
                normed_bbox_width = (x_max - x_min) / image_width
                normed_bbox_height = (y_max - y_min) / image_height
                line = '{} {} {} {} {}'.format(
                    class_index,
                    normed_center_x,
                    normed_center_y,
                    normed_bbox_width,
                    normed_bbox_height)
                anno_contents.append(line)

            image_file_path = Path(anno_info['image_file_path'])
            anno_config_file_path = image_file_path.with_suffix('.txt')

            with open(anno_config_file_path, 'w') as file_stream:
                for line in anno_contents:
                    file_stream.write('{}\n'.format(line))

        set_random_seed(seed)
        random.shuffle(anno_info_list)

        num_val_data = min(max_num_val_data,
                           int(len(anno_info_list) * max_val_percent))

        candidate_train_anno_info_list = anno_info_list[:-num_val_data]
        # logging.info('Start analyze_anno_infos...')
        # dataset_info_dict = self.analyze_anno_infos(candidate_train_anno_info_list)

        # logging.info('Start select_balanced_anno_infos...')
        # selected_anno_info_list = self.select_balanced_anno_infos(
        #     candidate_train_anno_info_list,
        #     dataset_info_dict['class_obj_dict']
        # )

        if expand_data:
            set_random_seed(seed)
            logging.info('Start expand_data...')
            selected_anno_info_list = self.expand_data(candidate_train_anno_info_list)
            message = 'Expanded train data from {} to {}'.format(
                len(candidate_train_anno_info_list),
                len(selected_anno_info_list))
            logging.info(message)
        else:
            selected_anno_info_list = candidate_train_anno_info_list

        logging.info('Start analyze_selected_anno_infos...')
        selected_dataset_info_dict = self.analyze_anno_infos(selected_anno_info_list)

        logging.info('Start serialize_dataset_info_dict...')
        self.serialize_dataset_info_dict(selected_dataset_info_dict)

        anno_infos_dict = {
        'train': selected_anno_info_list,
        'val': anno_info_list[-num_val_data:]
        }

        for data_type, anno_infos in anno_infos_dict.items():
            message = r'{}: writing file {} with num_data {}'.format(
                data_type,
                config_file_path_dict[data_type],
                len(anno_infos))
            logging.info(message)

            with open(config_file_path_dict[data_type], 'w') as file_stream:
                for anno_info in anno_infos:
                    image_file_path = anno_info['image_file_path']
                    file_stream.write('{}\n'.format(image_file_path))

        dataset_config = {
            'path': config_file_path_dict['path'].as_posix(),
            'train': config_file_path_dict['train'].name,
            'val': config_file_path_dict['val'].name,
            'names': {
                class_index: class_name
                for class_index, class_name
                in enumerate(class_names)
            }
        }

        message = r'Writing dataset config file: {}'.format(
            config_file_path_dict['dataset'])
        logging.info(message)

        with open(config_file_path_dict['dataset'], 'w') as file_stream:
            yaml.dump(dataset_config, file_stream, indent=4)

    def expand_data(self, anno_info_list):
        min_num_expand = 3
        class_obj_dict = {}

        for anno_info in anno_info_list:
            bnd_box_list = anno_info['bnd_box_list']

            for bnd_box_dict in bnd_box_list:
                class_name = bnd_box_dict['class_name']
                class_obj_dict.setdefault(class_name, 0)
                class_obj_dict[class_name] += 1

        max_num_class_obj = max(class_obj_dict.values())

        num_expand_class_dict = {}

        for class_name, num_class_obj in class_obj_dict.items():
            num_expanded = max_num_class_obj // num_class_obj
            if num_expanded >= min_num_expand:
                num_expand_class_dict[class_name] = num_expanded

        message = r'num_expand_class_dict: {}'.format(num_expand_class_dict)
        logging.info(message)

        expand_list = []

        for anno_info in anno_info_list:
            num_expanded = 1
            bnd_box_list = anno_info['bnd_box_list']

            for bnd_box_dict in bnd_box_list:
                class_name = bnd_box_dict['class_name']

                curr_num_expanded = num_expand_class_dict.get(class_name, 1)
                num_expanded = max(num_expanded, curr_num_expanded)

            for _ in range(num_expanded - 1):
                expand_list.append(anno_info)

        all_anno_info_list = anno_info_list + expand_list
        random.shuffle(all_anno_info_list)
        return all_anno_info_list

    def parse_size(self, size):
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        size_dict = {
            'width': width,
            'height': height
        }

        return size_dict

    def parse_bnd_box(self, bnd_box):
        if not bnd_box:
            return None

        x_min = float(bnd_box.find('xmin').text)
        y_min = float(bnd_box.find('ymin').text)
        x_max = float(bnd_box.find('xmax').text)
        y_max = float(bnd_box.find('ymax').text)

        bnd_box_dict = {
            'xmin': x_min,
            'ymin': y_min,
            'xmax': x_max,
            'ymax': y_max
        }

        return bnd_box_dict


def main():
    data_root_path = Path(r'/home/data')

    config_file_path_dict = {
        'path': data_root_path,
        'train': data_root_path / 'train.txt',
        'val': data_root_path / 'val.txt',
        'dataset': data_root_path / 'custom_dataset.yaml',
        'dataset_info': data_root_path / 'dataset_info.json'
    }

    log_file_path = '/project/train/log/log.txt'

    set_logging(log_file_path)

    logging.info('=' * 80)
    logging.info('Start DataConfigManager')
    data_manager = DataConfigManager(config_file_path_dict)
    data_manager.generate()
    logging.info('End DataConfigManager')
    logging.info('=' * 80)

if __name__ == '__main__':
    main()
