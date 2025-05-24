import os
import json
import random
from ..dataset_info import *
import cv2
import numpy as np

def sort_list_by_criteria(obj_list):
    return sorted(
        obj_list,
        key=lambda x: (
            int(x['object_anomaly']),
            int(x['object_name'][-3:]),  # object_name
            int(x['view']),             # view
            int(x['illumination'])       # illumination
        )
    )

class M2ADSolver(object):
    CLSNAMES = CLASS_NAMES['m2ad']

    def __init__(self, version='m2ad', normal_ratio=0.6, random_seed=42):
        self.root = get_data_root(version)
        self.meta_dir = f'{self.root}/jsons'
        os.makedirs(self.meta_dir, exist_ok=True)
        self.normal_ratio = normal_ratio
        self.random_seed = random_seed

    def run(self):
        info = self.generate_meta_info()
        split_meta(info, self.meta_dir)

    def get_obj_image(self, root_dir, object_name, category, object_anomaly, cls_json):
        obj_info = []

        image_paths = os.listdir(os.path.join(root_dir, object_name))

        for image_path in image_paths:
            if object_anomaly:
                mask_path = os.path.join(root_dir, '../GT', object_name, image_path[:-8] + '_mask.png')
                seg_path = os.path.join(root_dir, '../GT', object_name, image_path[:-8] + '_seg.png')

                if os.path.exists(mask_path):
                    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if np.sum(mask) > 0:
                        image_anomaly = 1
                    else:
                        image_anomaly = 0
                    mask_path = os.path.relpath(mask_path, self.root)
                    seg_path = os.path.relpath(seg_path, self.root)

                else:
                    mask_path = ''
                    seg_path = ''
                    image_anomaly = 0

                if image_anomaly == 0:
                    detectable = ''
                else:
                    detectable = True
                    img_rel_path = os.path.relpath(os.path.join(root_dir, object_name, image_path), self.root)
                    for indx, element in enumerate(cls_json):
                        if element.get('img_path') == img_rel_path:
                            detectable = element.get('detectable', False)
                            break
                        if indx == len(cls_json) - 1:
                            print(f"Fail to load detectable value for {img_rel_path}")
            else:
                mask_path = ''
                seg_path = ''
                image_anomaly = 0
                detectable = ''

            obj_info_temp = {
                'img_path':os.path.relpath(os.path.join(root_dir, object_name, image_path), self.root),
                'view': image_path[1:4],
                'illumination': image_path[6:8],
                'object_name': object_name,
                'object_anomaly': object_anomaly,
                'image_anomaly': image_anomaly,
                'cls_name': category,
                'mask_path': mask_path,
                'seg_path': seg_path,
                'detectable': detectable,
            }

            obj_info.append(obj_info_temp)

        return obj_info

    def generate_meta_info(self):
        info = dict(train={}, test={})

        for cls_name in self.CLSNAMES:
            cls_info = dict(train={}, test={})
            cls_dir = os.path.join(self.root, cls_name)

            # load jsons
            json_path = os.path.join(self.meta_dir, 'detectable', f'{cls_name}.json')
            data_json = {'train': [], 'test': []}

            try:
                if os.path.exists(json_path):
                    with open(json_path, 'r') as f:
                        data_json = json.load(f)
            except Exception as e:
                print(f"Error loading JSON {json_path}: {e}")

            cls_json = data_json.get('train', []) + data_json.get('test', [])

            cls_normal_info = []
            # normal objs
            normal_obj_dir = os.path.join(cls_dir, 'Good')
            normal_obj_names = os.listdir(normal_obj_dir)
            normal_obj_names = sorted(normal_obj_names)
            random.seed(self.random_seed)
            random.shuffle(normal_obj_names)
            for obj_name in normal_obj_names:
                obj_info = self.get_obj_image(normal_obj_dir, obj_name, category=cls_name, object_anomaly=0, cls_json=cls_json)
                cls_normal_info.append(obj_info)

            # abnormal objs
            cls_abnormal_info = []
            abnormal_obj_dir = os.path.join(cls_dir, 'NG')
            abnormal_obj_names = os.listdir(abnormal_obj_dir)
            for obj_name in abnormal_obj_names:
                obj_info = self.get_obj_image(abnormal_obj_dir, obj_name, category=cls_name, object_anomaly=1, cls_json=cls_json)
                cls_abnormal_info.append(obj_info)

            train_info = cls_normal_info[:int(len(cls_normal_info) * self.normal_ratio)]
            test_info = cls_normal_info[int(len(cls_normal_info) * self.normal_ratio):] + cls_abnormal_info

            info['train'][cls_name] = []
            for _train_info in train_info:
                info['train'][cls_name].extend(_train_info)

            info['train'][cls_name] = sort_list_by_criteria(info['train'][cls_name])

            print(f"[{cls_name}] Save train info: {len(train_info)} objects, {len(info['train'][cls_name])} images ")

            info['test'][cls_name] = []
            for _test_info in test_info:
                info['test'][cls_name].extend(_test_info)

            info['test'][cls_name] = sort_list_by_criteria(info['test'][cls_name])
            print(f"[{cls_name}] Save test info: {len(test_info)} objects, {len(info['test'][cls_name])} images ")

            cls_info['train'] = info['train'][cls_name]
            cls_info['test'] = info['test'][cls_name]
            with open(os.path.join(self.meta_dir, f'{cls_name}.json'), 'w') as f:
                f.write(json.dumps(cls_info, indent=4) + "\n")

        with open(os.path.join(self.meta_dir, f'meta.json'), 'w') as f:
            f.write(json.dumps(info, indent=4) + "\n")

        return info

