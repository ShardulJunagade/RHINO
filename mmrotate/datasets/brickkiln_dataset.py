# ******** ADDED THIS CODE TO DOTA.PY ********


# from mmengine.dataset import BaseDataset
# from mmrotate.registry import DATASETS
# import glob
# import os.path as osp
# from typing import List

# @DATASETS.register_module()
# class BrickKilnDataset(BaseDataset):
#     METAINFO = {
#         'classes': ('CFCBK', 'FCBK', 'Zigzag'),
#         'palette': [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Example colors
#     }

#     def __init__(self, img_shape=(640, 640), diff_thr=100, **kwargs) -> None:
#         self.img_shape = img_shape
#         self.diff_thr = diff_thr
#         super().__init__(**kwargs)

#     def load_data_list(self) -> List[dict]:
#         """Load annotations from an annotation file named as ``self.ann_file``
#         Returns:
#             List[dict]: A list of annotation.
#         """  # noqa: E501
#         cls_map = {c: i
#                    for i, c in enumerate(self.metainfo['classes'])
#                    }  # in mmdet v2.0 label is 0-based
#         data_list = []
#         if self.ann_file == '':
#             img_files = glob.glob(
#                 osp.join(self.data_prefix['img_path'], '*.png'))
#             for img_path in img_files:
#                 data_info = {}
#                 data_info['img_path'] = img_path
#                 img_name = osp.split(img_path)[1]
#                 data_info['file_name'] = img_name
#                 img_id = img_name[:-4]
#                 data_info['img_id'] = img_id
#                 data_info['height'] = self.img_shape[0]
#                 data_info['width'] = self.img_shape[1]

#                 instance = dict(bbox=[], bbox_label=[], ignore_flag=0)
#                 data_info['instances'] = [instance]
#                 data_list.append(data_info)

#             return data_list
#         else:
#             txt_files = glob.glob(osp.join(self.ann_file, '*.txt'))
#             if len(txt_files) == 0:
#                 raise ValueError('There is no txt file in '
#                                  f'{self.ann_file}')
#             for txt_file in txt_files:
#                 data_info = {}
#                 img_id = osp.split(txt_file)[1][:-4]
#                 data_info['img_id'] = img_id
#                 img_name = img_id + '.png'
#                 data_info['file_name'] = img_name
#                 data_info['img_path'] = osp.join(self.data_prefix['img_path'],
#                                                  img_name)
#                 data_info['height'] = self.img_shape[0]
#                 data_info['width'] = self.img_shape[1]

#                 instances = []
#                 with open(txt_file) as f:
#                     s = f.readlines()
#                     for si in s:
#                         instance = {}
#                         bbox_info = si.split()
#                         instance['bbox'] = [[float(i) for i in bbox_info[:8]]]
#                         cls_name = bbox_info[8]
#                         instance['bbox_label'] = cls_map[cls_name]
#                         difficulty = int(bbox_info[9])
#                         if difficulty > self.diff_thr:
#                             instance['ignore_flag'] = 1
#                         else:
#                             instance['ignore_flag'] = 0
#                         instances.append(instance)
#                 data_info['instances'] = instances
#                 data_list.append(data_info)

#             return data_list

#     def filter_data(self) -> List[dict]:
#         """Filter annotations according to filter_cfg.

#         Returns:
#             List[dict]: Filtered results.
#         """
#         if self.test_mode:
#             return self.data_list

#         filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False) \
#             if self.filter_cfg is not None else False

#         valid_data_infos = []
#         for i, data_info in enumerate(self.data_list):
#             if filter_empty_gt and len(data_info['instances']) == 0:
#                 continue
#             valid_data_infos.append(data_info)

#         return valid_data_infos

#     def get_cat_ids(self, idx: int) -> List[int]:
#         """Get DOTA category ids by index.

#         Args:
#             idx (int): Index of data.
#         Returns:
#             List[int]: All categories in the image of specified index.
#         """

#         instances = self.get_data_info(idx)['instances']
#         return [instance['bbox_label'] for instance in instances]