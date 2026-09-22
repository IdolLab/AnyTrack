import os
import os.path
import numpy as np
import torch
import csv
import pandas
import random
import cv2
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.admin import env_settings
from lib.train.dataset.depth_utils import get_x_frame

def get_lasher_frame(rgb_path, ir_path, gray_path, dtype='rgbrgb'):
    """读取LasHeR数据集的多模态帧（RGB + Infrared + Gray）
    
    Args:
        rgb_path: RGB图像路径
        ir_path: 红外图像路径
        gray_path: 灰度图像路径
        dtype: 数据类型，默认'rgbrgb'
    
    Returns:
        合并后的多模态图像 (H, W, 15)
    """
    # 读取RGB图像
    rgb = cv2.imread(rgb_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    
    # 读取红外图像
    ir = cv2.imread(ir_path, -1)
    ir = cv2.cvtColor(ir, cv2.COLOR_BGR2RGB)
    
    # 读取灰度图像
    if gray_path and os.path.exists(gray_path):
        gray = cv2.imread(gray_path)
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)
    else:
        # 如果gray不存在，使用RGB转灰度
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    
    # 合并三个模态 (H, W, 15)
    img = cv2.merge((rgb, gray, ir, ir))
    # img = cv2.merge((rgb, ir, ir))
    return img


class LasHeR(BaseVideoDataset):
    """ LasHeR dataset(aligned version).

    Publication:
        A Large-scale High-diversity Benchmark for RGBT Tracking
        Chenglong Li, Wanlin Xue, Yaqing Jia, Zhichen Qu, Bin Luo, Jin Tang, and Dengdi Sun
        https://arxiv.org/pdf/2104.13202.pdf

    Download dataset from https://github.com/BUGPLEASEOUT/LasHeR
    """

    def __init__(self, root=None, split='train', dtype='rgbrgb', seq_ids=None, data_fraction=None):
        """
        args:
            root - path to the LasHeR trainingset.
            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            seq_ids - List containing the ids of the videos to be used for training. Note: Only one of 'split' or 'seq_ids'
                        options can be used at the same time.
            data_fraction - Fraction of dataset to be used. The complete dataset is used by default
        """
        root = env_settings().lasher_dir if root is None else root
        assert split in ['train', 'val','all'], 'Only support all, train or val split in LasHeR, got {}'.format(split)
        super().__init__('LasHeR', root)
        self.dtype = dtype

        # all folders inside the root
        self.sequence_list = self._get_sequence_list(split)

        # seq_id is the index of the folder inside the got10k root path
        if seq_ids is None:
            seq_ids = list(range(0, len(self.sequence_list)))

        self.sequence_list = [self.sequence_list[i] for i in seq_ids]

        if data_fraction is not None:
            self.sequence_list = random.sample(self.sequence_list, int(len(self.sequence_list)*data_fraction))
        
        # 预加载文本标注（体积小）
        self.text_annotations = {}
        for seq_name in self.sequence_list:
            seq_path = os.path.join(self.root, seq_name)
            self.text_annotations[seq_name] = self._read_text_anno(seq_path)

    def get_name(self):
        return 'lasher'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return True # w=h=0 in visible.txt and infrared.txt is occlusion/oov

    def _get_sequence_list(self, split):
        ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
        file_path = os.path.join(ltr_path, 'data_specs', 'lasher_{}.txt'.format(split))
        with open(file_path, 'r') as f:
            dir_list = f.read().splitlines()
        return dir_list
    
    def _read_text_anno(self, seq_path):
        """读取序列的文本描述"""
        text_file = os.path.join(seq_path, "text.txt")
        if os.path.isfile(text_file):
            try:
                with open(text_file, 'r', encoding='utf-8') as f:
                    return f.read().strip()
            except Exception as e:
                print(f"Error reading text file {text_file}: {e}")
        return None
    
    def _read_audio_anno(self, seq_path):
        """读取序列的语音描述"""
        audio_file = os.path.join(seq_path, "audio_description.mp3")
        if os.path.isfile(audio_file):
            return audio_file
        return None

    def _read_bb_anno(self, seq_path):
        # in lasher dataset, visible.txt is same as infrared.txt
        rgb_bb_anno_file = os.path.join(seq_path, "init.txt")
        # ir_bb_anno_file = os.path.join(seq_path, "infrared.txt")
        rgb_gt = pandas.read_csv(rgb_bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        # ir_gt = pandas.read_csv(ir_bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=False, low_memory=False).values
        return torch.tensor(rgb_gt)

    def _get_sequence_path(self, seq_id):
        return os.path.join(self.root, self.sequence_list[seq_id])

    def get_sequence_info(self, seq_id):
        """2022/8/10 ir and rgb have synchronous w=h=0 frame_index"""
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)
        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        visible = valid.clone().byte()
        
        # 获取序列名称
        seq_name = self.sequence_list[seq_id]
        
        return {'bbox': bbox, 'valid': valid, 'visible': visible,
                'text_description': self.text_annotations.get(seq_name, None),
                'audio_description': self._read_audio_anno(seq_path)}

    def _get_frame_path(self, seq_path, frame_id):
        # Note original filename is chaotic, we rename them
        rgb_frame_path = sorted(os.listdir(os.path.join(seq_path,"visible")))  # frames start from 0
        ir_frame_path = sorted(os.listdir(os.path.join(seq_path,"infrared")))
        
        # 检查gray文件夹是否存在
        gray_dir = os.path.join(seq_path, "gray")
        if os.path.exists(gray_dir):
            gray_frame_path = sorted(os.listdir(gray_dir))
            gray_path = os.path.join(seq_path, 'gray', gray_frame_path[frame_id])
        else:
            gray_path = None

        return (os.path.join(seq_path,'visible',rgb_frame_path[frame_id]),
                os.path.join(seq_path,'infrared',ir_frame_path[frame_id]),
                gray_path)

    def _get_frame(self, seq_path, frame_id):
        rgb_frame_path, ir_frame_path, gray_frame_path = self._get_frame_path(seq_path, frame_id)
        img = get_lasher_frame(rgb_frame_path, ir_frame_path, gray_frame_path, dtype=self.dtype)
        return img  # (h,w,15)

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)

        frame_list = [self._get_frame(seq_path, f_id) for f_id in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            if key in ['text_description', 'audio_description']:
                anno_frames[key] = value
            else:
                anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]
        
        # 添加前一帧bbox信息
        prev_frame_bboxes = []
        for f_id in frame_ids:
            prev_id = self._get_previous_valid_frame(anno['bbox'], f_id)
            if prev_id is not None:
                prev_frame_bboxes.append(anno['bbox'][prev_id, ...].clone())
            else:
                # 如果找不到前一帧，使用当前帧的bbox
                prev_frame_bboxes.append(anno['bbox'][f_id, ...].clone())
        
        anno_frames['prev_bbox'] = prev_frame_bboxes

        object_meta = OrderedDict({'object_class_name': None,
                                   'motion_class': None,
                                   'major_class': None,
                                   'root_class': None,
                                   'motion_adverb': None})

        return frame_list, anno_frames, object_meta
    
    def _get_previous_valid_frame(self, bbox, frame_id):
        """
        获取指定帧的前一个有效帧ID（bbox不为[0,0,0,0]）
        
        Args:
            bbox: 所有帧的bbox标注 (N, 4)
            frame_id: 当前帧ID
            
        Returns:
            前一个有效帧的ID，如果找不到则返回None
        """
        # 从当前帧往前查找
        for prev_id in range(frame_id - 1, -1, -1):
            # 检查该帧的bbox是否有效（不全为0）
            if torch.sum(torch.abs(bbox[prev_id])) > 0:
                return prev_id
        return None
