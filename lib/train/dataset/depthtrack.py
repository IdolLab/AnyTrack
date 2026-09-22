import os
import os.path
import torch
import numpy as np
import pandas
import csv
import cv2
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader_w_failsafe
from lib.train.admin import env_settings
from lib.train.dataset.depth_utils import get_x_frame

def get_depthtrack_frame(color_path, depth_path, gray_path, dtype='rgbcolormap', depth_clip=True):
    """读取DepthTrack数据集的多模态帧（RGB + Depth + Gray）
    
    Args:
        color_path: RGB图像路径
        depth_path: 深度图像路径
        gray_path: 灰度图像路径
        dtype: 数据类型，默认'rgbcolormap'
        depth_clip: 是否裁剪深度值
    
    Returns:
        合并后的多模态图像 (H, W, 15)
    """
    # 读取RGB图像
    rgb = cv2.imread(color_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    
    # 读取深度图像
    depth = cv2.imread(depth_path, -1)
    if depth_clip:
        max_depth = min(np.median(depth) * 3, 10000)
        depth[depth > max_depth] = max_depth
    
    # 将深度图转换为colormap
    depth = cv2.normalize(depth, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    depth = np.asarray(depth, dtype=np.uint8)
    depth_colormap = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
    depth_colormap = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2RGB)
    
    # 读取灰度图像
    if gray_path and os.path.exists(gray_path):
        gray = cv2.imread(gray_path)
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)
    else:
        # 如果gray不存在，使用RGB转灰度
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    
    # 合并三个模态 (H, W, 15)
    img = cv2.merge((rgb, gray, depth_colormap, depth_colormap))
    # img = cv2.merge((rgb, depth_colormap, depth_colormap))
    return img

class DepthTrack(BaseVideoDataset):
    """ DepthTrack dataset.
    """

    def __init__(self, root=None, dtype='rgbcolormap', split='train', image_loader=jpeg4py_loader_w_failsafe): #  vid_ids=None, split=None, data_fraction=None
        """
        args:

            image_loader (jpeg4py_loader) -  The function to read the images. jpeg4py (https://github.com/ajkxyz/jpeg4py)
                                            is used by default.
            vid_ids - List containing the ids of the videos (1 - 20) used for training. If vid_ids = [1, 3, 5], then the
                    videos with subscripts -1, -3, and -5 from each class will be used for training.
            # split - If split='train', the official train split (protocol-II) is used for training. Note: Only one of
            #         vid_ids or split option can be used at a time.
            # data_fraction - Fraction of dataset to be used. The complete dataset is used by default

            root     - path to the lasot depth dataset.
            dtype    - colormap or depth,, colormap + depth
                        if colormap, it returns the colormap by cv2,
                        if depth, it returns [depth, depth, depth]
        """
        root = env_settings().depthtrack_dir if root is None else root
        super().__init__('DepthTrack', root, image_loader)

        self.dtype = dtype  # colormap or depth
        self.split = split
        self.sequence_list = self._build_sequence_list()

        self.seq_per_class, self.class_list = self._build_class_list()
        self.class_list.sort()
        self.class_to_id = {cls_name: cls_id for cls_id, cls_name in enumerate(self.class_list)}
        
        # 预加载文本标注（体积小）
        self.text_annotations = {}
        for seq_name in self.sequence_list:
            seq_path = os.path.join(self.root, seq_name)
            self.text_annotations[seq_name] = self._read_text_anno(seq_path)

    def _build_sequence_list(self):

        ltr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
        file_path = os.path.join(ltr_path, 'data_specs', 'depthtrack_%s.txt'%self.split)
        sequence_list = pandas.read_csv(file_path, header=None).squeeze("columns").values.tolist()
        return sequence_list

    def _build_class_list(self):
        seq_per_class = {}
        class_list = []
        for seq_id, seq_name in enumerate(self.sequence_list):
            class_name = seq_name.split('_')[0]

            if class_name not in class_list:
                class_list.append(class_name)

            if class_name in seq_per_class:
                seq_per_class[class_name].append(seq_id)
            else:
                seq_per_class[class_name] = [seq_id]

        return seq_per_class, class_list

    def get_name(self):
        return 'depthtrack'

    def has_class_info(self):
        return True

    def has_occlusion_info(self):
        return True

    def get_num_sequences(self):
        return len(self.sequence_list)

    def get_num_classes(self):
        return len(self.class_list)

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]
    
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
        bb_anno_file = os.path.join(seq_path, "groundtruth.txt")
        gt = pandas.read_csv(bb_anno_file, delimiter=',', header=None, dtype=np.float32, na_filter=True, low_memory=False).values
        return torch.tensor(gt)

    def _get_sequence_path(self, seq_id):
        seq_name = self.sequence_list[seq_id]
        return os.path.join(self.root, seq_name)

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)  # xywh just one kind label
        '''
        if the box is too small, it will be ignored
        '''
        # valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        valid = (bbox[:, 2] > 10.0) & (bbox[:, 3] > 10.0)
        visible = valid.clone().byte()
        
        # 获取序列名称
        seq_name = self.sequence_list[seq_id]
        
        return {'bbox': bbox, 'valid': valid, 'visible': visible,
                'text_description': self.text_annotations.get(seq_name, None),
                'audio_description': self._read_audio_anno(seq_path)}

    def _get_frame_path(self, seq_path, frame_id):
        '''
        return depth image path
        '''
        color_path = os.path.join(seq_path, 'color', '{:08}.jpg'.format(frame_id+1))
        depth_path = os.path.join(seq_path, 'depth', '{:08}.png'.format(frame_id+1))
        
        # 检查gray文件夹是否存在
        gray_dir = os.path.join(seq_path, "gray")
        if os.path.exists(gray_dir):
            gray_path = os.path.join(gray_dir, '{:08}.jpg'.format(frame_id+1))
        else:
            gray_path = None
        
        return color_path, depth_path, gray_path  # frames start from 1

    def _get_frame(self, seq_path, frame_id):
        '''
        Return :
            - colormap from depth image
            - 3xD = [depth, depth, depth], 255
            - rgbcolormap
            - rgb3d
            - color
            - raw_depth
        '''
        color_path, depth_path, gray_path = self._get_frame_path(seq_path, frame_id)
        img = get_depthtrack_frame(color_path, depth_path, gray_path, dtype=self.dtype, depth_clip=True)

        return img  # (h,w,15)

    def _get_class(self, seq_path):
        # raw_class = seq_path.split('/')[-2]
        # return raw_class
        return self.split

    def get_class_name(self, seq_id):
        depth_path = self._get_sequence_path(seq_id)
        obj_class = self._get_class(depth_path)

        return obj_class

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)

        obj_class = self._get_class(seq_path)

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            if key in ['text_description', 'audio_description']:
                anno_frames[key] = value
            else:
                anno_frames[key] = [value[f_id, ...].clone() for ii, f_id in enumerate(frame_ids)]
        
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

        frame_list = [self._get_frame(seq_path, f_id) for ii, f_id in enumerate(frame_ids)]

        object_meta = OrderedDict({'object_class_name': obj_class,
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
