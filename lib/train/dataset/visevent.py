import os
import os.path
import torch
import numpy as np
import pandas
import csv
import cv2
from glob import glob
from collections import OrderedDict
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader_w_failsafe
from lib.train.admin import env_settings
from lib.train.dataset.depth_utils import get_x_frame

def get_visevent_frame(vis_path, event_path, gray_path, dtype='rgbrgb'):
    """读取VisEvent数据集的多模态帧（Visible + Event + Gray）
    
    Args:
        vis_path: 可见光图像路径
        event_path: 事件图像路径
        gray_path: 灰度图像路径
        dtype: 数据类型，默认'rgbrgb'
    
    Returns:
        合并后的多模态图像 (H, W, 15)
    """
    # 读取可见光图像
    vis = cv2.imread(vis_path)
    vis = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    
    # 读取事件图像
    event = cv2.imread(event_path, -1)
    event = cv2.cvtColor(event, cv2.COLOR_BGR2RGB)
    
    # 读取灰度图像
    if gray_path and os.path.exists(gray_path):
        gray = cv2.imread(gray_path)
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)
    else:
        # 如果gray不存在，使用visible转灰度
        gray = cv2.cvtColor(vis, cv2.COLOR_RGB2GRAY)
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    
    # 合并三个模态 (H, W, 15)
    img = cv2.merge((vis, gray, event, event))
    # img = cv2.merge((vis, event, event))
    return img


class VisEvent(BaseVideoDataset):
    """ VisEvent dataset.
    """

    def __init__(self, root=None, dtype='rgbrgb', split='train', image_loader=jpeg4py_loader_w_failsafe): #  vid_ids=None, split=None, data_fraction=None
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
        root = env_settings().visevent_dir if root is None else root
        assert split in ['train'], 'Only support train split in VisEvent, got {}'.format(split)
        super().__init__('VisEvent', root, image_loader)

        self.dtype = dtype  # colormap or depth
        self.split = split
        self.sequence_list = self._build_sequence_list()
        
        # 预加载文本标注（体积小）
        self.text_annotations = {}
        for seq_name in self.sequence_list:
            seq_path = os.path.join(self.root, seq_name)
            self.text_annotations[seq_name] = self._read_text_anno(seq_path)


    def _build_sequence_list(self):
        
        file_path = os.path.join(self.root, '../{}list.txt'.format(self.split))
        sequence_list = pandas.read_csv(file_path, header=None).squeeze("columns").values.tolist()
        return sequence_list

    def get_name(self):
        return 'visevent'

    def has_class_info(self):
        return False

    def has_occlusion_info(self):
        return True

    def get_num_sequences(self):
        return len(self.sequence_list)
    
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

    def _read_target_visible(self, seq_path):
        # Read full occlusion and out_of_view
        occlusion_file = os.path.join(seq_path, "absent_label.txt")

        with open(occlusion_file, 'r', newline='') as f:
            occlusion = torch.ByteTensor([int(v[0]) for v in list(csv.reader(f))])

        target_visible = occlusion

        return target_visible

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
        valid = (bbox[:, 2] > 5.0) & (bbox[:, 3] > 5.0)
        visible = self._read_target_visible(seq_path) & valid.byte()
        
        # 获取序列名称
        seq_name = self.sequence_list[seq_id]
        
        return {'bbox': bbox, 'valid': valid, 'visible': visible,
                'text_description': self.text_annotations.get(seq_name, None),
                'audio_description': self._read_audio_anno(seq_path)}

    def _get_frame_path(self, seq_path, frame_id):
        '''
        return rgb event image path
        '''
        vis_img_files = sorted(glob(os.path.join(seq_path, 'vis_imgs', '*.bmp')))

        try:
            vis_path = vis_img_files[frame_id]
        except:
            print(f"seq_path: {seq_path}")
            print(f"vis_img_files: {vis_img_files}")
            print(f"frame_id: {frame_id}")

        event_path = vis_path.replace('vis_imgs', 'event_imgs')
        
        # 检查gray文件夹是否存在
        gray_dir = os.path.join(seq_path, "gray_imgs")
        if os.path.exists(gray_dir):
            gray_files = sorted(glob(os.path.join(gray_dir, '*.bmp')))
            if frame_id < len(gray_files):
                gray_path = gray_files[frame_id]
            else:
                gray_path = None
        else:
            gray_path = None

        return vis_path, event_path, gray_path  # frames start irregularly

    def _get_frame(self, seq_path, frame_id):
        '''
        Return :
            - rgb+event_colormap+gray
        '''
        vis_path, event_path, gray_path = self._get_frame_path(seq_path, frame_id)
        img = get_visevent_frame(vis_path, event_path, gray_path, dtype=self.dtype)
        return img  # (h,w,15)

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)

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
