from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import pdb
import cv2
import torch
# import vot
import sys
import time
import os
from lib.test.evaluation import Tracker
import lib.test.vot.vot as vot
from lib.test.vot.vot22_utils import *
from lib.train.dataset.depth_utils import get_rgbd_frame
from unified_test import multi_modal_get_x_frame
import json


class sttrack(object):
    def __init__(self, tracker_name='', para_name='',run_id=None,conf_thr=None,update_intervals=None):
        # create tracker
        tracker_info = Tracker(tracker_name, para_name, "vot22", run_id=run_id)
        params = tracker_info.get_parameters()
        params.visualization = False
        params.debug = False
        self.tracker = tracker_info.create_tracker(params,conf_thr=conf_thr,update_intervals=update_intervals)

    def write(self, str):
        txt_path = ""
        file = open(txt_path, 'a')
        file.write(str)

    def initialize(self, img_rgb, selection, text_description=None, audio_description=None, seq_mask_list=None):
        # init on the 1st frame
        # region = rect_from_mask(mask)
        x, y, w, h = selection
        bbox = [x, y, w, h]
        self.H, self.W, _ = img_rgb.shape
        init_info = {'init_bbox': bbox}

        # 添加文本和音频描述到初始化信息中
        if text_description is not None:
            init_info['text_description'] = text_description
        if audio_description is not None:
            init_info['audio_description'] = audio_description
        if seq_mask_list is not None:
            init_info['seq_mask_list'] = seq_mask_list

        _ = self.tracker.initialize(img_rgb, init_info)

    def track(self, img_rgb):
        # track
        outputs = self.tracker.track(img_rgb)
        pred_bbox = outputs['target_bbox']
        max_score = outputs['best_score']  #.max().cpu().numpy()
        return pred_bbox, max_score


def run_vot_exp(tracker_name, para_name, vis=False, out_conf=False, channel_type='color', run_id=None, conf_thr=None, update_intervals=None, modalities=None, fill_modalities='text,audio', miss=False):

    torch.set_num_threads(1)
    save_root = os.path.join('', para_name)
    if vis and (not os.path.exists(save_root)):
        os.mkdir(save_root)
    tracker = sttrack(tracker_name=tracker_name, para_name=para_name,run_id=run_id,conf_thr=conf_thr,update_intervals=update_intervals)

    if channel_type=='rgb':
        channel_type=None
    handle = vot.VOT("rectangle", channels=channel_type)

    selection = handle.region()
    imagefile = handle.frame()
    if not imagefile:
        sys.exit(0)
    if vis:
        '''for vis'''
        seq_name = imagefile.split('/')[-3]
        save_v_dir = os.path.join(save_root,seq_name)
        if not os.path.exists(save_v_dir):
            os.mkdir(save_v_dir)
        cur_time = int(time.time() % 10000)
        save_dir = os.path.join(save_v_dir, str(cur_time))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    # 获取序列路径以读取文本和音频描述
    seq_path = None
    if isinstance(imagefile, list):
        # 从第一个图像路径推断序列路径
        img_path = imagefile[0] if len(imagefile) > 0 else imagefile
        # 从图像路径中提取序列路径，假设格式为: .../sequence_name/color/00000001.jpg
        path_parts = img_path.split(os.sep)
        for i in range(len(path_parts) - 1, -1, -1):
            if path_parts[i] == 'color':
                seq_path = os.sep.join(path_parts[:i])
                break
    else:
        img_path = imagefile
        path_parts = img_path.split(os.sep)
        for i in range(len(path_parts) - 1, -1, -1):
            if path_parts[i] == 'color':
                seq_path = os.sep.join(path_parts[:i])
                break

    if isinstance(fill_modalities, str):
        fill_modalities = fill_modalities.split(',')

    use_text = 'text' in fill_modalities
    use_audio = 'audio' in fill_modalities
    # 读取文本描述
    text_description = None
    if seq_path and use_text:
        text_file = os.path.join(seq_path, "text.txt")
        if os.path.exists(text_file):
            try:
                with open(text_file, 'r', encoding='utf-8') as f:
                    text_description = f.read().strip()
            except Exception as e:
                print(f"Error reading text file {text_file}: {e}")

    # 读取音频描述
    audio_description = None
    if seq_path and use_audio:
        audio_file = os.path.join(seq_path, "audio_description.mp3")
        if os.path.exists(audio_file):
            audio_description = audio_file

    seq_mask_list = None
    seq_name = imagefile[0].split('/')[-3]
    if miss:
        json_path = '/media/sqh/DataDisk/RGBDdatasets/DepthTrack/miss/missing_results.json'
        with open(json_path, 'r') as load_f:
            mask_dict = json.load(load_f)
        seq_mask_list = mask_dict[seq_name]['data']

    # read rgbd data
    if isinstance(imagefile, list) and len(imagefile)==2:
        rgb_path = imagefile[0]
        depth_path = imagefile[1]

        # 根据 modalities 参数分配模态路径
        color_path = None
        gray_path = None
        depth_path_assigned = None

        if modalities is None:
            # 默认使用所有可用模态
            color_path = rgb_path
            depth_path_assigned = depth_path
        else:
            # 根据指定的模态分配路径
            for modality in modalities:
                if modality == 'rgb':
                    color_path = rgb_path
                elif modality == 'gray':
                    # 通过路径替换获取 gray 路径
                    path_parts = rgb_path.split(os.sep)
                    for i in range(len(path_parts) - 1, -1, -1):
                        if path_parts[i] == 'color':
                            path_parts[i] = 'gray'
                            break
                    gray_path = os.sep.join(path_parts)
                elif modality == 'depth':
                    depth_path_assigned = depth_path

        image = multi_modal_get_x_frame(color_path=color_path, depth_path=depth_path_assigned,
                                        infrared_path=None, gray_path=gray_path,
                                        event_path=None)
        # rgb_path = imagefile[0]
        # # 通过路径替换直接获取gray路径
        # path_parts = rgb_path.split(os.sep)
        # for i in range(len(path_parts) - 1, -1, -1):
        #     if path_parts[i] == 'color':
        #         path_parts[i] = 'gray'
        #         break
        # gray_path = os.sep.join(path_parts)
        # image = multi_modal_get_x_frame(color_path=imagefile[0], depth_path=imagefile[1],
        #                                 infrared_path=None, gray_path=None,
        #                                 event_path=None)
    else:
        image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB) # Right

    tracker.initialize(image, selection, text_description=text_description, audio_description=audio_description, seq_mask_list=seq_mask_list)

    while True:
        imagefile = handle.frame()
        if not imagefile:
            break

        # read rgbd data
        if isinstance(imagefile, list) and len(imagefile) == 2:
            rgb_path = imagefile[0]
            depth_path = imagefile[1]

            # 根据 modalities 参数分配模态路径
            color_path = None
            gray_path = None
            depth_path_assigned = None

            if modalities is None:
                # 默认使用所有可用模态
                color_path = rgb_path
                depth_path_assigned = depth_path
            else:
                # 根据指定的模态分配路径
                for modality in modalities:
                    if modality == 'rgb':
                        color_path = rgb_path
                    elif modality == 'gray':
                        # 通过路径替换获取 gray 路径
                        path_parts = rgb_path.split(os.sep)
                        for i in range(len(path_parts) - 1, -1, -1):
                            if path_parts[i] == 'color':
                                path_parts[i] = 'gray'
                                break
                        gray_path = os.sep.join(path_parts)
                    elif modality == 'depth':
                        depth_path_assigned = depth_path

            image = multi_modal_get_x_frame(color_path=color_path, depth_path=depth_path_assigned,
                                            infrared_path=None, gray_path=gray_path,
                                            event_path=None)
            # rgb_path = imagefile[0]
            # # 通过路径替换直接获取gray路径
            # path_parts = rgb_path.split(os.sep)
            # for i in range(len(path_parts) - 1, -1, -1):
            #     if path_parts[i] == 'color':
            #         path_parts[i] = 'gray'
            #         break
            # gray_path = os.sep.join(path_parts)
            # image = multi_modal_get_x_frame(color_path=imagefile[0], depth_path=imagefile[1],
            #                                 infrared_path=None, gray_path=None,
            #                                 event_path=None)
        else:
            image = cv2.cvtColor(cv2.imread(imagefile), cv2.COLOR_BGR2RGB)  # Right

        b1, max_score = tracker.track(image)


        if out_conf:
            handle.report(vot.Rectangle(*b1), max_score)
        else:
            handle.report(vot.Rectangle(*b1))
        if vis:
            '''Visualization'''
            # original image
            image_ori = image[:,:,::-1].copy() # RGB --> BGR
            image_name = imagefile.split('/')[-1]
            save_path = os.path.join(save_dir, image_name)
            image_b = image_ori.copy()
            cv2.rectangle(image_b, (int(b1[0]), int(b1[1])),
                          (int(b1[0] + b1[2]), int(b1[1] + b1[3])), (0, 0, 255), 2)
            image_b_name = image_name.replace('.jpg','_bbox.jpg')
            save_path = os.path.join(save_dir, image_b_name)
            cv2.imwrite(save_path, image_b)

