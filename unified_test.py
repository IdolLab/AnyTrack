import os
import cv2
import sys
from os.path import join, isdir, abspath, dirname
import numpy as np
import argparse
prj = join(dirname(__file__), '..')
if prj not in sys.path:
    sys.path.append(prj)
import json

# 数据集路径配置 - 请根据实际情况修改
DATASET_PATHS = {
    'LasHeR': '/media/sqh/DataDisk/RGBTdatasets/LasHeR/test',  # 请替换为实际路径
    'VisEvent': '/media/sqh/DataDisk/RGBEdatasets/VisEvent/test',  # 请替换为实际路径
    'RGBDT500': '/media/sqh/DataDisk/RGBDT500/Test',  # 请替换为实际路径
}

from lib.test.tracker.anytrack import AnyTrack
import lib.test.parameter.anytrack as params
import multiprocessing
import torch
import time
import pandas

def genConfig(seq_path, set_type):
    """
    生成序列配置，包括图像路径列表和标注
    
    Args:
        seq_path: 序列路径
        set_type: 数据集类型
        
    Returns:
        img_lists: 包含所有模态图像路径的列表
        RGB_gt: 标注边界框
    """
    if set_type == 'LasHeR':
        # LasHeR数据集格式：visible + gray + infrared (RGB + GRAY + TIR)
        RGB_img_list = sorted([os.path.join(seq_path, 'visible', p) for p in os.listdir(os.path.join(seq_path, 'visible')) if p.endswith(('.jpg', '.png'))])
        
        # 获取gray模态
        gray_dir = os.path.join(seq_path, 'gray')
        gray_img_list = []
        if os.path.exists(gray_dir):
            gray_files = [f for f in os.listdir(gray_dir) if f.endswith(('.jpg', '.png'))]
            gray_img_list = sorted([os.path.join(seq_path, 'gray', f) for f in gray_files])
        
        # 获取infrared模态
        infrared_dir = os.path.join(seq_path, 'infrared')
        infrared_img_list = []
        if os.path.exists(infrared_dir):
            infrared_files = [f for f in os.listdir(infrared_dir) if f.endswith(('.jpg', '.png'))]
            infrared_img_list = sorted([os.path.join(seq_path, 'infrared', f) for f in infrared_files])
        
        # 确保所有模态图像列表长度一致
        min_len = len(RGB_img_list)
        if gray_img_list:
            min_len = min(min_len, len(gray_img_list))
        if infrared_img_list:
            min_len = min(min_len, len(infrared_img_list))
        
        RGB_img_list = RGB_img_list[:min_len]
        if gray_img_list:
            gray_img_list = gray_img_list[:min_len]
        if infrared_img_list:
            infrared_img_list = infrared_img_list[:min_len]
        
        # 对于LasHeR，使用init.txt作为标注
        init_path = os.path.join(seq_path, 'init.txt')
        if os.path.exists(init_path):
            RGB_gt = np.loadtxt(init_path, delimiter=',')
        else:
            # 如果init.txt不存在，则使用groundtruth.txt
            groundtruth_path = os.path.join(seq_path, 'groundtruth.txt')
            if os.path.exists(groundtruth_path):
                RGB_gt = np.loadtxt(groundtruth_path, delimiter=',')
            else:
                RGB_gt = np.loadtxt(os.path.join(seq_path, 'visible.txt'), delimiter=',')
        RGB_gt = RGB_gt[:min_len]
        
        # 返回所有模态的图像列表
        img_lists = [RGB_img_list]
        if gray_img_list:
            img_lists.append(gray_img_list)
        if infrared_img_list:
            img_lists.append(infrared_img_list)
        
        return img_lists, RGB_gt

    elif set_type == 'VisEvent':
        # VisEvent数据集格式：vis_imgs + gray + event_imgs (RGB + GRAY + EVENT)
        RGB_img_list = sorted([os.path.join(seq_path, 'vis_imgs', p) for p in os.listdir(os.path.join(seq_path, 'vis_imgs')) if p.endswith((".bmp", ".jpg", ".png"))])
        
        # 获取gray模态
        gray_dir = os.path.join(seq_path, 'gray_imgs')
        gray_img_list = []
        if os.path.exists(gray_dir):
            gray_files = [f for f in os.listdir(gray_dir) if f.endswith((".bmp", ".jpg", ".png"))]
            gray_img_list = sorted([os.path.join(seq_path, 'gray_imgs', f) for f in gray_files])
        
        # 获取event_imgs模态
        event_dir = os.path.join(seq_path, 'event_imgs')
        event_img_list = []
        if os.path.exists(event_dir):
            event_files = [f for f in os.listdir(event_dir) if f.endswith((".bmp", ".jpg", ".png"))]
            event_img_list = sorted([os.path.join(seq_path, 'event_imgs', f) for f in event_files])
        
        # 读取absent标签
        RGB_gt = np.loadtxt(os.path.join(seq_path, 'groundtruth.txt'), delimiter=',')
        absent_label = np.loadtxt(os.path.join(seq_path, 'absent_label.txt'))
        # 处理缺失帧
        if absent_label[0] == 0: # first frame is absent in some seqs
            first_present_idx = absent_label.argmax()
            RGB_img_list = RGB_img_list[first_present_idx:]
            if gray_img_list:
                gray_img_list = gray_img_list[first_present_idx:]
            if event_img_list:
                event_img_list = event_img_list[first_present_idx:]
            RGB_gt = RGB_gt[first_present_idx:]
        
        # 确保所有模态图像列表长度一致
        min_len = len(RGB_img_list)
        if gray_img_list:
            min_len = min(min_len, len(gray_img_list))
        if event_img_list:
            min_len = min(min_len, len(event_img_list))
        
        RGB_img_list = RGB_img_list[:min_len]
        if gray_img_list:
            gray_img_list = gray_img_list[:min_len]
        if event_img_list:
            event_img_list = event_img_list[:min_len]
        RGB_gt = RGB_gt[:min_len]
        
        # 返回所有模态的图像列表
        img_lists = [RGB_img_list]
        if gray_img_list:
            img_lists.append(gray_img_list)
        if event_img_list:
            img_lists.append(event_img_list)
        
        return img_lists, RGB_gt
        
    elif set_type == 'RGBDT500':
        # RGBDT500数据集格式：color + gray + depth + infrared (RGB + GRAY + DEPTH + TIR)
        color_dir = os.path.join(seq_path, 'color')
        depth_dir = os.path.join(seq_path, 'depth')
        infrared_dir = os.path.join(seq_path, 'infrared')
            
        # 检查是否有gray模态
        gray_dir = os.path.join(seq_path, 'gray')
            
        # 根据文件数量生成文件名列表
        color_files = [f for f in os.listdir(color_dir) if f.endswith(('.jpg', '.png'))]
        RGB_img_list = sorted([os.path.join(seq_path, 'color', f) for f in color_files])
            
        # 获取gray模态
        gray_img_list = []
        if os.path.exists(gray_dir):
            gray_files = [f for f in os.listdir(gray_dir) if f.endswith(('.jpg', '.png'))]
            gray_img_list = sorted([os.path.join(seq_path, 'gray', f) for f in gray_files])
            
        # 获取depth模态
        depth_img_list = []
        if os.path.exists(depth_dir):
            depth_files = [f for f in os.listdir(depth_dir) if f.endswith(('.png'))]  # depth files are typically .png
            depth_img_list = sorted([os.path.join(seq_path, 'depth', f) for f in depth_files])
            
        # 获取infrared模态
        infrared_img_list = []
        if os.path.exists(infrared_dir):
            infrared_files = [f for f in os.listdir(infrared_dir) if f.endswith(('.jpg', '.png'))]
            infrared_img_list = sorted([os.path.join(seq_path, 'infrared', f) for f in infrared_files])
    
        RGB_gt = np.loadtxt(os.path.join(seq_path, 'groundtruth.txt'), delimiter=',')
        # 检查是否有absent标签
        absent_path = os.path.join(seq_path, 'absent_label.txt')
        if os.path.exists(absent_path):
            absent_label = np.loadtxt(absent_path)
            if absent_label[0] == 0:
                first_present_idx = absent_label.argmax()
                RGB_img_list = RGB_img_list[first_present_idx:]
                if gray_img_list:
                    gray_img_list = gray_img_list[first_present_idx:]
                if depth_img_list:
                    depth_img_list = depth_img_list[first_present_idx:]
                if infrared_img_list:
                    infrared_img_list = infrared_img_list[first_present_idx:]
                RGB_gt = RGB_gt[first_present_idx:]
            
        # 确保所有模态图像列表长度一致
        min_len = len(RGB_img_list)
        if gray_img_list:
            min_len = min(min_len, len(gray_img_list))
        if depth_img_list:
            min_len = min(min_len, len(depth_img_list))
        if infrared_img_list:
            min_len = min(min_len, len(infrared_img_list))
            
        RGB_img_list = RGB_img_list[:min_len]
        if gray_img_list:
            gray_img_list = gray_img_list[:min_len]
        if depth_img_list:
            depth_img_list = depth_img_list[:min_len]
        if infrared_img_list:
            infrared_img_list = infrared_img_list[:min_len]
        RGB_gt = RGB_gt[:min_len]
            
        # 返回所有模态的图像列表
        img_lists = [RGB_img_list]
        if gray_img_list:
            img_lists.append(gray_img_list)
        if depth_img_list:
            img_lists.append(depth_img_list)
        if infrared_img_list:
            img_lists.append(infrared_img_list)
            
        return img_lists, RGB_gt

    else:
        raise ValueError(f"Unsupported dataset type: {set_type}")


def read_text_description(seq_path):
    """
    读取序列的文本描述
    """
    text_file = os.path.join(seq_path, "text.txt")
    if os.path.isfile(text_file):
        try:
            with open(text_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            print(f"Error reading text file {text_file}: {e}")

    #读取 language.txt（用于 TNL2K数据集）
    language_file = os.path.join(seq_path, "language.txt")
    if os.path.isfile(language_file):
        try:
            with open(language_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            print(f"Error reading language file {language_file}: {e}")

    return None


def read_audio_description(seq_path):
    """
    读取序列的音频描述
    """
    audio_file = os.path.join(seq_path, "audio_description.mp3")
    if os.path.isfile(audio_file):
        return audio_file
    return None


def check_dataset_paths():
    """
    检查数据集路径是否存在
    """
    for dataset_name, path in DATASET_PATHS.items():
        if os.path.exists(path):
            print(f"✓ {dataset_name} 路径存在: {path}")
        else:
            print(f"✗ {dataset_name} 路径不存在: {path}")
    
    print("\n请确保数据集路径正确配置！")


def multi_modal_get_x_frame(color_path=None, depth_path=None, infrared_path=None, gray_path=None, event_path=None):
    """
    处理多个模态的图像路径并合并，参考训练代码的处理方式

    Args:
        color_path: RGB图像路径
        depth_path: 深度图像路径
        infrared_path: 红外图像路径
        gray_path: 灰度图像路径
        event_path: 事件图像路径
        fill_modalities: 填充模态类型列表，用于补齐到4个模态 (['rgb'], ['rgb', 'gray'], 等)

    Returns:
        合并后的多模态图像 (固定4个通道)
    """
    import cv2
    import numpy as np
    
    # 读取RGB图像
    rgb = None
    if color_path:
        rgb = cv2.imread(color_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    
    # 读取深度图像并转换为colormap
    colormap = None
    if depth_path:
        dp = cv2.imread(depth_path, -1)
        if dp is not None:
            # 对深度图进行处理
            max_depth = min(np.median(dp) * 3, 10000)
            dp[dp > max_depth] = max_depth
            dp = cv2.normalize(dp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            dp = np.asarray(dp, dtype=np.uint8)
            colormap = cv2.applyColorMap(dp, cv2.COLORMAP_JET)  # (h,w) -> (h,w,3)
            colormap = cv2.cvtColor(colormap, cv2.COLOR_BGR2RGB)
    
    # 读取红外图像
    infrared = None
    if infrared_path:
        ir = cv2.imread(infrared_path, -1)
        infrared = cv2.cvtColor(ir, cv2.COLOR_BGR2RGB)
    
    # 读取灰度图像
    gray = None
    if gray_path:
        gray_img = cv2.imread(gray_path, -1)
        gray = cv2.cvtColor(gray_img, cv2.COLOR_BGR2RGB)
    
    # 读取事件图像
    event = None
    if event_path:
        event_img = cv2.imread(event_path, -1)
        event = cv2.cvtColor(event_img, cv2.COLOR_BGR2RGB)

    # 按照rgb, gray, depth(colormap), infrared, event的顺序收集有效的模态
    valid_modalities = []

    # 按顺序添加存在的模态
    if rgb is not None:
        valid_modalities.append(rgb)
    if gray is not None:
        valid_modalities.append(gray)
    if colormap is not None:  
        valid_modalities.append(colormap)
    if infrared is not None:
        valid_modalities.append(infrared)
    if event is not None:
        valid_modalities.append(event)

    if len(valid_modalities) == 1:
        img = valid_modalities[0]
    else:
        img = cv2.merge(tuple(valid_modalities))

    return img


def run_sequence(seq_name, seq_home, dataset_name, yaml_name, num_gpu=1, epoch=300, debug=0, script_name='anytrack', fill_modalities=['rgb'], modalities=None, miss=0,):
    """
    运行单个序列的跟踪
    """
    try:
        worker_name = multiprocessing.current_process().name
        worker_id = int(worker_name[worker_name.find('-') + 1:]) - 1
        gpu_id = worker_id % num_gpu
        torch.cuda.set_device(gpu_id)
    except:
        pass

    # 设置结果保存路径
    seq_txt = seq_name.split('/')[-1] if '/' in seq_name else seq_name
    save_name = f'{yaml_name}'
    save_path = f'./results/{dataset_name}/{save_name}_{epoch}/' + seq_txt + '.txt'
    save_folder = f'./results/{dataset_name}/{save_name}_{epoch}/'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if os.path.exists(save_path):
        print(f'-1 {seq_name}')
        return

    # 初始化跟踪器
    if script_name == 'anytrack':
        param = params.parameters(yaml_name, epoch)
        anytrack = AnyTrack(param, dataset_name=dataset_name)
        tracker = AnyTrackWrapper(tracker=anytrack)

    seq_path = seq_home + '/' + seq_name
    print('——————————Process sequence: '+seq_name +'——————————————')
    
    # 读取图像路径和标注
    img_lists, RGB_gt = genConfig(seq_path, dataset_name)

    if miss:
        miss_path = '/'.join(seq_home.split('/')[:-1]) + '/miss'
        json_path = miss_path + '/missing_results.json'
        with open(json_path, 'r') as load_f:
            mask_dict = json.load(load_f)
        seq_mask_list = mask_dict[seq_name]['data']
    
    # 根据数据集名称确定模态类型，但如果指定了自定义模态组合，则优先使用自定义的
    if modalities is not None:
        # 使用自定义模态组合
        modalities = modalities.split(',')
    elif dataset_name == 'LasHeR':
        modalities = ['rgb', 'infrared']
    elif dataset_name == 'VisEvent':
        modalities = ['rgb', 'event']
    elif dataset_name == 'RGBDT500':
        modalities = ['rgb', 'depth', 'infrared']
    elif dataset_name == 'TNL2K':
        modalities = ['rgb']

    # 根据数据集类型建立模态到索引的映射
    modality_indices = {}
    if dataset_name == 'LasHeR':
        modality_indices = {'rgb': 0, 'gray': 1, 'infrared': 2}
    elif dataset_name == 'VisEvent':
        modality_indices = {'rgb': 0, 'gray': 1, 'event': 2}
    elif dataset_name == 'RGBDT500':
        modality_indices = {'rgb': 0, 'gray': 1, 'depth': 2, 'infrared': 3}

    # 处理填充模态参数
    if isinstance(fill_modalities, str):
        fill_modalities = fill_modalities.split(',')

    use_text = 'text' in fill_modalities
    use_audio = 'audio' in fill_modalities
    text_description = read_text_description(seq_path) if use_text else None
    audio_description = read_audio_description(seq_path) if use_audio else None

    # 初始化结果数组
    if len(img_lists[0]) == len(RGB_gt):  # img_lists[0] is RGB image list
        result = np.zeros_like(RGB_gt)
    else:
        result = np.zeros((len(img_lists[0]), 4), dtype=RGB_gt.dtype)
    result[0] = np.copy(RGB_gt[0])
    
    toc = 0
    for frame_idx in range(len(img_lists[0])):  # Iterate over the frames
        tic = cv2.getTickCount()
        if frame_idx == 0:
            # 初始化
            # 获取当前帧的所有模态图像路径
            frame_paths = [img_list[frame_idx] for img_list in img_lists]

            # 根据模态类型分配模态路径
            color_path = None
            gray_path = None
            depth_path = None
            infrared_path = None
            event_path = None

            # 根据实际模态列表分配路径
            for i, modality in enumerate(modalities):
                if i < len(frame_paths):
                    if modality == 'rgb':
                        color_path = frame_paths[modality_indices['rgb']] if 'rgb' in modality_indices and \
                                                                             modality_indices['rgb'] < len(
                            frame_paths) else None
                    elif modality == 'gray':
                        gray_path = frame_paths[modality_indices['gray']] if 'gray' in modality_indices and \
                                                                             modality_indices['gray'] < len(
                            frame_paths) else None
                    elif modality == 'depth':
                        depth_path = frame_paths[modality_indices['depth']] if 'depth' in modality_indices and \
                                                                               modality_indices['depth'] < len(
                            frame_paths) else None
                    elif modality == 'infrared':
                        infrared_path = frame_paths[modality_indices['infrared']] if 'infrared' in modality_indices and \
                                                                                     modality_indices['infrared'] < len(
                            frame_paths) else None
                    elif modality == 'event':
                        event_path = frame_paths[modality_indices['event']] if 'event' in modality_indices and \
                                                                               modality_indices['event'] < len(
                            frame_paths) else None

            image = multi_modal_get_x_frame(color_path=color_path, depth_path=depth_path,
                               infrared_path=infrared_path, gray_path=gray_path,
                               event_path=event_path)

            init_info = {
                'init_bbox': RGB_gt[0].tolist(),  # xywh
                'text_description': text_description,
                'audio_description': audio_description,
                'seq_mask_list': seq_mask_list if miss else None
            }
            tracker.initialize(image, init_info)
        elif frame_idx > 0:
            # 跟踪
            # 获取当前帧的所有模态图像路径
            frame_paths = [img_list[frame_idx] for img_list in img_lists]

            # 根据模态类型分配模态路径
            color_path = None
            gray_path = None
            depth_path = None
            infrared_path = None
            event_path = None

            # 根据实际模态列表分配路径
            for i, modality in enumerate(modalities):
                if i < len(frame_paths):
                    if modality == 'rgb':
                        color_path = frame_paths[modality_indices['rgb']] if 'rgb' in modality_indices and \
                                                                             modality_indices['rgb'] < len(
                            frame_paths) else None
                    elif modality == 'gray':
                        gray_path = frame_paths[modality_indices['gray']] if 'gray' in modality_indices and \
                                                                             modality_indices['gray'] < len(
                            frame_paths) else None
                    elif modality == 'depth':
                        depth_path = frame_paths[modality_indices['depth']] if 'depth' in modality_indices and \
                                                                               modality_indices['depth'] < len(
                            frame_paths) else None
                    elif modality == 'infrared':
                        infrared_path = frame_paths[modality_indices['infrared']] if 'infrared' in modality_indices and \
                                                                                     modality_indices['infrared'] < len(
                            frame_paths) else None
                    elif modality == 'event':
                        event_path = frame_paths[modality_indices['event']] if 'event' in modality_indices and \
                                                                               modality_indices['event'] < len(
                            frame_paths) else None

            image = multi_modal_get_x_frame(color_path=color_path, depth_path=depth_path,
                                            infrared_path=infrared_path, gray_path=gray_path,
                                            event_path=event_path)

            region, confidence = tracker.track(image)  # xywh
            result[frame_idx] = np.array(region)
        toc += cv2.getTickCount() - tic
    
    toc /= cv2.getTickFrequency()
    if not debug:
        np.savetxt(save_path, result, fmt='%.14f', delimiter=',')
    print('{} , fps:{}'.format(seq_name, frame_idx / toc))


class AnyTrackWrapper(object):

    def __init__(self, tracker):
        self.tracker = tracker

    def initialize(self, image, init_info):
        self.H, self.W, _ = image.shape
        gt_bbox_np = np.array(init_info['init_bbox']).astype(np.float32)
        
        # 将文本和音频描述添加到初始化信息中
        init_info['init_bbox'] = list(gt_bbox_np)  # input must be (x,y,w,h)
        self.tracker.initialize(image, init_info)

    def track(self, img_RGB):
        '''TRACK'''
        outputs = self.tracker.track(img_RGB)
        pred_bbox = outputs['target_bbox']
        pred_score = outputs['best_score']
        return pred_bbox, pred_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run tracker on multiple datasets.')
    parser.add_argument('--script_name', type=str, default='anytrack', help='Name of tracking method.')
    parser.add_argument('--yaml_name', type=str, default='anytrack', help='Name of tracking method.')
    parser.add_argument('--dataset_name', type=str, default='LasHeR', help='Name of dataset (LasHeR, VisEvent, RGBDT500).')
    parser.add_argument('--threads', default=1, type=int, help='Number of threads')
    parser.add_argument('--num_gpus', default=torch.cuda.device_count(), type=int, help='Number of gpus')
    parser.add_argument('--epoch', default=17, type=int, help='epochs of ckpt')
    parser.add_argument('--mode', default='sequential', type=str, help='sequential or parallel')
    parser.add_argument('--debug', default=0, type=int, help='to vis tracking results')
    parser.add_argument('--video', default='', type=str, help='specific video name')
    parser.add_argument('--check_paths', action='store_true', help='Check dataset paths and exit')
    parser.add_argument('--fill_modalities', type=str, default='text,audio', help='Fill modalities')
    parser.add_argument('--modalities', type=str, default=None, help='Custom modalities combination, e.g., "rgb,tir"')
    parser.add_argument('--miss', default=0, type=int, help='Use miss dataset')
    args = parser.parse_args()

    yaml_name = args.yaml_name
    dataset_name = args.dataset_name
    
    # 检查数据集路径
    if args.check_paths:
        check_dataset_paths()
        exit(0)
    
    # 数据集路径初始化
    seq_list = None
    if dataset_name in DATASET_PATHS:
        seq_home = DATASET_PATHS[dataset_name]
        if dataset_name == 'VisEvent':
            testlist_path = join(seq_home, '../testlist.txt')
            if os.path.exists(testlist_path):
                with open(testlist_path, 'r') as f:
                    seq_list = f.read().splitlines()
            else:
                seq_list = [f for f in os.listdir(seq_home) if isdir(join(seq_home,f))]
        else:
            seq_list = [f for f in os.listdir(seq_home) if isdir(join(seq_home,f))]
        seq_list.sort()
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    start = time.time()
    if args.mode == 'parallel':
        sequence_list = [(s, seq_home, dataset_name, args.yaml_name, args.num_gpus, args.epoch, args.debug, args.script_name, args.fill_modalities, args.modalities, args.miss) for s in seq_list]
        multiprocessing.set_start_method('spawn', force=True)
        with multiprocessing.Pool(processes=args.threads) as pool:
            pool.starmap(run_sequence, sequence_list)
    else:
        seq_list = [args.video] if args.video != '' else seq_list
        sequence_list = [(s, seq_home, dataset_name, args.yaml_name, args.num_gpus, args.epoch, args.debug, args.script_name, args.fill_modalities, args.modalities, args.miss) for s in seq_list]
        for seqlist in sequence_list:
            run_sequence(*seqlist)
    print(f"Totally cost {time.time()-start} seconds!")
