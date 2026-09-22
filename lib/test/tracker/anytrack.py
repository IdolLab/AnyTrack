import math
import numpy as np
from lib.models.anytrack import build_anytrack
from lib.test.tracker.basetracker import BaseTracker
import torch

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
import torchaudio
from lib.train.data.processing_utils import sample_target, transform_image_to_crop
from lib.utils.box_ops import box_xywh_to_xyxy
# for debug
import cv2
import os

from lib.test.tracker.data_utils import PreprocessorMM
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond


class AnyTrack(BaseTracker):
    _sequence_id_counter = 0
    def __init__(self, params, dataset_name=None):
        super(AnyTrack, self).__init__(params)
        network = build_anytrack(params.cfg, training=False)

        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        print('Load pretrained model from: ' + self.params.checkpoint)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = PreprocessorMM(mean= self.cfg.DATA.MEAN, std = self.cfg.DATA.STD)
        self.state = None
        self.num_template = self.cfg.DATA.TEMPLATE.NUMBER
        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()
        # self.update_intervals = self.cfg.TEST.UPDATE_INTERVALS
        # self.update_threshold = self.cfg.TEST.UPDATE_THRESHOLD
        if dataset_name:
            # 从配置文件中获取数据集特定的参数
            test_params = getattr(self.cfg, 'TEST', {})

            update_intervals_map = test_params.get('UPDATE_INTERVALS', {})
            update_threshold_map = test_params.get('UPDATE_THRESHOLD', {})

            # 获取数据集特定的参数，如果不存在则使用默认值
            self.update_intervals = update_intervals_map.get(dataset_name,
                                                             test_params.get('DEFAULT_UPDATE_INTERVALS', 5))
            self.update_threshold = update_threshold_map.get(dataset_name,
                                                             test_params.get('DEFAULT_UPDATE_THRESHOLD', 0.65))
        else:
            # 使用配置文件中的默认值
            test_params = getattr(self.cfg, 'TEST', {})
            self.update_intervals = test_params.get('DEFAULT_UPDATE_INTERVALS', 5)
            self.update_threshold = test_params.get('DEFAULT_UPDATE_THRESHOLD', 0.65)

        # for debug
        if getattr(params, 'debug', None) is None:
            setattr(params, 'debug', 0)
        self.use_visdom = False #params.debug
        self.debug = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                # self.add_hook()
                self._init_visdom(None, 1)

        # self.template_list_v = []
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}
        self.text_feat = None
        self.audio_feat = None
        self.traj = []
        self.crop_sz = torch.Tensor([self.params.search_size, self.params.search_size])
        self.sequence_id = AnyTrack._sequence_id_counter
        AnyTrack._sequence_id_counter += 1

    def initialize(self, image, info: dict):

        z_patch_arr, resize_factor, z_amask_arr  = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        
        template = self.preprocessor.process(z_patch_arr)
        with torch.no_grad():
            self.z_dict = [template]* self.num_template

        # Extract text and audio features if provided in info
        self.text_feat = None
        self.audio_feat = None
        
        # Load and process text description
        if 'text_description' in info and info['text_description'] is not None:
            text_description = info['text_description']
            if isinstance(text_description, str) and os.path.isfile(text_description):
                # Read text from file
                with open(text_description, 'r', encoding='utf-8') as f:
                    text_content = f.read().strip()
            else:
                text_content = str(text_description)

            self.text_feat = [text_content]
        
        # Load and process audio description
        if 'audio_description' in info and info['audio_description'] is not None:
            audio_path = info['audio_description']
            self.audio_feat = [audio_path]

        self.box_mask_z = None
        self.memory_bank = None
        self.track_query_before = None
        self.seq_mask_list = info.get('seq_mask_list', None)
        self.conf_score = None

        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        
        # Initialize trajectory with initial bbox
        if 'init_bbox' in info:
            self.traj = [torch.Tensor(self.state)]  # Only keep the initial bbox
        else:
            self.traj = []
        
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr)

        with torch.no_grad():
            x_dict = [search]
            # Calculate pre_box based on trajectory (first frame uses ground truth, subsequent frames use predictions)
            if len(self.traj) > 0:
                # Transform the previous bbox to current search crop coordinate system
                box_traj_crop = [
                    transform_image_to_crop(a_gt, torch.Tensor(self.state), resize_factor, self.crop_sz,
                                            normalize=True) for a_gt in self.traj]
                pre_box = torch.stack(box_traj_crop, dim=0)
                pre_box = pre_box.to(search.device)
                pre_box = pre_box.unsqueeze(0)

            # 如果有 seq_mask_list，根据当前帧的模态状态处理
            if self.seq_mask_list is not None and self.frame_id < len(self.seq_mask_list):
                mask_frame = self.seq_mask_list[self.frame_id]
                # 判断模态数量
                num_modalities = len(mask_frame)
                # 双模态情况 [rgb_mask, tir_mask]
                if num_modalities == 2:
                    rgb_mask = mask_frame[0]
                    tir_mask = mask_frame[1]
                    if rgb_mask == 0.0 and tir_mask > 0.0:
                        # 010: RGB 缺失，只用 TIR (后 3 通道)
                        template_input = [z[:, 3:, :, :] for z in self.z_dict]
                        search_input = [x[:, 3:, :, :] for x in x_dict]
                        track_query_before_input = [self.track_query_before[1]] if self.track_query_before else None
                    elif rgb_mask > 0.0 and tir_mask == 0.0:
                        # 100: TIR 缺失，只用 RGB (前 3 通道)
                        template_input = [z[:, :3, :, :] for z in self.z_dict]
                        search_input = [x[:, :3, :, :] for x in x_dict]
                        track_query_before_input = [self.track_query_before[0]] if self.track_query_before else None
                    elif rgb_mask == 0.0 and tir_mask == 0.0:
                        if self.conf_score != None:
                            return {"target_bbox": self.state,
                                    "best_score": self.conf_score.cpu().numpy()[0][0]
                                    }
                        else:
                            return {"target_bbox": self.state,
                                    "best_score": None
                                    }
                # 三模态情况
                elif num_modalities == 3:
                    m1_mask = mask_frame[0]  # 第一个模态（如 RGB）
                    m2_mask = mask_frame[1]  # 第二个模态（如 Gray）
                    m3_mask = mask_frame[2]  # 第三个模态（如 Depth/TIR）
                    channels_per_modality = 3  # 每个模态 3 通道
                    if m1_mask == 0.0 and m2_mask > 0.0 and m3_mask == 0.0:
                        # 010: 只有第 2 个模态存在
                        start_ch = 1 * channels_per_modality
                        end_ch = 2 * channels_per_modality
                        template_input = [z[:, start_ch:end_ch, :, :] for z in self.z_dict]
                        search_input = [x[:, start_ch:end_ch, :, :] for x in x_dict]
                        track_query_before_input = [self.track_query_before[1]] if self.track_query_before else None
                    elif m1_mask > 0.0 and m2_mask == 0.0 and m3_mask == 0.0:
                        # 100: 只有第 1 个模态存在
                        start_ch = 0
                        end_ch = channels_per_modality
                        template_input = [z[:, start_ch:end_ch, :, :] for z in self.z_dict]
                        search_input = [x[:, start_ch:end_ch, :, :] for x in x_dict]
                        track_query_before_input = [self.track_query_before[0]] if self.track_query_before else None
                    elif m1_mask == 0.0 and m2_mask == 0.0 and m3_mask > 0.0:
                        # 001: 只有第 3 个模态存在
                        start_ch = 2 * channels_per_modality
                        end_ch = 3 * channels_per_modality
                        template_input = [z[:, start_ch:end_ch, :, :] for z in self.z_dict]
                        search_input = [x[:, start_ch:end_ch, :, :] for x in x_dict]
                        track_query_before_input = [self.track_query_before[2]] if self.track_query_before else None
                    elif m1_mask > 0.0 and m2_mask > 0.0 and m3_mask == 0.0:
                        # 110: 第 1 和第 2 个模态存在
                        ch1_start = 0
                        ch1_end = channels_per_modality
                        ch2_start = 1 * channels_per_modality
                        ch2_end = 2 * channels_per_modality
                        template_input = [torch.cat([z[:, ch1_start:ch1_end, :, :],
                                                     z[:, ch2_start:ch2_end, :, :]], dim=1) for z in self.z_dict]
                        search_input = [torch.cat([x[:, ch1_start:ch1_end, :, :],
                                                   x[:, ch2_start:ch2_end, :, :]], dim=1) for x in x_dict]
                        track_query_before_input = [self.track_query_before[0],
                                                    self.track_query_before[1]] if self.track_query_before else None
                    elif m1_mask == 0.0 and m2_mask > 0.0 and m3_mask > 0.0:
                        # 011: 第 2 和第 3 个模态存在
                        ch2_start = 1 * channels_per_modality
                        ch2_end = 2 * channels_per_modality
                        ch3_start = 2 * channels_per_modality
                        ch3_end = 3 * channels_per_modality
                        template_input = [torch.cat([z[:, ch2_start:ch2_end, :, :],
                                                     z[:, ch3_start:ch3_end, :, :]], dim=1) for z in self.z_dict]
                        search_input = [torch.cat([x[:, ch2_start:ch2_end, :, :],
                                                   x[:, ch3_start:ch3_end, :, :]], dim=1) for x in x_dict]
                        track_query_before_input = [self.track_query_before[1],
                                                    self.track_query_before[2]] if self.track_query_before else None
                    elif m1_mask > 0.0 and m2_mask == 0.0 and m3_mask > 0.0:
                        # 101: 第 1 和第 3 个模态存在，第 2 个缺失
                        ch1_start = 0
                        ch1_end = channels_per_modality
                        ch3_start = 2 * channels_per_modality
                        ch3_end = 3 * channels_per_modality
                        template_input = [torch.cat([z[:, ch1_start:ch1_end, :, :],
                                                     z[:, ch3_start:ch3_end, :, :]], dim=1) for z in self.z_dict]
                        search_input = [torch.cat([x[:, ch1_start:ch1_end, :, :],
                                                   x[:, ch3_start:ch3_end, :, :]], dim=1) for x in x_dict]
                        track_query_before_input = [self.track_query_before[0],
                                                    self.track_query_before[2]] if self.track_query_before else None
                    elif m1_mask == 0.0 and m2_mask == 0.0 and m3_mask == 0.0:
                        if self.conf_score != None:
                            return {"target_bbox": self.state,
                                    "best_score": self.conf_score.cpu().numpy()[0][0]
                                    }
                        else:
                            return {"target_bbox": self.state,
                                    "best_score": None
                                    }
                # 如果没有匹配到任何缺失模式，使用原始数据
                if 'template_input' not in locals():
                    template_input = self.z_dict
                    search_input =  x_dict
                    track_query_before_input = self.track_query_before
                out_dict = self.network.forward(
                    template=template_input,
                    search=search_input, memory_bank=self.memory_bank,
                    text_description=self.text_feat, audio_description=self.audio_feat, pre_box=pre_box,
                    track_query_before=track_query_before_input)
            else:
                # merge the template and the search
                # run the transformer
                out_dict = self.network.forward(
                    template= self.z_dict,
                    search=x_dict, memory_bank=self.memory_bank,
                    text_description=self.text_feat, audio_description=self.audio_feat, pre_box=pre_box,
                    track_query_before=self.track_query_before)

        # Update memory bank with output from network
        if 'memory_bank' in out_dict[0]:
            self.memory_bank = out_dict[0]['memory_bank']

        if 'track_query_before' in out_dict[0]:
            if self.seq_mask_list is not None and self.frame_id < len(self.seq_mask_list):
                mask_frame = self.seq_mask_list[self.frame_id]
                num_modalities = len(mask_frame)
                # 双模态情况
                if num_modalities == 2:
                    rgb_mask = mask_frame[0]
                    tir_mask = mask_frame[1]
                    # 初始化 self.track_query_before（如果是第一次）
                    if self.track_query_before is None:
                        self.track_query_before = [None] * num_modalities
                    if rgb_mask == 0.0 and tir_mask > 0.0:
                        # 010: RGB 缺失，只更新 TIR (索引 1)
                        self.track_query_before[1] = out_dict[0]['track_query_before'][0]
                    elif rgb_mask > 0.0 and tir_mask == 0.0:
                        # 100: TIR 缺失，只更新 RGB (索引 0)
                        self.track_query_before[0] = out_dict[0]['track_query_before'][0]
                    else:
                        # 双模态都存在，完整更新
                        self.track_query_before = out_dict[0]['track_query_before']
                # 三模态情况
                elif num_modalities == 3:
                    m1_mask = mask_frame[0]
                    m2_mask = mask_frame[1]
                    m3_mask = mask_frame[2]
                    # 初始化 self.track_query_before（如果是第一次）
                    if self.track_query_before is None:
                        self.track_query_before = [None] * num_modalities
                    if m1_mask == 0.0 and m2_mask > 0.0 and m3_mask == 0.0:
                        # 010: 只有第 2 个模态存在，更新索引 1
                        self.track_query_before[1] = out_dict[0]['track_query_before'][0]
                    elif m1_mask > 0.0 and m2_mask == 0.0 and m3_mask == 0.0:
                        # 100: 只有第 1 个模态存在，更新索引 0
                        self.track_query_before[0] = out_dict[0]['track_query_before'][0]
                    elif m1_mask == 0.0 and m2_mask == 0.0 and m3_mask > 0.0:
                        # 001: 只有第 3 个模态存在，更新索引 2
                        self.track_query_before[2] = out_dict[0]['track_query_before'][0]
                    elif m1_mask > 0.0 and m2_mask > 0.0 and m3_mask == 0.0:
                        # 110: 第 1+2 个模态存在，更新索引 0 和 1
                        track_query_output = out_dict[0]['track_query_before']
                        # 假设输出是两个模态的拼接，需要拆分
                        self.track_query_before[0] = track_query_output[0]
                        self.track_query_before[1] = track_query_output[1]
                    elif m1_mask == 0.0 and m2_mask > 0.0 and m3_mask > 0.0:
                        # 011: 第 2+3 个模态存在，更新索引 1 和 2
                        track_query_output = out_dict[0]['track_query_before']
                        self.track_query_before[1] = track_query_output[0]
                        self.track_query_before[2] = track_query_output[1]
                    elif m1_mask > 0.0 and m2_mask == 0.0 and m3_mask > 0.0:
                        # 101: 第 1 和第 3 个模态存在，第 2 个缺失，更新索引 0 和 2
                        track_query_output = out_dict[0]['track_query_before']
                        self.track_query_before[0] = track_query_output[0]
                        self.track_query_before[2] = track_query_output[1]
                    else:
                        # 所有模态都存在，完整更新
                        self.track_query_before = out_dict[0]['track_query_before']
            else:
                # 没有 seq_mask_list，直接更新
                self.track_query_before = out_dict[0]['track_query_before']
            # self.track_query_before = out_dict[0]['track_query_before']

        if 'text_feat' in out_dict[0]:
            self.text_feat = out_dict[0]['text_feat']

        if 'audio_feat' in out_dict[0]:
            self.audio_feat = out_dict[0]['audio_feat']

        # add hann windows
        pred_score_map = out_dict[0]['score_map']
        response = self.output_window * pred_score_map
        pred_boxes = self.network.box_head.cal_bbox(response, out_dict[0]['size_map'], out_dict[0]['offset_map'])
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        self.traj.append(torch.Tensor(self.state))
        # Update trajectory: remove oldest and add current frame's prediction
        if len(self.traj) > 1:
            self.traj.pop(0)  # Remove the oldest trajectory

        
        conf_score = None
        if self.num_template > 1:
            conf_score, idx = torch.max(response.flatten(1), dim=1, keepdim=True)
            self.conf_score =conf_score
            if (self.frame_id % self.update_intervals == 0) and (conf_score > self.update_threshold):
            
                z_patch_arr, resize_factor, z_amask_arr = sample_target(image, self.state, self.params.template_factor,
                                                                        output_sz=self.params.template_size)
                self.z_patch_arr = z_patch_arr
                template = self.preprocessor.process(z_patch_arr)
                self.z_dict.append(template)
                if len(self.z_dict) > self.num_template:
                    self.z_dict.pop(1)

        # for debug
        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=2)
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image_BGR)
            else:
                self.visdom.register((image, info['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking')

                self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region')
                self.visdom.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), 'image', 1, 'template')
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap',
                                     1, 'score_map_hann')

                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                    removed_indexes_s = out_dict['removed_indexes_s']
                    removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                    masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search')

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            if conf_score != None:
                return {"target_bbox": self.state,
                        "all_boxes": all_boxes_save,
                        "best_score":conf_score.cpu().numpy()[0][0]}
            else:
                return {"target_bbox": self.state,
                        "all_boxes": all_boxes_save,
                        "best_score":None
                        }
        else:
            if conf_score != None:
                return {"target_bbox": self.state,
                        "best_score":conf_score.cpu().numpy()[0][0]
                        }
            else:
                return {"target_bbox": self.state,
                        "best_score":None
                        }


    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1)  # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights


def get_tracker_class():
    return AnyTrack
