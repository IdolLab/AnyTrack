import math
from operator import ipow
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head, conv
from lib.utils.box_ops import box_xyxy_to_cxcywh, box_xywh_to_xyxy, box_cxcywh_to_xyxy, box_xyxy_to_xywh

from CLIP import clip
from timm.models.layers import Mlp
from lib.models.layers.attn import Attention_qkv
from transformers import WavLMModel, Wav2Vec2FeatureExtractor
import torchaudio
from lib.models.anytrack.prompt import build_promptDecoder_traj,build_promptEncoder_traj
import torch.nn.functional as F
from lib.models.anytrack.itpn import fast_itpn_base_3324_patch16_224
from lib.models.layers.moe_lora import MoE_lora

model_clip, _ = clip.load("/home/sqh/lihao/AnyTrack/CLIP/ViT-B-32.pt", device='cpu')
for p in model_clip.parameters():
    p.requires_grad = False

wavlm_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("/home/sqh/lihao/AnyTrack/microsoft/wavlm-base-plus-sv", local_files_only=True)
wavlm_model = WavLMModel.from_pretrained("/home/sqh/lihao/AnyTrack/microsoft/wavlm-base-plus-sv", local_files_only=True)
wavlm_model.eval()
for p in wavlm_model.parameters():
    p.requires_grad = False

from lib.models.sam2_components import build_memory_encoder

class AnyTrack(nn.Module):

    def __init__(self, transformer, box_head, promptEnc, promptDec, memory_encoder, cfg, aux_loss=False, head_type="CORNER"):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        hidden_dim = transformer.embed_dim
        self.backbone = transformer
        self.decode_fuse_search = conv(hidden_dim, hidden_dim)
        self.box_head = box_head

        self.model_clip = model_clip
        # self.text_inverseNet = Mlp(in_features=512, hidden_features=512 * 4, out_features=768)
        self.audio_inverseNet = Mlp(in_features=768, hidden_features=768 * 4, out_features=512)
        self.wavlm_processor, self.wavlm_model =  wavlm_feature_extractor, wavlm_model
        self.promptEnc = promptEnc
        self.promptDec = promptDec

        self.CSS_strengthen = Attention_qkv(hidden_dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.)
        self.CSS_process = Mlp(in_features=hidden_dim, hidden_features=int(hidden_dim * 4.), act_layer=nn.GELU,
                drop=0.)

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)
        self.SparseMoE = MoE_lora(input_size=hidden_dim, output_size=hidden_dim, num_experts=6, hidden_size=hidden_dim, noisy_gating=True, k=2,patch_num=self.feat_len_s)

        self.memory_encoder = memory_encoder

        self.sigma_adjustment_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 2)  # 输出2个值：dx, dy用于调整sigma_x和sigma_y
        )
        
    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                pre_box=None,
                text_description=None,
                audio_description=None,
                cur_box=None,
                memory_bank=None,
                track_query_before = None,
                tau=0.1,
                ):

        out_dict = []
        device = next(self.parameters()).device
        # 文本特征提取
        # 如果text_description已经是特征（tensor），直接使用；否则进行编码
        text_feat = None
        if text_description is not None:
            if isinstance(text_description, torch.Tensor):
                # 已经是预计算的特征，直接使用
                text_feat = text_description
            else:
                # 需要进行编码
                text_tokens = clip.tokenize(texts=text_description, truncate=True).to(device)
                text_feat = self.model_clip.encode_text(text_tokens)
                text_feat = text_feat.unsqueeze(1)
        
        # ===== 语音特征提取（直接从全局模型获取）=====
        audio_feat = None
        if audio_description is not None and self.wavlm_model is not None:
            if isinstance(audio_description, torch.Tensor):
                # 已经是预计算的特征，直接使用
                audio_feat = audio_description
            else:
                # 需要进行编码
                batch_audio_feats = []
                for audio_path in audio_description:
                    if audio_path is not None and os.path.isfile(audio_path):
                        try:
                            # 音频加载（保持原有逻辑）
                            waveform, sr = torchaudio.load(audio_path)
                            if waveform.shape[0] > 1:
                                waveform = torch.mean(waveform, dim=0, keepdim=True)
                            if sr != 16000:
                                resampler = torchaudio.transforms.Resample(sr, 16000)
                                waveform = resampler(waveform)
                            # 使用全局处理器和模型
                            inputs = self.wavlm_processor(
                                waveform.squeeze().numpy(),
                                sampling_rate=16000,
                                return_tensors="pt",
                                padding=True
                            )
                            input_values = inputs.input_values.to(device)
                            with torch.no_grad():
                                outputs = self.wavlm_model(input_values)
                                hidden_states = outputs.last_hidden_state
                            pooled_feat = hidden_states.mean(dim=1, keepdim=True)
                            batch_audio_feats.append(pooled_feat)
                        except Exception as e:
                            print(f"Error processing audio {audio_path}: {e}")
                            batch_audio_feats.append(torch.zeros(1, 1, 768).to(device))
                    else:
                        batch_audio_feats.append(torch.zeros(1, 1, 768).to(device))
                audio_feat = torch.cat(batch_audio_feats, dim=0)
                audio_feat = self.audio_inverseNet(audio_feat)

        for i in range(len(search)):
            x, aux_dict, len_zx = self.backbone(z=template, x=search[i], track_query = track_query_before, tau = tau)

            pre_box_ = box_xywh_to_xyxy(pre_box[i])
            pre_box_ = pre_box_.unsqueeze(1)
            box_feat = self.promptEnc(pre_box_)
            x_pe = self.promptEnc.get_dense_pe((self.feat_sz_s, self.feat_sz_s))
            # 收集所有有效的prompt特征
            prompt_feats = []
            if text_feat is not None:
                prompt_feats.append(text_feat)
            if audio_feat is not None:
                prompt_feats.append(audio_feat)
            if box_feat is not None:
                prompt_feats.append(box_feat)
            # 按dim=1拼接
            if len(prompt_feats) > 0:
                prompt_feat = torch.cat(prompt_feats, dim=1)
            else:
                prompt_feat = None

            num_modalities = search[i].shape[1] // 3
            _, N_t, C_per_modality = x.size()
            N_per_modality = N_t // num_modalities
            modality_features = []
            track_query_list = []
            l_aux_list = []
            moe_total_ms = 0.0
            for nm in range(num_modalities):
                modality_feat = x[:, nm * N_per_modality:(nm + 1) * N_per_modality, :]
                track_query_cue = modality_feat[:, :1, :]
                track_query_list.append((track_query_cue.clone()).detach())
                modality_feat_feature = modality_feat[:, -self.feat_len_s:]
                modality_feat_feature, l_aux = self.SparseMoE(modality_feat_feature)
                track_query_cue_attn = track_query_cue + self.CSS_strengthen(track_query_cue, modality_feat_feature, modality_feat_feature)
                track_query_cue_attn = track_query_cue_attn + self.CSS_process(track_query_cue_attn)
                att_r = torch.matmul(modality_feat_feature, track_query_cue_attn.transpose(1, 2))
                modality_feat_feature = att_r * modality_feat_feature
                l_aux_list.append(l_aux)
                modality_features.append(modality_feat_feature)
            track_query_before = track_query_list
            cat_features = torch.sum(torch.stack(modality_features), dim=0)
            opt = (cat_features.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
            bs, Nq, C, HW = opt.size()
            opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

            opt_feat = self.decode_fuse_search(opt_feat)
            VIS = opt_feat.clone()
            if memory_bank is not None:
                prompt, src = self.promptDec(opt_feat, x_pe.expand(bs * Nq, -1, -1, -1), prompt_feat, memory_bank)
            else:
                prompt, src = self.promptDec(opt_feat, x_pe.expand(bs * Nq, -1, -1, -1), prompt_feat, opt_feat)
            src_enc = (src.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
            opt_feat = src_enc.view(-1, C_per_modality, self.feat_sz_s, self.feat_sz_s)

            out = self.forward_head(opt_feat, None)

            out['l_aux'] = sum(l_aux_list) if l_aux_list else torch.tensor(0.0, device=x.device)
            out['l_dag'] = aux_dict['l_dag_list']

            mask_size = search[i].size(2)
            search_mask = torch.zeros(bs, mask_size, mask_size, device=opt_feat.device)
            # 判断是训练还是测试
            if cur_box is not None:
                # 训练阶段：使用真值cur_box (xywh格式，需要转换为xyxy)
                box = box_xywh_to_xyxy(cur_box[i])
            else:
                # 测试阶段：使用预测框out['pred_boxes'] (cxcywh格式，需要转换为xyxy)
                box = box_cxcywh_to_xyxy(out['pred_boxes'].squeeze(1))  # [bs, 4]
            # 将归一化坐标乘以256得到实际坐标
            box = box * mask_size
            # 为每个batch生成mask
            for b in range(bs):
                x1, y1, x2, y2 = box[b]
                x1 = max(0, int(x1.item()))
                y1 = max(0, int(y1.item()))
                x2 = min(mask_size, int((x2.item())))
                y2 = min(mask_size, int((y2.item())))
                # # 框内区域设置为1
                # search_mask[b, y1:y2, x1:x2] = 1.0

                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0
                # 计算目标框的宽高
                box_w = x2 - x1
                box_h = y2 - y1

                # 提取目标框内的特征用于预测标准差调整参数
                feat_H, feat_W = opt_feat.size(2), opt_feat.size(3)
                scale_x = feat_W / mask_size
                scale_y = feat_H / mask_size

                # 计算特征图上的坐标
                x1_f = int(x1 * scale_x)
                y1_f = int(y1 * scale_y)
                x2_f = int(x2 * scale_x)
                y2_f = int(y2 * scale_y)

                # 确保坐标在范围内
                x1_f = max(0, min(x1_f, feat_W - 1))
                y1_f = max(0, min(y1_f, feat_H - 1))
                x2_f = max(x1_f + 1, min(x2_f, feat_W))
                y2_f = max(y1_f + 1, min(y2_f, feat_H))

                # 提取目标区域特征
                roi_feat = opt_feat[b:b + 1, :, y1_f:y2_f, x1_f:x2_f]  # [1, C, H_roi, W_roi]

                # 全局平均池化得到固定长度特征
                roi_feat_pooled = F.adaptive_avg_pool2d(roi_feat, (1, 1))  # [1, C, 1, 1]
                roi_feat_flat = roi_feat_pooled.squeeze(-1).squeeze(-1)  # [1, C]

                # 使用网络预测标准差调整参数
                adjustments = self.sigma_adjustment_net(roi_feat_flat)  # [1, 2]
                adjustment_x = adjustments[0, 0]  # 标量
                adjustment_y = adjustments[0, 1]  # 标量

                # 设置高斯分布的标准差，可以根据目标框大小自适应调整
                sigma_x = max(1.0, box_w / 6.0 + adjustment_x.item())  # 应用学习到的调整
                sigma_y = max(1.0, box_h / 6.0 + adjustment_y.item())  # 应用学习到的调整


                # 生成坐标网格
                y_coords, x_coords = torch.meshgrid(
                    torch.arange(mask_size, device=opt_feat.device, dtype=torch.float),
                    torch.arange(mask_size, device=opt_feat.device, dtype=torch.float)
                )
                # 计算高斯分布（只在目标框内）
                gaussian = torch.exp(
                    -0.5 * (((x_coords - center_x) / sigma_x) ** 2 + ((y_coords - center_y) / sigma_y) ** 2))

                # 只在目标框内应用高斯分布，框外为0
                mask_region = torch.zeros_like(gaussian)
                mask_region[y1:y2, x1:x2] = gaussian[y1:y2, x1:x2]

                search_mask[b] = mask_region

            search_mask = search_mask.unsqueeze(1)

            memory_bank = self.memory_encoder(
                pix_feat=opt_feat,  
                masks=search_mask,  
                skip_mask_sigmoid=True 
            )

            out['memory_bank'] = memory_bank
            out['track_query_before'] = track_query_before
            out['audio_feat'] = audio_feat
            out['text_feat'] = text_feat
            out['backbone_feat'] = VIS
            out_dict.append(out)
        return out_dict

    def forward_head(self, opt_feat ,gt_score_map=None):

        bs, C, H, W = opt_feat.size()
        Nq = 1

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out
        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError


def build_anytrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained')
    if cfg.MODEL.PRETRAIN_FILE and ('SOT' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'itpn_base':
        backbone = fast_itpn_base_3324_patch16_224(pretrained, cfg=cfg)
        patch_start_index = 0
        backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)
    else:
        raise NotImplementedError

    hidden_dim = backbone.embed_dim

    box_head = build_box_head(cfg, hidden_dim)

    promptEnc = build_promptEncoder_traj(cfg)
    promptDec = build_promptDecoder_traj(cfg)
    memory_encoder = build_memory_encoder(
        out_dim=hidden_dim,  
        in_dim=hidden_dim,  
    )

    model = AnyTrack(
        backbone,
        box_head,
        promptEnc,
        promptDec,
        memory_encoder,
        cfg,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )


    if training and 'SOT' in cfg.MODEL.PRETRAIN_FILE:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        param_dict_rgbt = dict()
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)
        print('missing_keys, unexpected_keys', missing_keys, unexpected_keys)

    return model

