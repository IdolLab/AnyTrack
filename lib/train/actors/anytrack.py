import pdb

from . import BaseActor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate, cosine_anneal_tau
from lib.train.admin import multigpu
from lib.train.data.processing_utils import transform_image_to_crop
import torch.nn.functional as F


class ANYTRACKActor(BaseActor):
    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg

    def fix_bns(self):
        net = self.net.module if multigpu.is_multi_gpu(self.net) else self.net
        net.box_head.apply(self.fix_bn)

    def fix_bn(self, m):
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.eval()

    def __call__(self, data):

        # forward pass
        out_dict = self.forward_pass(data)

        # compute losses
        loss, status = self.compute_losses(out_dict, data)

        return loss, status

    def forward_pass(self, data):

        template_list = []
        search_list = []
        for i in range(self.settings.num_template):
            template_img = data['template_images'][i].view(-1, *data['template_images'].shape[2:])
            template_list.append(template_img)
        for i in range(self.settings.num_search):
            search_img = data['search_images'][i].view(-1, *data['search_images'].shape[2:])
            search_list.append(search_img)

        text_description = data['text_description']
        audio_description = data['audio_description']
        pre_box = data['search_prev_anno']
        cur_box = data['search_anno']
        if pre_box is not None and cur_box is not None:
            # 创建新的pre_box张量
            new_pre_box = pre_box.clone()  # 保留原始的pre_box[0]
            # 从cur_box复制值到new_pre_box的后续位置
            for i in range(len(cur_box)):
                if i + 1 < len(new_pre_box):  # 确保不会越界
                    new_pre_box[i + 1] = cur_box[i]
            pre_box = new_pre_box

        tau = cosine_anneal_tau(epoch=data['epoch'], total_anneal_epochs=self.cfg.TRAIN.LR_DROP_EPOCH)

        out_dict = self.net(template=template_list,
                            search=search_list,
                            text_description=text_description,
                            audio_description=audio_description,
                            pre_box=pre_box,
                            cur_box=cur_box,
                            track_query_before=None,
                            tau=tau)

        return out_dict

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        
        loss_dict = {}
        total_status = {}
        total_loss = torch.tensor(0., dtype=torch.float).cuda()
        gt_gaussian_maps_list = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)

        # gt gaussian map
        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)  # (B,1,H,W)


        for i in range(len(pred_dict)):
            # get GT
            gt_bbox = gt_dict['search_anno'][i]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
            gt_gaussian_maps = gt_gaussian_maps_list[i].unsqueeze(1)

            # Get boxes
            pred_boxes = pred_dict[i]['pred_boxes']
            if torch.isnan(pred_boxes).any():
                raise ValueError("Network outputs is NAN! Stop Training")
            num_queries = pred_boxes.size(1)
            pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
            gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0, max=1.0)
            # (B,4) --> (B,1,4) --> (B,N,4)
            
            # compute giou and iou
            try:
                giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
            except:
                giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
            loss_dict['giou'] = giou_loss
            
            # compute l1 loss
            l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
            loss_dict['l1'] = l1_loss
            
            # compute location loss
            if 'score_map' in pred_dict[i]:
                location_loss = self.objective['focal'](pred_dict[i]['score_map'], gt_gaussian_maps)
            else:
                location_loss = torch.tensor(0.0, device=l1_loss.device)
            loss_dict['focal'] = location_loss
                
            # weighted sum
            loss = sum(loss_dict[k] * self.loss_weight[k] for k in loss_dict.keys() if k in self.loss_weight)

            if 'l_aux' in pred_dict[i]:
                aux_loss = pred_dict[i]['l_aux']
                loss += aux_loss
            else:
                aux_loss = torch.tensor(0.0, device=l1_loss.device)

            dag_loss = torch.tensor(0.0, device=l1_loss.device)
            if 'l_dag' in pred_dict[i]:
                def dag_regularization(adj):
                    B, N, _ = adj.shape
                    A_sq = adj * adj
                    loss_dag = torch.tensor(0.0, device=l1_loss.device)
                    for b in range(B):
                        exp_A = torch.matrix_exp(A_sq[b])
                        loss_dag += torch.trace(exp_A) - N
                    return loss_dag / B

                for l, adj in enumerate(pred_dict[i]['l_dag']):
                    dag_loss += dag_regularization(adj)
                dag_loss = 0.0001 * dag_loss
                loss += dag_loss
            # if self.cfg.PROMPT.token_loss:
            #     # gt_cls_tokens = self.compute_gt_feat(gt_dict)
            #     token_bbox = pred_dict[i]['token_feats']
            #     try:
            #         giou_loss_token, iou_token = self.objective['giou'](token_bbox, gt_boxes_vec)
            #     except:
            #         giou_loss_token, iou_token = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
            #     l1_loss_token = self.objective['l1'](token_bbox, gt_boxes_vec)
            #     token_loss = self.loss_weight['giou'] * giou_loss_token + self.loss_weight['l1'] * l1_loss_token
            #     loss += token_loss

            total_loss += loss
            
            if return_status:
                # status for log
                status = {}
                
                mean_iou = iou.detach().mean()
                status = {f"{i}frame_Loss/total": loss.item(),
                        f"{i}frame_Loss/giou": giou_loss.item(),
                        f"{i}frame_Loss/l1": l1_loss.item(),
                        f"{i}frame_Loss/location": location_loss.item(),
                        f"{i}frame_Loss/aux": aux_loss.item(),
                        f"{i}frame_Loss/dag": dag_loss.item(),
                        f"{i}frame_IoU": mean_iou.item()}
                    
                total_status.update(status)

        if return_status:
            return total_loss, total_status
        else:
            return total_loss
        
        # # Get boxes
        # pred_boxes = pred_dict['pred_boxes']
        # if torch.isnan(pred_boxes).any():
        #     raise ValueError("Network outputs is NAN! Stop Training")
        # num_queries = pred_boxes.size(1)
        # pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        # gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
        #                                                                                                    max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # # compute giou and iou
        # try:
        #     giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        # except:
        #     giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # # compute l1 loss
        # l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        # # compute location loss
        # if 'score_map' in pred_dict:
        #     location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        # else:
        #     location_loss = torch.tensor(0.0, device=l1_loss.device)
        # # weighted sum
        # loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss
        # if return_status:
        #     # status for log
        #     mean_iou = iou.detach().mean()
        #     status = {"Loss/total": loss.item(),
        #               "Loss/giou": giou_loss.item(),
        #               "Loss/l1": l1_loss.item(),
        #               "Loss/location": location_loss.item(),
        #               "IoU": mean_iou.item()}
        #     return loss, status
        # else:
        #     return loss
