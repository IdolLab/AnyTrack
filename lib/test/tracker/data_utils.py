import torch
import numpy as np
import torchvision.transforms.functional as tvisf

class Preprocessor(object):
    def __init__(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view((1, 3, 1, 1)).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).view((1, 3, 1, 1)).cuda()

    def process(self, img_arr: np.ndarray):
        # Deal with the image patch
        img_tensor = torch.tensor(img_arr).cuda().float().permute((2,0,1)).unsqueeze(dim=0)
        img_tensor_norm = ((img_tensor / 255.0) - self.mean) / self.std  # (1,3,H,W)
        return img_tensor_norm


class PreprocessorMM(object):
    def __init__(self, mean, std):
        self.mean_base = mean  # [0.485, 0.456, 0.406] - 第一个模态的mean
        self.std_base = std  # [0.229, 0.224, 0.225] - 第一个模态的std
        self.mean_others = [0.449, 0.449, 0.449]  # 其他模态的mean
        self.std_others = [0.226, 0.226, 0.226]  # 其他模态的std

    def process(self, img_arr: np.ndarray):
        # Deal with the image patch
        img_tensor = torch.tensor(img_arr).cuda().float().permute((2, 0, 1)).unsqueeze(dim=0)  # (1,C,H,W)

        # 获取通道数
        C = img_tensor.shape[1]

        # 根据通道数确定模态数量（每个模态3个通道）
        num_modalities = C // 3

        # 构建mean和std列表，第一个模态使用base值，其他模态使用others值
        mean_repeated = []
        std_repeated = []

        for i in range(num_modalities):
            if i == 0:  # 第一个模态使用base mean和std
                mean_repeated.extend(self.mean_base)
                std_repeated.extend(self.std_base)
            else:  # 其他模态使用others mean和std
                mean_repeated.extend(self.mean_others)
                std_repeated.extend(self.std_others)

        # 转换为tensor并调整形状
        mean_tensor = torch.tensor(mean_repeated).view((1, C, 1, 1)).cuda()
        std_tensor = torch.tensor(std_repeated).view((1, C, 1, 1)).cuda()

        # 归一化
        img_tensor_norm = ((img_tensor / 255.0) - mean_tensor) / std_tensor  # (1,C,H,W)
        return img_tensor_norm


class PreprocessorX(object):
    def __init__(self):
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view((1, 3, 1, 1)).cuda()
        self.std = torch.tensor([0.229, 0.224, 0.225]).view((1, 3, 1, 1)).cuda()

    def process(self, img_arr: np.ndarray, amask_arr: np.ndarray):
        # Deal with the image patch
        img_tensor = torch.tensor(img_arr).cuda().float().permute((2,0,1)).unsqueeze(dim=0)
        img_tensor_norm = ((img_tensor / 255.0) - self.mean) / self.std  # (1,3,H,W)
        # Deal with the attention mask
        amask_tensor = torch.from_numpy(amask_arr).to(torch.bool).cuda().unsqueeze(dim=0)  # (1,H,W)
        return img_tensor_norm, amask_tensor


class PreprocessorX_onnx(object):
    def __init__(self):
        self.mean = np.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))
        self.std = np.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))

    def process(self, img_arr: np.ndarray, amask_arr: np.ndarray):
        """img_arr: (H,W,3), amask_arr: (H,W)"""
        # Deal with the image patch
        img_arr_4d = img_arr[np.newaxis, :, :, :].transpose(0, 3, 1, 2)
        img_arr_4d = (img_arr_4d / 255.0 - self.mean) / self.std  # (1, 3, H, W)
        # Deal with the attention mask
        amask_arr_3d = amask_arr[np.newaxis, :, :]  # (1,H,W)
        return img_arr_4d.astype(np.float32), amask_arr_3d.astype(np.bool)
