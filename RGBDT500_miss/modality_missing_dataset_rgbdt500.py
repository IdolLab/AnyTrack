import numpy as np
import random
import json
import os

# 新增函数：读取 RGBDT 数据集并返回格式化数据
def read_rgbdt_dataset(dataset_path):
    """
    读取 RGBDT 数据集，支持三个模态：color, infrared, depth
    """
    dataset = {}

    # 获取数据集路径下的所有序列文件夹
    sequence_folders = [folder for folder in os.listdir(dataset_path) 
                       if os.path.isdir(os.path.join(dataset_path, folder))]

    for sequence_folder in sequence_folders:
        sequence_dir = os.path.join(dataset_path, sequence_folder)
        
        # 初始化序列信息
        sequence_info = {"frames": None, "modalities": []}
        
        # 遍历三种模态：color, infrared, depth
        modalities = ["color", "infrared", "depth"]
        max_frames = 0
        
        for modality in modalities:
            modality_dir = os.path.join(sequence_dir, modality)
            if os.path.exists(modality_dir):
                # 读取模态下的图像序列
                frame_files = [file for file in os.listdir(modality_dir) 
                              if file.lower().endswith(('.jpg', '.png', '.jpeg'))]
                num_frames = len(frame_files)
                if num_frames > max_frames:
                    max_frames = num_frames
                sequence_info["modalities"].append(modalities.index(modality))

        sequence_info["frames"] = max_frames
        
        # 将序列信息添加到数据集字典中
        dataset[sequence_folder] = sequence_info
    
    return dataset

# 随机缺失函数，每个随机缺失帧有 7 种缺失情况（除了全不缺失）
# 编码：0=color, 1=infrared, 2=depth
# 缺失情况：
# 0: 只有 color 缺失 (011)
# 1: 只有 infrared 缺失 (101)
# 2: 只有 depth 缺失 (110)
# 3: color + infrared 缺失 (001)
# 4: color + depth 缺失 (010)
# 5: infrared + depth 缺失 (100)
# 6: 全部缺失 (000)
def random_missing(frames, missing_ratio, modalities=3):
    num_missing_frames = int(frames * missing_ratio)
    missing_frames = random.sample(range(frames), num_missing_frames)
    missing_data = np.ones((frames, modalities))
    
    for frame in missing_frames:
        # 7 种缺失情况：0-6
        missing_case = random.choice([0, 1, 2, 3, 4, 5, 6])
        
        if missing_case == 0:  # 只有 color 缺失 (011)
            missing_data[frame][0] = 0
        elif missing_case == 1:  # 只有 infrared 缺失 (101)
            missing_data[frame][1] = 0
        elif missing_case == 2:  # 只有 depth 缺失 (110)
            missing_data[frame][2] = 0
        elif missing_case == 3:  # color + infrared 缺失 (001)
            missing_data[frame][0] = 0
            missing_data[frame][1] = 0
        elif missing_case == 4:  # color + depth 缺失 (010)
            missing_data[frame][0] = 0
            missing_data[frame][2] = 0
        elif missing_case == 5:  # infrared + depth 缺失 (100)
            missing_data[frame][1] = 0
            missing_data[frame][2] = 0
        elif missing_case == 6:  # 全部缺失 (000)
            missing_data[frame][0] = 0
            missing_data[frame][1] = 0
            missing_data[frame][2] = 0
            
    return missing_data

# 长期缺失函数，缺失某个模态是随机选择的
def long_term_missing(frames, missing_ratio, modalities=3):
    num_missing_frames = int(frames * missing_ratio)
    start = random.randint(0, frames - num_missing_frames)
    missing_data = np.ones((frames, modalities))
    missing_modality = random.choice([0, 1, 2])  # 随机选择一个模态缺失
    
    for i in range(start, start + num_missing_frames):
        missing_data[i][missing_modality] = 0
    
    return missing_data

# 交换缺失函数，缺失模态是随机选择的
def swap_missing(frames, missing_ratio, modalities=3):
    num_missing_frames = int(frames * missing_ratio)
    unit_length = random.choice([frames // 6, frames // 4, frames // 2])
    missing_data = np.ones((frames, modalities))
    start = random.randint(0, frames - num_missing_frames)
    count = 0
    missing_modality = random.choice([0, 1, 2])
    
    for i in range(start, start + num_missing_frames):
        count = count + 1
        missing_data[i][missing_modality] = 0
        
        if unit_length == count:
            # 切换到下一个模态
            missing_modality = (missing_modality + 1) % 3
            count = 0
        
    return missing_data

# 长期缺失 + 随机缺失函数
def long_term_and_random_missing(frames, missing_ratio, modalities=3):
    # 长期缺失操作
    missing_data = long_term_missing(frames, missing_ratio, modalities)
    
    # 找到缺失操作的帧
    missing_frames = np.where(missing_data == 0)[0]
    
    # 随机挑选 50% 的帧并执行随机缺失操作
    num_random_missing_frames = int(len(missing_frames) * 0.5)
    random_missing_frames = random.sample(list(missing_frames), min(num_random_missing_frames, len(missing_frames)))
    
    for frame in random_missing_frames:
        # 随机选择 1-2 个额外模态缺失
        num_extra_missing = random.choice([1, 2])
        available_modalities = [m for m in range(3) if missing_data[frame][m] == 1]
        if available_modalities:
            extra_missing = random.sample(available_modalities, 
                                         min(num_extra_missing, len(available_modalities)))
            for m in extra_missing:
                missing_data[frame][m] = 0
    
    return missing_data

# 交换缺失 + 随机缺失函数
def swap_and_random_missing(frames, missing_ratio, modalities=3):
    # 交换缺失操作
    missing_data = swap_missing(frames, missing_ratio, modalities)
    
    # 找到缺失操作的帧
    missing_frames = np.where(missing_data == 0)[0]
    
    # 随机挑选 50% 的帧并执行随机缺失操作
    num_random_missing_frames = int(len(missing_frames) * 0.5)
    random_missing_frames = random.sample(list(missing_frames), min(num_random_missing_frames, len(missing_frames)))
    
    for frame in random_missing_frames:
        # 随机选择 1-2 个额外模态缺失
        num_extra_missing = random.choice([1, 2])
        available_modalities = [m for m in range(3) if missing_data[frame][m] == 1]
        if available_modalities:
            extra_missing = random.sample(available_modalities, 
                                         min(num_extra_missing, len(available_modalities)))
            for m in extra_missing:
                missing_data[frame][m] = 0
    
    return missing_data

# 定义分组函数
def split_and_shuffle_dataset(dataset):
    # 将数据集按序列数量降序排列
    sorted_dataset = sorted(dataset.items(), key=lambda x: x[1]["frames"], reverse=True)
    num_sequences = len(sorted_dataset)
    print(f"总序列数：{num_sequences}")
    
    # 缺失函数名称列表，按顺序与大分组对应
    missing_functions = ["random_missing", "long_term_missing", "swap_missing", 
                        "long_term_and_random_missing", "swap_and_random_missing"]
    
    # 初始化分组
    groups = {}

    # 每次取出 5 个序列并打乱顺序后分配到分组中
    batch_size = 5
    batch = []

    for i, (seq_name, seq_info) in enumerate(sorted_dataset):
        batch.append((seq_name, seq_info))
        
        if len(batch) == batch_size:
            random.shuffle(batch)  # 随机打乱顺序

            for j, (seq_name, seq_info) in enumerate(batch):
                group_idx = j % 5  # 计算序列应该分配到哪个大组
                group_name = f"Group{group_idx + 1}_{missing_functions[group_idx]}"  # 使用缺失函数名称
                
                if group_name not in groups:
                    groups[group_name] = {}
                    groups[group_name]["missing_function"] = missing_functions[group_idx]
                    groups[group_name]["sequences"] = []

                groups[group_name]["sequences"].append((seq_name, seq_info))
            
            batch = []  # 清空批次
    
    # 如果还有不足 5 个的剩余序列，同样要打乱顺序后分配到分组中
    if batch:
        random.shuffle(batch)

        for j, (seq_name, seq_info) in enumerate(batch):
            group_idx = j % 5  # 计算序列应该分配到哪个大组
            group_name = f"Group{group_idx + 1}_{missing_functions[group_idx]}"  # 使用缺失函数名称
            
            if group_name not in groups:
                groups[group_name] = {}
                groups[group_name]["missing_function"] = missing_functions[group_idx]
                groups[group_name]["sequences"] = []

            groups[group_name]["sequences"].append((seq_name, seq_info))
    
    # 生成五大组的 JSON 文件
    with open('groups_data_rgbdt500.json', 'w') as json_file:
        json.dump(groups, json_file, indent=4)

    return groups


def split_group_subgroups(groups):
    subgroups_data = {}

    # 缺失比例列表
    missing_ratios = [0.3, 0.6, 0.9]

    for group_name, group_info in groups.items():
        # 初始化小组信息
        subgroups_info = {}
        sequences = group_info["sequences"]
        num_sequences = len(sequences)

        # 按帧数降序排列序列
        sorted_sequences = sorted(sequences, key=lambda x: x[1]["frames"], reverse=True)

        batch = []  # 用于存储每个 batch 中的序列

        for i, (seq_name, seq_info) in enumerate(sorted_sequences):
            batch.append((seq_name, seq_info))

            # 当 batch 中有三个序列时，执行随机打乱并分配到小组中
            if len(batch) == 3:
                random.shuffle(batch)  # 随机打乱顺序

                for j, (batch_seq_name, batch_seq_info) in enumerate(batch):
                    # 每个小组对应不同的缺失比例
                    subgroup_missing_ratio = missing_ratios[j]

                    # 分配到小组中
                    subgroup_name = f"{group_name}_SubGroup{j + 1}"
                    subgroup_info = group_info.copy()
                    subgroup_info["sequences"] = [(batch_seq_name, batch_seq_info, subgroup_missing_ratio)]

                    if subgroup_name not in subgroups_info:
                        subgroups_info[subgroup_name] = subgroup_info
                    else:
                        subgroups_info[subgroup_name]["sequences"].append(
                            (batch_seq_name, batch_seq_info, subgroup_missing_ratio))

                batch = []  # 清空 batch
            
        if batch:
            random.shuffle(batch)  # 随机打乱顺序
            for j, (batch_seq_name, batch_seq_info) in enumerate(batch):
                # 每个小组对应不同的缺失比例
                subgroup_missing_ratio = missing_ratios[2-j]

                # 分配到小组中
                subgroup_name = f"{group_name}_SubGroup{2-j + 1}"
                subgroup_info = group_info.copy()
                subgroup_info["sequences"] = [(batch_seq_name, batch_seq_info, subgroup_missing_ratio)]

                if subgroup_name not in subgroups_info:
                    subgroups_info[subgroup_name] = subgroup_info
                else:
                    subgroups_info[subgroup_name]["sequences"].append(
                        (batch_seq_name, batch_seq_info, subgroup_missing_ratio))

            batch = []  # 清空 batch            
        subgroups_data[group_name] = subgroups_info

    # 生成小组划分的 JSON 文件
    with open('subgroups_data_rgbdt500.json', 'w') as json_file:
        json.dump(subgroups_data, json_file, indent=4)

    return subgroups_data


# 新增函数：执行缺失操作并保存结果
def execute_missing_operations_and_save_results(subgroups_data):
    result_data = {}

    for group_name, subgroup_info in subgroups_data.items():
        for subgroup_name, subgroup_data in subgroup_info.items():
            missing_function = subgroup_data["missing_function"]
            sequences = subgroup_data["sequences"]

            for seq_name, seq_info, missing_ratio in sequences:
                frames = seq_info["frames"]
                modalities = 3  # RGBDT 有三个模态
                missing_data = None

                if missing_function == "random_missing":
                    missing_data = random_missing(frames, missing_ratio, modalities)
                elif missing_function == "long_term_missing":
                    missing_data = long_term_missing(frames, missing_ratio, modalities)
                elif missing_function == "swap_missing":
                    missing_data = swap_missing(frames, missing_ratio, modalities)
                elif missing_function == "long_term_and_random_missing":
                    missing_data = long_term_and_random_missing(frames, missing_ratio, modalities)
                elif missing_function == "swap_and_random_missing":
                    missing_data = swap_and_random_missing(frames, missing_ratio, modalities)

                if missing_data is not None:
                    result_data[seq_name] = {
                        "frames": frames,
                        "data": missing_data.tolist()
                    }

    # 保存结果到 JSON 文件
    with open('missing_results_rgbdt500.json', 'w') as json_file:
        json.dump(result_data, json_file, indent=4)

# 主程序
if __name__ == "__main__":
    # 设置随机种子以保证可重复性
    random.seed(42)
    np.random.seed(42)
    
    dataset_path = '/media/sqh/DataDisk/RGBDT500/Test/'  # 替换为您的数据集路径
    
    print(f"正在读取 RGBDT500 数据集...")
    dataset = read_rgbdt_dataset(dataset_path)
    print(f"读取完成，共 {len(dataset)} 个序列")
    
    # 划分数据集为五组，确定每个大组的缺失函数
    print("正在划分大组...")
    groups = split_and_shuffle_dataset(dataset)
    print(f"大组划分完成，共 {len(groups)} 个大组")

    # 划分大组为小组，确定每个小组的缺失比例
    print("正在划分小组...")
    subgroups = split_group_subgroups(groups)
    print(f"小组划分完成")

    # 执行缺失操作并保存结果
    print("正在执行缺失操作并保存结果...")
    execute_missing_operations_and_save_results(subgroups)
    print("完成！结果已保存到 missing_results_rgbdt500.json")
