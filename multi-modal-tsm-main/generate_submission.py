import argparse
import csv
import torch
import torchvision
import numpy as np
import os  # 添加os模块导入
from torch.nn import functional as F
from ops.dataset import TSNDataSet
from ops.models import TSN
from ops.transforms import *
from ops import dataset_config_for_pred as dataset_config

# Windows 多进程支持
if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    
    # options
    parser = argparse.ArgumentParser(description="TSM testing on the full validation set")
    parser.add_argument('dataset', type=str)
    parser.add_argument('--weights', type=str, default=None)
    parser.add_argument('--test_segments', type=int, default=25)
    parser.add_argument('--full_res', default=False, action="store_true",
                        help='use full resolution 256x256 for test as in Non-local I3D')
    parser.add_argument('--test_crops', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--workers', default=0, type=int, metavar='N',  # 修改为0以避免多进程问题
                        help='number of data loading workers (0 to disable multiprocessing)')
    parser.add_argument('--test_list', type=str, default=None)
    parser.add_argument('--csv_file', type=str, default='submission.csv')
    parser.add_argument('--softmax', default=False, action="store_true")
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--crop_fusion_type', type=str, default='avg')
    parser.add_argument('--gpus', nargs='+', type=int, default=None)
    parser.add_argument('--img_feature_dim', type=int, default=256)
    parser.add_argument('--pretrain', type=str, default='imagenet')
    parser.add_argument('--test_file', type=str, default='H:\\Data\\test_set\\test_videofolder.txt')
    args = parser.parse_args()

    def parse_shift_option_from_log_name(log_name):
        if 'shift' in log_name:
            strings = log_name.split('_')
            for i, s in enumerate(strings):
                if 'shift' in s:
                    break
            return True, int(strings[i].replace('shift', '')), strings[i + 1]
        else:
            return False, None, None

    # Load the model
    is_shift, shift_div, shift_place = parse_shift_option_from_log_name(args.weights)
    if 'RGB' in args.weights:
        args.modality = 'RGB'
    elif 'RTD' in args.weights:
        args.modality = 'RTD'
    else:
        args.modality = 'Flow'
    this_arch = args.weights.split('TSM_')[1].split('_')[2]

    # 修复：使用正确的数据集配置函数
    if args.dataset == 'mmvpr':
        if args.modality == 'RTD':
            num_class, _, args.val_list, args.root_path, args.root_data_depth, args.root_data_ir, prefix, prefix_ir, prefix_depth = dataset_config.return_mmvpr(args.modality)
        else:
            num_class, _, args.val_list, args.root_path, prefix = dataset_config.return_mmvpr(args.modality)
    else:
        # 其他数据集的处理（如果需要）
        raise NotImplementedError(f'数据集 {args.dataset} 暂不支持')

    # 规范化路径以确保Windows兼容性
    if hasattr(args, 'root_path'):
        args.root_path = os.path.normpath(args.root_path)
    if hasattr(args, 'root_data_ir'):
        args.root_data_ir = os.path.normpath(args.root_data_ir)
    if hasattr(args, 'root_data_depth'):
        args.root_data_depth = os.path.normpath(args.root_data_depth)

    net = TSN(num_class, args.test_segments if is_shift else 1, args.modality,
              base_model=this_arch,
              consensus_type=args.crop_fusion_type,
              img_feature_dim=args.img_feature_dim,
              pretrain=args.pretrain,
              is_shift=is_shift, shift_div=shift_div, shift_place=shift_place,
              non_local='_nl' in args.weights)

    # 修复：添加 weights_only=False 参数
    checkpoint = torch.load(args.weights, weights_only=False)
    checkpoint = checkpoint['state_dict']
    base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.items())}
    
    # 添加键名转换逻辑以处理模型结构不匹配问题
    converted_dict = {}
    for k, v in base_dict.items():
        # 处理键名中的 '.net.' 层次问题
        if 'layer2.net' in k or 'layer3.net' in k or 'layer4.net' in k:
            # 将 'layer2.net.0' 转换为 'layer2.0'
            parts = k.split('.')
            new_parts = []
            i = 0
            while i < len(parts):
                if i + 1 < len(parts) and parts[i] in ['layer2', 'layer3', 'layer4'] and parts[i+1] == 'net':
                    new_parts.append(parts[i])
                    i += 2  # 跳过 'net'
                else:
                    new_parts.append(parts[i])
                    i += 1
            converted_key = '.'.join(new_parts)
            converted_dict[converted_key] = v
        else:
            converted_dict[k] = v
    
    replace_dict = {'base_model.classifier.weight': 'new_fc.weight',
                    'base_model.classifier.bias': 'new_fc.bias'}
    for k, v in replace_dict.items():
        if k in converted_dict:
            converted_dict[v] = converted_dict.pop(k)
    
    # 使用strict=False允许部分键不匹配
    net.load_state_dict(converted_dict, strict=False)
    
    input_size = net.scale_size if args.full_res else net.input_size
    cropping = torchvision.transforms.Compose([
        GroupScale(net.scale_size),
        GroupCenterCrop(input_size),
    ])

    if args.modality != 'RGBDiff':
        normalize = GroupNormalize(net.input_mean, net.input_std)
    else:
        normalize = IdentityTransform()

    if args.modality in ['RGB', 'IR', 'Depth']:
        data_length = 1
    elif args.modality in ['RTD']:
        data_length = 1  # 对于多模态，每个模态单独加载
    elif args.modality in ['Flow', 'RGBDiff']:
        data_length = 5

    # 读取测试列表文件，获取视频名称
    video_names = []
    with open(args.test_file, 'r') as f:
        for line in f:
            video_names.append(line.strip().split()[0])  # 假设第一列是视频名称

    # 根据模态创建适当的数据集加载器
    if args.modality == 'RTD':
        # 多模态情况
        dataset = TSNDataSet(args.root_path, args.root_data_ir, args.root_data_depth, 
                           list_file=args.test_file, 
                           num_segments=args.test_segments,
                           new_length=data_length,
                           modality=args.modality,
                           image_tmpl=prefix, 
                           image_tmpl_ir=prefix_ir, 
                           image_tmpl_depth=prefix_depth,
                           test_mode=True,
                           transform=torchvision.transforms.Compose([
                               cropping,
                               Stack(roll=(this_arch in ['BNInception', 'InceptionV3'])),
                               ToTorchFormatTensor(div=(this_arch not in ['BNInception', 'InceptionV3'])),
                               normalize,
                           ]))
    else:
        # 单模态情况
        dataset = TSNDataSet(args.root_path, None, None, 
                           list_file=args.test_file, 
                           num_segments=args.test_segments,
                           new_length=data_length,
                           modality=args.modality,
                           image_tmpl=prefix, 
                           test_mode=True,
                           transform=torchvision.transforms.Compose([
                               cropping,
                               Stack(roll=(this_arch in ['BNInception', 'InceptionV3'])),
                               ToTorchFormatTensor(div=(this_arch not in ['BNInception', 'InceptionV3'])),
                               normalize,
                           ]))

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # 如果没有指定GPU，使用CPU
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpus is not None else 'cpu')
    net = net.to(device)
    net.eval()

    output = []

    def eval_video(video_data, net, this_test_segments):
        with torch.no_grad():
            i, data = video_data
            batch_size = args.batch_size
            num_crop = args.test_crops

            # 修复：正确处理数据结构
            # dataset.py的get方法返回 (process_data, label) 元组
            frames, _ = data  # 只提取帧数据，忽略标签
            
            # 将数据移动到正确的设备
            frames = frames.to(device)
            
            # 根据模态处理数据形状
            # 修改eval_video函数中的RTD模态处理部分
            if args.modality == 'RGB':
                sample_length = 3
                frames = frames.view(-1, sample_length, frames.size(2), frames.size(3))
            elif args.modality == 'RTD':
                # 恢复为简单处理，不进行额外的形状修改
                # 让数据集加载器和模型内部处理多模态融合
                pass
            elif args.modality == 'Flow':
                sample_length = 10
                frames = frames.view(-1, sample_length, frames.size(2), frames.size(3))
            
            # 处理时间移位模块的情况
            if is_shift:
                # 对所有模态统一处理时间移位
                if args.modality in ['RGB', 'Flow']:
                    frames = frames.view(batch_size * num_crop, this_test_segments, -1, 
                                        frames.size(2), frames.size(3))
                elif args.modality == 'RTD':
                    # RTD模态特殊处理
                    frames = frames.view(batch_size * num_crop, this_test_segments, -1, 
                                        frames.size(2), frames.size(3))
            
            # 进行模型推理
            rst = net(frames)
            
            # 融合结果
            rst = rst.reshape(batch_size, num_crop, -1).mean(1)

            if args.softmax:
                rst = F.softmax(rst, dim=1)  # 计算概率分布

            return i, rst.data.cpu().numpy().copy()

    # 处理每个视频
    for i, data in enumerate(data_loader):
        rst = eval_video((i, data), net, args.test_segments)
        output.append(rst[1])  # 只存储预测结果
        print(f'Processing video {i+1}/{len(data_loader)}')

    # 计算预测类别 - 修改为获取Top-5预测结果
    video_pred_top5 = []
    for x in output:
        # 获取概率最高的5个类别索引（从高到低排序）
        top5_indices = np.argsort(x)[0][::-1][:5]
        # 将索引转换为字符串并用空格连接
        # 修改这一行，移除加1操作
        top5_str = ' '.join([str(idx) for idx in top5_indices])  # 保持类别编号为0-19
        video_pred_top5.append(top5_str)
    
    # 使用读取的视频名称保存结果
    with open(args.csv_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['video_id', 'prediction'])  # 修改列名为要求的格式
        for vid_name, pred in zip(video_names[:len(video_pred_top5)], video_pred_top5):
            csvwriter.writerow([vid_name, pred])
    
    print(f'Predictions saved to {args.csv_file}')