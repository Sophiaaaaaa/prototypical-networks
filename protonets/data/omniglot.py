import os
import sys
import glob

from functools import partial

import numpy as np
from PIL import Image

import torch
from torchvision.transforms import ToTensor

from torchnet.dataset import ListDataset, TransformDataset
from torchnet.transform import compose

import protonets
from protonets.data.base import convert_dict, CudaTransform, EpisodicBatchSampler, SequentialBatchSampler

OMNIGLOT_DATA_DIR  = os.path.join(os.path.dirname(__file__), '../../data/omniglot')
OMNIGLOT_CACHE = { }

# 将图片放在一个字典中
def load_image_path(key, out_field, d):
    d[out_field] = Image.open(d[key])
    return d

# 将d[key]转化为tensor
def convert_tensor(key, d):
    d[key] = 1.0 - torch.from_numpy(np.array(d[key], np.float32, copy=False)).transpose(0, 1).contiguous().view(1, d[key].size[0], d[key].size[1])
    return d

# 将图像旋转rot角度
def rotate_image(key, rot, d):
    d[key] = d[key].rotate(rot)
    return d

# 统一图片大小
def scale_image(key, height, width, d):
    d[key] = d[key].resize((height, width))
    return d
 
# 获取每个类对应的图片
def load_class_images(d):
    if d['class'] not in OMNIGLOT_CACHE:
		  # 获取路径
        alphabet, character, rot = d['class'].split('/')
        image_dir = os.path.join(OMNIGLOT_DATA_DIR, 'data', alphabet, character)
        # 获取指定路径下的所有图片
        class_images = sorted(glob.glob(os.path.join(image_dir, '*.png')))
        if len(class_images) == 0:
            raise Exception("No images found for omniglot class {} at {}. Did you run download_omniglot.sh first?".format(d['class'], image_dir))

         # ListDataset从图片列表中加载数据
        # 数据处理，包括旋转、创建字典、规范图片大小，转化为tensor
        image_ds = TransformDataset(ListDataset(class_images),
                                    compose([partial(convert_dict, 'file_name'),
                                             partial(load_image_path, 'file_name', 'data'),
                                             partial(rotate_image, 'data', float(rot[3:])),
                                             partial(scale_image, 'data', 28, 28),
                                             partial(convert_tensor, 'data')]))

			# 所有数据放到一个batch内
        loader = torch.utils.data.DataLoader(image_ds, batch_size=len(image_ds), shuffle=False)
        
        # 取一个数据
        for sample in loader:
            # 将图片数据写入Omniglot_cache中
            OMNIGLOT_CACHE[d['class']] = sample['data']
            break # only need one sample because batch size equal to dataset length
    
    # 返回类及类中的一个数据组成的字典
    return { 'class': d['class'], 'data': OMNIGLOT_CACHE[d['class']] }

# 取每个类的一个episode的数据
def extract_episode(n_support, n_query, d):
    # data: N x C x H x W
    n_examples = d['data'].size(0)
    # n_query是-1则代表将除去support之外的所有数据都当作query
    if n_query == -1:
        n_query = n_examples - n_support
    
    # 数据随机打乱之后，取support数据之和+query数据之和的数据量，返回的是index
    example_inds = torch.randperm(n_examples)[:(n_support+n_query)]
    # 返回support数据的下标
    support_inds = example_inds[:n_support]
    # 返回query数据的下标
    query_inds = example_inds[n_support:]
    
    # 形成support数据和query数据
    xs = d['data'][support_inds]
    xq = d['data'][query_inds]

    # 获取每个类的support set和query set
    return {
        'class': d['class'],
        'xs': xs,
        'xq': xq
    }

def load(opt, splits):
    split_dir = os.path.join(OMNIGLOT_DATA_DIR, 'splits', opt['data.split'])

    ret = { }
    for split in splits:
	      # 获取n_way
        if split in ['val', 'test'] and opt['data.test_way'] != 0:
            n_way = opt['data.test_way']
        else:
            n_way = opt['data.way']
        # 获取support的数量
        if split in ['val', 'test'] and opt['data.test_shot'] != 0:
            n_support = opt['data.test_shot']
        else:
            n_support = opt['data.shot']
        # 获取query的数量
        if split in ['val', 'test'] and opt['data.test_query'] != 0:
            n_query = opt['data.test_query']
        else:
            n_query = opt['data.query']
        # 获取episode
        if split in ['val', 'test']:
            n_episodes = opt['data.test_episodes']
        else:
            n_episodes = opt['data.train_episodes']
        # 定义了三个函数：class字典，加载类的一张图片，取一个episode的数据
        transforms = [partial(convert_dict, 'class'), # 取key是class的字典内容
                      load_class_images, # 取一个类中的一条数据
                      partial(extract_episode, n_support, n_query)] # 获取每个类的support和query

        if opt['data.cuda']:
            transforms.append(CudaTransform())

        transforms = compose(transforms)

        class_names = []
        # 按照分割数据集的方式，获取相应的所有类名
        with open(os.path.join(split_dir, "{:s}.txt".format(split)), 'r') as f:
            for class_name in f.readlines():
                class_names.append(class_name.rstrip('\n'))
        
        # 对所有类划分support和query数据集
        ds = TransformDataset(ListDataset(class_names), transforms)

        if opt['data.sequential']:
            sampler = SequentialBatchSampler(len(ds))
        else:
            sampler = EpisodicBatchSampler(len(ds), n_way, n_episodes)

        # use num_workers=0, otherwise may receive duplicate episodes
        ret[split] = torch.utils.data.DataLoader(ds, batch_sampler=sampler, num_workers=0)

    return ret
