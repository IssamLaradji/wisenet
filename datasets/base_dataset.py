import torch 
import re
import sys
import threading
import traceback
import os
import ann_utils as au
import time
import atexit
import collections
from addons.pycocotools import mask as maskUtils
# from torch._six import string_classes, int_classes, FileNotFoundError
# from torch._six import container_abcs
import numpy as np
import scipy
from torch._six import string_classes, int_classes, FileNotFoundError
# from models.base_models.retina_container.maskrcnn_benchmark.structures.bounding_box import BoxList

from maskrcnn_benchmark.structures.bounding_box import BoxList



class SharpProposals:
    def __init__(self, batch):
        # if dataset_name == "pascal":
        proposal_fname = batch["meta"]["proposal_fname"][0]

            
        _, _, self.h, self.w = batch["images"].shape

        if "resized" in batch and batch["resized"].item() == 1:
            name_resized = self.proposals_path + "{}_{}_{}.json".format(batch["name"][0], 
                                                                        self.h, self.w)
            
            if not os.path.exists(name_resized):
                proposals = ms.load_json(proposal_fname)
                json_file = loop_and_resize(self.h, self.w, proposals)
                ms.save_json(name_resized, json_file)
        else:
            name_resized = proposal_fname
        # name_resized = name         
        
        proposals = ms.load_json(name_resized)
        self.proposals = sorted(proposals, key=lambda x:x["score"], 
                                reverse=True)         

    def __getitem__(self, i):
        encoded = self.proposals[i]["segmentation"]
        proposal_mask = maskUtils.decode(encoded)
        
        return {"mask":proposal_mask, 
                "category_id": 1,
                "score":self.proposals[i]["score"]}


    def __len__(self):
        return len(self.proposals)


    def sharpmask2psfcn_proposals(self):
        import ipdb; ipdb.set_trace()  # breakpoint 102ed333 //

        pass
from skimage.transform import resize
def loop_and_resize(h, w, proposals):
    proposals_resized = []
    n_proposals = len(proposals)
    for i in range(n_proposals):
        print("{}/{}".format(i, n_proposals))
        prop = proposals[i]
        seg = prop["segmentation"]
        proposal_mask = maskUtils.decode(seg)
        # proposal_mask = resize(proposal_mask*255, (h, w), order=0).astype("uint8")

        if not proposal_mask.shape == (h, w):
            proposal_mask = (resize(proposal_mask*255, (h, w), order=0)>0).astype(int)
            seg = maskUtils.encode(np.asfortranarray(proposal_mask).astype("uint8"))
            seg["counts"] = seg["counts"].decode("utf-8")

            prop["segmentation"] = seg 
            proposals_resized += [proposals[i]]

        else:
            proposals_resized += [proposals[i]]
        
    return proposals_resized


def collate_fn_0_4(batch, level=0):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if isinstance(batch[0], torch.Tensor):
        out = None
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int_classes):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], list) and level==1:
        return batch[0]
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], BoxList):
      return batch
    elif isinstance(batch[0], collections.Mapping):
        return {key: collate_fn_0_4([d[key] for d in batch],  level=level+1) for key in batch[0]}
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [collate_fn_0_4(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))

# input and output size
############################
multi_scale_inp_size = [np.array([320, 320], dtype=np.int),
                        np.array([352, 352], dtype=np.int),
                        np.array([384, 384], dtype=np.int),
                        np.array([416, 416], dtype=np.int),
                        np.array([448, 448], dtype=np.int),
                        np.array([480, 480], dtype=np.int),
                        np.array([512, 512], dtype=np.int),
                        np.array([544, 544], dtype=np.int),
                        np.array([576, 576], dtype=np.int),
                        # np.array([608, 608], dtype=np.int),
                        ]   # w, h
                        
# multi_scale_out_size = [multi_scale_inp_size[0] / 32,
#                         multi_scale_inp_size[1] / 32,
#                         multi_scale_inp_size[2] / 32,
#                         multi_scale_inp_size[3] / 32,
#                         multi_scale_inp_size[4] / 32,
#                         multi_scale_inp_size[5] / 32,
#                         multi_scale_inp_size[6] / 32,
#                         multi_scale_inp_size[7] / 32,
#                         multi_scale_inp_size[8] / 32,
#                         # multi_scale_inp_size[9] / 32,
#                         ]   # w, h
inp_size = np.array([416, 416], dtype=np.int)   # w, h
out_size = inp_size / 32
from torchvision import transforms
mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])



def resize_image(image_original, min_dim=800, max_dim=1024, padding=True):
    """
    Resizes an image keeping the aspect ratio.

    min_dim: if provided, resizes the image such that it's smaller
        dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't
        exceed this value.
    padding: If true, pads image with zeros so it's size is max_dim x max_dim

    Returns:
    image: the resized image
    window: (y1, x1, y2, x2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
    """
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image_original.shape[:2]
    window = (0, 0, h, w)
    scale = 1

    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    # Does it exceed max dim?
    if max_dim:
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max
    # Resize image and mask
    if scale != 1:
        images_resized = scipy.misc.imresize(image_original, (round(h * scale), round(w * scale)))
    # Need padding?
    if padding:
        # Get new height and width
        h, w = images_resized.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        images_resized = np.pad(images_resized, padding, mode='constant', constant_values=0)
        window_box = (top_pad, left_pad, h + top_pad, w + left_pad)


    images_resized = transforms.Compose([
              transforms.ToTensor(),
              transforms.Normalize(*mean_std)])(images_resized)


    image_meta = {"image":images_resized,
                  "shape_original":image_original.shape, 
                   "window_box":window_box,
                   "scale":scale}

    return image_meta


import misc as ms
def maskClassesObjects2annList(name2id_dict, 
                               categoryList,
                               image_id, 
                               maskClasses, 
                               maskObjects):
  height, width = maskClasses.shape
  # n_objects = int(maskObjects[maskObjects!=255].max())
  object_uniques = np.unique(maskObjects)
  object_uniques = object_uniques[object_uniques!=0]


  annList = []
  for i in range(len(object_uniques)):

    obj_id = object_uniques[i]
    binmask = (maskObjects == obj_id).astype("uint8")

    segmentation = maskUtils.encode(np.asfortranarray(ms.t2n(binmask).squeeze())) 
    segmentation["counts"] = segmentation["counts"].decode("utf-8")
    
    uniques = np.unique(binmask*maskClasses)
    uniques = uniques[uniques!=0]
    if len(uniques) == 0:
      continue


    if len(uniques) > 1:
      for u in uniques:
        best = 0
        results = ((binmask*maskClasses) == u).sum()
        if best < results:
          category_id = u
          best = results


    # assert len(uniques) == 1
    else:
      category_id = uniques[0]
    if category_id == 255:
      continue
    annList += [{"segmentation":segmentation,
                 "iscrowd":0,
                 "bbox":maskUtils.toBbox(segmentation).tolist(),
                 "area":int(maskUtils.area(segmentation)),
                 "height":height,
                 "width":width,
                 "image_id":image_id,
                 "category_id":int(category_id)}]

  return annList

def maskClasses_subset(name2id_dict, categoryList, maskClasses):
    new_maskClasses = np.zeros(maskClasses.shape)

    for c, category in enumerate(categoryList):
        ind = maskClasses == name2id_dict[category]
        new_maskClasses[ind] = c + 1

    return new_maskClasses.astype(int)


def maskClasses_rgb_subset(name2id_dict, categoryList, maskClasses):
    new_maskClasses = np.ones(maskClasses.shape[:2])*255
    maskClasses = maskClasses[:,:,::-1]
    for c, category in enumerate(categoryList):
      
      ind = np.all(maskClasses ==  name2id_dict[category], axis=2) 
      new_maskClasses[ind] = c

    return new_maskClasses.astype(int)

cityscapes_seg_categoryList = ["road", "sidewalk", "building", "wall", "fence",
                               "pole", "traffic light", "traffic sign",
                               "vegetation", "terrain", "sky",
                               "person", "rider", "car", 
                               "truck",
                               "bus", "train", "motorcycle", 
                               "bicycle"]
gta5_seg_categoryList  = cityscapes_seg_categoryList

# Obtained from paper
synthia_seg_categoryList =  ["road", "sidewalk", "building", 
                          "traffic light", "traffic sign",
                               "vegetation", "sky",
                               "person", "rider", "car",
                               "bus",  "motorcycle", "bicycle"]


cityscapes_name2id_dict = {'unlabeled':0,
                           'ego vehicle':1,
                           'rectification border':2,
                           'out of roi':3,
                           'static':4,
                            'dynamic':5,
                           'ground':6,
                           'road':7,
                           'sidewalk':8,
                           'parking':9 ,
                           'rail track':10 ,
                            'building' :11,
                            'wall':12,
                          'fence':13,
                           'guard rail' :14,
                          'bridge'  :15,
                           'tunnel' :16,
                            'pole'   :17,
                             'polegroup'  :18,
                              'traffic light':19,
                               'traffic sign' :20,
                                'vegetation':21,
                                 'terrain' :22,
                                  'sky'  :23,
                                   'person':24,
                                    'rider':25,
                                     'car'  :26,
                                     'truck'  :27,
                                     'bus'  :28,
                                     'caravan'  :29,
                                     'trailer' :30,
                                     'train'  :31,
                                      'motorcycle'  :32,
                                      'bicycle'   :33,
                                      'license plate'   :-1}


synthia_name2id_dict = {'void':0,
                           'sky':1,
                           'building':2,
                           'road':3,
                           'sidewalk':4,
                            'fence':5,
                           'vegetation':6,
                           'pole':7,
                           'car':8,
                           'traffic sign':9 ,
                           'person':10 ,
                            'bicycle' :11,
                            'motorcycle':12,
                          'parking':13,
                           'road' :14,
                          'traffic light'  :15,
                           'terrain' :16,
                            'rider'   :17,
                             'truck'  :18,
                              'bus':19,
                              'train':20,
                              'wall':21,
                              'labemarking':22}


viper_seg_categoryList = ["road", "sidewalk", "building", "fence",
                                "traffic light", "traffic sign",
                               "vegetation", "terrain", "sky",
                               "person",  "car", 
                               "truck",
                               "bus", "train", "motorcycle", 
                               "bicycle"]

viper_name2id_dict = {'unlabeled':(0,0,0),
                           'ambiguous':(111,74,0),
                           'sky':(70,130,180),
                           'road':(128,64,128),
                           'sidewalk':(244,35,232),
                            'railtrack':(230,150,140),
                           'terrain':(152,251,152),
                           'tree':(87,182,35),
                           'vegetation':(35,142,35),
                           'building':(70,70,70),
                           'infrastructure':(153,153,153),
                            'fence' :(190,153,153),
                            'billboard':(150,20,20),
                          'traffic light':(250,170,30),
                           'traffic sign' :(220,220,0),
                          'mobile barrier'  :(180,180,100),
                           'fire hydrant' :(173,153,153),
                            'chair'   :(168,153,153),
                             'trash'  :(81,0,21),
                              'trashcan':(81,0,81),
                               'person' :(220,20,60),
                                'animal':(255,0,0),
                                 'bicycle' :(119,11,32),
                                  'motorcycle'  :(0,0,230),
                                   'car':(0,0,142),
                                    'van':(0,80,100),
                                     'bus'  :(0,60,100),
                                     'truck'  :(0,0,70),
                                     'trailer'  :(0,0,90),
                                     'train'  :(0,80,100),
                                     'plane' :(0,100,100),
                                     'boat'  :(50,0,90)}



# viper_name2id_dict = {'unlabeled':0,
#                            'ambiguous':1,
#                            'sky':2,
#                            'road':3,
#                            'sidewalk':4,
#                             'railtrack':5,
#                            'terrain':6,
#                            'tree':7,
#                            'vegetation':8,
#                            'building':9 ,
#                            'infrastructure':10 ,
#                             'fence' :11,
#                             'billboard':12,
#                           'traffic light':13,
#                            'traffic sign' :14,
#                           'mobile barrier'  :15,
#                            'fire hydrant' :16,
#                             'chair'   :17,
#                              'trash'  :18,
#                               'trashcan':19,
#                                'person' :20,
#                                 'animal':21,
#                                  'bicycle' :22,
#                                   'motorcycle'  :23,
#                                    'car':24,
#                                     'van':25,
#                                      'bus'  :26,
#                                      'truck'  :27,
#                                      'trailer'  :28,
#                                      'train'  :29,
#                                      'plane' :30,
#                                      'boat'  :31}
# 0,unlabeled,0,0,0,0,255,0
# 1,ambiguous,111,74,0,0,255,0
# 2,sky,70,130,180,1,0,0
# 3,road,128,64,128,1,1,0
# 4,sidewalk,244,35,232,1,2,0
# 5,railtrack,230,150,140,0,255,0
# 6,terrain,152,251,152,1,3,0
# 7,tree,87,182,35,1,4,0
# 8,vegetation,35,142,35,1,5,0
# 9,building,70,70,70,1,6,0
# 10,infrastructure,153,153,153,1,7,0
# 11,fence,190,153,153,1,8,0
# 12,billboard,150,20,20,1,9,0
# 13,trafficlight,250,170,30,1,10,1
# 14,trafficsign,220,220,0,1,11,0
# 15,mobilebarrier,180,180,100,1,12,0
# 16,firehydrant,173,153,153,1,13,1
# 17,chair,168,153,153,1,14,1
# 18,trash,81,0,21,1,15,0
# 19,trashcan,81,0,81,1,16,1
# 20,person,220,20,60,1,17,1
# 21,animal,255,0,0,0,255,0
# 22,bicycle,119,11,32,0,255,0
# 23,motorcycle,0,0,230,1,18,1
# 24,car,0,0,142,1,19,1
# 25,van,0,80,100,1,20,1
# 26,bus,0,60,100,1,21,1
# 27,truck,0,0,70,1,22,1
# 28,trailer,0,0,90,0,255,0
# 29,train,0,80,100,0,255,0
# 30,plane,0,100,100,0,255,0
# 31,boat,50,0,90,0,255,0

gta_name2id_dict = cityscapes_name2id_dict





# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random

import torch
import torchvision
from torchvision.transforms import functional as F
from yacs.config import CfgNode as CN
cfg = CN()
cfg.INPUT = CN()
# Size of the smallest side of the image during training
cfg.INPUT.MIN_SIZE_TRAIN = 800 # 800
# Maximum size of the side of the image during training
cfg.INPUT.MAX_SIZE_TRAIN = 1333
# Size of the smallest side of the image during testing
cfg.INPUT.MIN_SIZE_TEST = 800
# Maximum size of the side of the image during testing
cfg.INPUT.MAX_SIZE_TEST = 1333
# Values to be used for image normalization
cfg.INPUT.PIXEL_MEAN = [102.9801, 115.9465, 122.7717]
# Values to be used for image normalization
cfg.INPUT.PIXEL_STD = [1., 1., 1.]
# Convert image to BGR format (for Caffe2 models), in range 0-255
cfg.INPUT.TO_BGR255 = True

def build_transforms(cfg, is_train=True):
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        flip_prob = 0.5  # cfg.INPUT.FLIP_PROB_TRAIN
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        flip_prob = 0

    to_bgr255 = cfg.INPUT.TO_BGR255
    normalize_transform = Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD, to_bgr255=to_bgr255
    )

    transform = Compose(
        [
            Resize(min_size, max_size),
            RandomHorizontalFlip(flip_prob),
            ToTensor(),
            normalize_transform,
        ]
    )
    return transform

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):
    def __init__(self, min_size=800, max_size=1333):
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = self.min_size
        max_size = self.max_size
        
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image, target):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        target = target.resize(image.size)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.hflip(image)
            target = target.transpose(0)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        return F.to_tensor(image), target


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target
