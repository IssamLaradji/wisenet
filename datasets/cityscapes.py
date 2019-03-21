import os

import numpy as np
import torch
from PIL import Image
from torch.utils import data
import misc as ms
from torchvision import transforms
from scipy.ndimage.morphology import distance_transform_edt
from datasets import base_dataset
root = '/mnt/datasets/public/segmentation/cityscapes'

import glob
import ann_utils as au

# name2category = {"pedestrian":1, "car":2}

class CityScapesObject(data.Dataset):
    def __init__(self, root, split, transform_function=None, 
                 **dataset_options):
        super().__init__()


        quality = "fine"
        self.path = "/mnt/datasets/public/segmentation/cityscapes/"
       
        # self.annList_path = annList_path
        
        root = "/mnt/datasets/public/segmentation/cityscapes/"
        list_path = "/mnt/datasets/public/issam/gta5/cityscapes_list/%s.txt" % split
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.files = []

        self.quality = quality
        self.split = split

        for name in self.img_ids:
            img_file = os.path.join(root, "leftImg8bit/%s/%s" % (self.split, name))
            label_file = os.path.join(root, "gtFine/%s/%s" % (self.split, name))
            maskClasses_file = label_file.replace("_leftImg8bit.png", 
                                          "_gtFine_labelIds.png")
            maskObjects_file =  label_file.replace("_leftImg8bit.png", 
                                          "_gtFine_instanceIds.png")
            assert os.path.exists(img_file)
            assert os.path.exists(maskClasses_file)
            assert os.path.exists(maskObjects_file)
            self.files.append({
                "img": img_file,
                "maskClasses": maskClasses_file,
                "maskObjects": maskObjects_file,
                "image_id": name
            })



        self.transform_function = transform_function()

        self.name2id_dict = base_dataset.cityscapes_name2id_dict

        self.categoryList = ["person", "car", "traffic ligt",
                             "traffic sign"]
        self.n_classes = len(self.categoryList) + 1
        
        self.collate_fn = base_dataset.collate_fn_0_4
        self.resize_transform = base_dataset.Resize()
        # self.ratio = (1./3.)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image_pil = Image.open(datafiles["img"]).convert('RGB')
        w,h = image_pil.size
        maskClasses = Image.open(datafiles["maskClasses"])
        maskObjects = Image.open(datafiles["maskObjects"])

        maskClasses = np.asarray(maskClasses, np.float32)
        maskClasses = base_dataset.maskClasses_subset(self.name2id_dict,
                                        self.categoryList, maskClasses)
        # ms.images(maskClasses==1)  
        maskObjects = np.asarray(maskObjects,np.float32 )
        maskObjects[maskObjects<=255] = 0
        
        image_id = datafiles["image_id"]

        # ms.images(maskObjects==0)
        annList = base_dataset.maskClassesObjects2annList(
                                   self.name2id_dict, 
                                   self.categoryList,
                                   image_id, 
                                   maskClasses, 
                                   maskObjects)

        # ms.images(image_pil, annList, pretty=True)
        targets = au.annList2targets(annList)
        image_pil, targets = self.resize_transform(image_pil, targets)
        
        if self.transform_function is not None:
            image = self.transform_function(image_pil)
        import ipdb; ipdb.set_trace()  # breakpoint fe7574b3 //

        counts = np.array([n_pedestrians, n_cars])
        assert np.unique(maskObjects)[-1] == counts.sum()

        points = transforms.functional.to_pil_image(points[:,:,None].astype("uint8"))
        proposals_path = "/mnt/datasets/public/issam/Cityscapes/ProposalsSharp/"


        return {"images":image,
            "annList":annList,
            "targets":targets,
            "meta":{"index":index, "image_id":image_id,
                    "split":self.split,
                    "shape":(1, 3, h, w)}}

    def __len__(self):
        return len(self.files)
        # self.ratio = (1./3.)

    def __getitem__(self, index):
        img_path = self.img_names[index]
        mask_path = self.img_inst[index]
        

        name = os.path.basename(img_path)

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)

        shape_original = image.size
        image = image.resize((1200, 600),Image.BILINEAR)
        mask = mask.resize((1200, 600),Image.NEAREST)

        shape_new = image.size

        w_ratio, h_ratio = np.array(shape_original) / np.array(shape_new)
        w_min = 100./w_ratio
        h_min = 100./h_ratio

        mask = np.array(mask)

        maskClasses = np.zeros(mask.shape,int)
        maskObjects = np.zeros(mask.shape,int)
        maskVoid = np.zeros(mask.shape, int)
        points = np.zeros(mask.shape, int)
    
        uniques = np.unique(mask)

        # Pedestrians
        ind = (mask>=24*1000) & (mask<25*1000)
        maskClasses[ind] = 1

        n_pedestrians = 0

        for i, u in enumerate(uniques[(uniques>=24*1000) & 
                             (uniques<25*1000)]):
            seg_ind = mask==u
            r, c = np.where(seg_ind)

            if (r.max()-r.min()) < h_min or (c.max()-c.min()) < w_min:
                maskVoid[seg_ind] = 1
                continue
            n_pedestrians += 1
            maskObjects[seg_ind] = n_pedestrians
            dist = distance_transform_edt(seg_ind)
            yx = np.unravel_index(dist.argmax(), dist.shape)
            points[yx] = 1


        # Cars
        ind = (mask>=26*1000) & (mask<27*1000)
        maskClasses[ind] = 2

        n_cars = 0
        for i, u in enumerate(uniques[(uniques>=26*1000) & 
                             (uniques<27*1000)]):
            
            seg_ind = mask==u
            r, c = np.where(seg_ind)
            if (r.max()-r.min()) < h_min or (c.max()-c.min()) < w_min:
                maskVoid[seg_ind] = 1
                continue

            n_cars += 1
            maskObjects[seg_ind] = n_cars + n_pedestrians
            dist = distance_transform_edt(seg_ind)
            yx = np.unravel_index(dist.argmax(), dist.shape)     
            points[yx] = 2


        counts = np.array([n_pedestrians, n_cars])
        assert np.unique(maskObjects)[-1] == counts.sum()

        points = transforms.functional.to_pil_image(points[:,:,None].astype("uint8"))
        proposals_path = "/mnt/datasets/public/issam/Cityscapes/ProposalsSharp/"
