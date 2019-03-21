from datasets import base_dataset
from datasets import base_dataset as bd
import os
import numpy as np
from datasets.base_datasets import pascal

from PIL import Image
import ann_utils as au
class PascalOriginal(pascal.Pascal2012):
    def __init__(self, root, split, transform_function, **dataset_options):
        super().__init__(root, split, transform_function)
        pascal_root = os.path.join(self.path)

        pascal_path = pascal_root + '/VOC2012/'
        if split == "train":
            self.img_names  = []
            self.mask_names = []
            self.cls_names = []

            for name in  self.data_dict["train_imgNames"]:
                name_img = os.path.join(pascal_path, 'JPEGImages/' + name + '.jpg')
                name_mask =  os.path.join(pascal_path, 'SegmentationObject/' +  name + '.png')
                name_cls =  os.path.join(pascal_path, 'SegmentationClass/' + name + '.png')

                if not os.path.exists(name_cls):
                    continue
                self.img_names += [name_img]
                self.mask_names += [name_mask]
                self.cls_names += [name_cls]

            self.img_names.sort()
            self.cls_names.sort()
            self.mask_names.sort()

            self.img_names = np.array(self.img_names)
            self.cls_names = np.array(self.cls_names)
            self.mask_names = np.array(self.mask_names)

        self.collate_fn = bd.collate_fn_0_4
        self.resize_transform = base_dataset.Resize()
            
    def __getitem__(self, index):
        # Image       

        img_path = self.img_names[index]
        image_pil = Image.open(img_path).convert('RGB')


        W, H = image_pil.size
        mask_path = self.mask_names[index]
        maskObjects = np.array(pascal.load_mask(mask_path))
        maskClass = self.load_mask(index)
        # print(np.unique(maskObjects))
        annList = pascal.mask2annList(maskClass, maskObjects, image_id=img_path)
        # ms.images(image_pil, annList, pretty=True)
        targets = au.annList2targets(annList)
        image_pil, targets = self.resize_transform(image_pil, targets)
        
        if self.transform_function is not None:
            image = self.transform_function(image_pil)

        return {"images":image,
            "annList":annList,
            "targets":targets,
            "meta":{"index":index, "image_id":img_path,
                    "split":self.split,
                    "shape":(1, 3, H, W)}}