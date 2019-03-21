from datasets import base_dataset
import os
import numpy as np
from datasets import pascal2012
from datasets import base_dataset as bd
from PIL import Image
import ann_utils as au
import misc as ms
from datasets.base_datasets import pascal
class PascalStageBase(pascal2012.PascalOriginal):
    def __init__(self, root, split, transform_function, stage, **dataset_options):
        super().__init__(root, split, transform_function)
        self.n_stages = self.n_classes / 5

        assert stage <= self.n_stages
        if stage == 0:
            classes = [0]
        else:

            increments =  np.arange(0, self.n_classes, 5) 
            s_class = increments[stage-1] + 1
            e_class = increments[stage] + 1

            if split == "train":
                classes = [0] + np.arange(s_class, e_class).tolist()

            elif split == "val":
                classes = np.arange(e_class).tolist()
        
        self.classes = classes
        self.n_classes = e_class
        self.collate_fn = base_dataset.collate_fn_0_4

        self.split = split

        # Get all labels
        name2class = self.get_name2class()

        img_names_new = []
        cls_names_new = []
        mask_names_new = []
        for img_name, cls_name, mask_name in zip(self.img_names, 
                                                 self.cls_names, 
                                                 self.mask_names):
            if np.in1d(name2class[img_name], classes).sum() > 0:
                img_names_new += [img_name]
                cls_names_new += [cls_name]
                mask_names_new += [mask_name]

        self.img_names = img_names_new
        self.cls_names = cls_names_new
        self.mask_names = mask_names_new

        self.collate_fn = bd.collate_fn_0_4
        self.resize_transform = base_dataset.Resize()


    def get_name2class(self):
        fname = self.path+"/name2class_{}.pkl".format(self.split)

        if not os.path.exists(fname):
            name2class = {}
            for i, img_path in enumerate(self.img_names):
                name2class[img_path] = np.setdiff1d(np.unique(self.load_mask(i)),[0,255])
  
            ms.save_pkl(fname, name2class)

        name2class = ms.load_pkl(fname)

        return name2class

     
    def __getitem__(self, index):
        # Image       

        img_path = self.img_names[index]
        image_pil = Image.open(img_path).convert('RGB')


        W, H = image_pil.size
        mask_path = self.mask_names[index]
        maskObjects = np.array(pascal.load_mask(mask_path))
        maskClass = self.load_mask(index)
        # print(np.unique(maskObjects))

        annList = pascal.mask2annList(maskClass, maskObjects, image_id=img_path,
                                      classes=self.classes)
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


class PascalStage1(PascalStageBase):
    def __init__(self, root, split, transform_function, **dataset_options):
        super().__init__(root, split, transform_function, stage=1)


class PascalStage2(PascalStageBase):
    def __init__(self, root, split, transform_function, **dataset_options):
        super().__init__(root, split, transform_function, stage=2)

class PascalStage3(PascalStageBase):
    def __init__(self, root, split, transform_function, **dataset_options):
        super().__init__(root, split, transform_function, stage=3)

class PascalStage4(PascalStageBase):
    def __init__(self, root, split, transform_function, **dataset_options):
        super().__init__(root, split, transform_function, stage=4)